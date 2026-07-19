"""Behavior-class invariants for the docker-service oracle (DAH-2382 PR-1a, group E).

E1-E6 of characterization-oracle-DRAFT.md §E: cancellation, concurrency,
idempotency, side-effect ordering, orphan resources, nondeterminism control.
Unlike groups A-D these encode INVARIANTS, not pure characterization: where
baseline 6be5649f already upholds an invariant the test is plain green; where
it deviates (the D1 deviation list in design.md) the test asserts the DESIRED
behavior under ``xfail(strict=True)`` so it documents the gap and flips loudly
when the fix lands.

Observed on baseline (feeds the oracle README):
- E1: ``CancelledError`` PROPAGATES out of ``create_container`` (both inner
  ``except Exception`` cleanups and the outer funnel are Exception-only), but
  NO cleanup runs and the pending pod LEAKS -> propagation is green, the
  cleanup half is an xfail SPEC test. The executor lock cannot leak at these
  seams: it is scoped inside ``generate_portMappings`` (patched here) and is
  released before any E1 seam runs.
- E2/E3/E4/E5/E6 all HOLD on baseline -> plain green.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from docker_oracle.harness import (
    CreateContainerHarness,
    RecordedCall,
    make_create_request,
    make_delete_request,
    make_docker_service,
    make_executor_info,
    patch_create_happy_path,
    patch_delete_happy_path,
)
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    ContainerDeleted,
    FailedContainerRequest,
)
from services.docker_service import DockerService
from services.rental_docker_sdk import RentalDockerOperationError

_JUPYTER_PORT_MAP: tuple[int, int] = (8888, 30888)

# The three awaited seams of E1: one inside the docker-run inner try, one inside
# the post-run provisioning inner try, one at the finalize redis write.
_CANCEL_SEAMS: list[str] = ["docker_run", "ssh_bootstrap", "finalize_add_rented_pod"]


async def _create(
    svc: DockerService,
    payload: ContainerCreateRequest,
    executor_info: ExecutorSSHInfo,
) -> Any:
    return await svc.create_container(
        payload=payload,
        executor_info=executor_info,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )


def _inject_cancelled_error(
    harness: CreateContainerHarness, monkeypatch: pytest.MonkeyPatch, seam: str
) -> None:
    # arm exactly one awaited step of the create path to raise CancelledError when reached
    if seam == "docker_run":
        harness.docker_client.run_error = asyncio.CancelledError()
    elif seam == "ssh_bootstrap":
        harness.mocks[
            "install_open_ssh_server_and_start_ssh_service_with_rental_docker"
        ].side_effect = asyncio.CancelledError()
    elif seam == "finalize_add_rented_pod":
        monkeypatch.setattr(
            harness.redis, "add_rented_pod", AsyncMock(side_effect=asyncio.CancelledError())
        )
    else:  # pragma: no cover - guards against a typo in the parametrize list
        raise ValueError(f"unknown seam: {seam}")


def _spy_cleanup(
    svc: DockerService,
    monkeypatch: pytest.MonkeyPatch,
    journal: list[RecordedCall],
) -> list[dict[str, Any]]:
    # record every cleanup_failed_container_creation call (also into the shared
    # journal for cross-layer ordering asserts) while delegating to the real method
    calls: list[dict[str, Any]] = []
    real_cleanup = svc.cleanup_failed_container_creation

    async def _wrapped(**kwargs: Any) -> None:
        calls.append(kwargs)
        journal.append(("cleanup.failed_container_creation", {}))
        await real_cleanup(**kwargs)

    monkeypatch.setattr(svc, "cleanup_failed_container_creation", _wrapped)
    return calls


# ---------------------------------------------------------------------------
# E1 — cancellation mid-flight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("seam", _CANCEL_SEAMS)
async def test_create_cancelled_midflight_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch, seam: str
) -> None:
    # Arrange
    svc = make_docker_service()
    harness = patch_create_happy_path(svc, monkeypatch)
    _inject_cancelled_error(harness, monkeypatch, seam)
    payload = make_create_request()
    executor_info = make_executor_info(payload)

    # Act / Assert
    # WHY: cancellation must never be swallowed into a FailedContainerRequest —
    # in py3.11 CancelledError is a BaseException, so the except-Exception
    # funnel does not catch it and it re-raises to the caller (HOLDS on baseline).
    with pytest.raises(asyncio.CancelledError):
        await _create(svc, payload, executor_info)

    # WHY: sanity that the cancel hit AFTER the pending-pod write, i.e. the seam
    # really was mid-flight and not an early return.
    pending_calls = [name for name, _ in harness.redis.calls if name == "add_pending_pod"]
    assert pending_calls == ["add_pending_pod"]


@pytest.mark.asyncio
@pytest.mark.parametrize("seam", _CANCEL_SEAMS)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "D1 deviation #1: create_container has no finally/BaseException handling — a "
        "CancelledError skips both inner except-Exception blocks AND the outer funnel, so "
        "cleanup_failed_container_creation never runs and the pending pod leaks; fixed in "
        "PR-8 (Compensations: LIFO on error and cancellation)"
    ),
)
async def test_create_cancelled_midflight_runs_cleanup_and_removes_pending_pod(
    monkeypatch: pytest.MonkeyPatch, seam: str
) -> None:
    # Arrange
    svc = make_docker_service()
    harness = patch_create_happy_path(svc, monkeypatch)
    cleanup_calls = _spy_cleanup(svc, monkeypatch, harness.journal)
    _inject_cancelled_error(harness, monkeypatch, seam)
    payload = make_create_request()
    executor_info = make_executor_info(payload)

    # Act
    with pytest.raises(asyncio.CancelledError):
        await _create(svc, payload, executor_info)

    # Assert
    # WHY: cancellation midway through resource acquisition must not orphan the
    # container/volume on the miner host nor leak the pending-pod redis entry
    # (desired invariant — SPEC test, no compliant baseline exists).
    assert cleanup_calls, "cleanup_failed_container_creation must run on cancellation"
    removed = [name for name, _ in harness.redis.calls if name == "remove_pending_pod"]
    assert removed, "remove_pending_pod must be awaited on cancellation"


# ---------------------------------------------------------------------------
# E2 — concurrent creates on distinct pods, one service instance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_create_distinct_pods_no_shared_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: ONE service instance, ONE harness (single shared journal). The
    # port planner seam returns a port keyed off the executor_id ARGUMENT, so
    # any cross-request state bleed would surface as swapped ports/names in the
    # results; the sleep(0) forces the two creates to actually interleave.
    svc = make_docker_service()
    harness = patch_create_happy_path(svc, monkeypatch)
    payload_a = make_create_request()
    payload_b = make_create_request()
    executor_a = make_executor_info(payload_a)
    executor_b = make_executor_info(payload_b)
    port_by_executor = {payload_a.executor_id: 20001, payload_b.executor_id: 20002}

    async def _ports_for_request(
        miner_hotkey: str, executor_id: str, *args: Any, **kwargs: Any
    ) -> tuple[list[tuple[int, int, int]], None]:
        await asyncio.sleep(0)
        port = port_by_executor[executor_id]
        return [(22, port, port)], None

    harness.mocks["generate_portMappings"].side_effect = _ports_for_request

    # Act
    result_a, result_b = await asyncio.gather(
        _create(svc, payload_a, executor_a),
        _create(svc, payload_b, executor_b),
    )

    # Assert
    # WHY: one facade instance served both requests concurrently — each result
    # must carry ITS OWN pod identity, names and ports (no cross-talk).
    assert isinstance(result_a, ContainerCreated)
    assert isinstance(result_b, ContainerCreated)
    assert result_a.pod_id == payload_a.pod_id
    assert result_b.pod_id == payload_b.pod_id
    assert result_a.container_name == f"pod_{payload_a.pod_id}"
    assert result_b.container_name == f"pod_{payload_b.pod_id}"
    assert result_a.volume_name == f"volume_{payload_a.pod_id}"
    assert result_b.volume_name == f"volume_{payload_b.pod_id}"
    assert result_a.port_maps == [(22, 20001)]
    assert result_b.port_maps == [(22, 20002)]

    # WHY: the shared journal must show BOTH pods' full redis lifecycle and both
    # docker runs — neither request may swallow or overwrite the other's records.
    pending_pod_ids = sorted(
        kwargs["pod_id"] for name, kwargs in harness.redis.calls if name == "add_pending_pod"
    )
    assert pending_pod_ids == sorted([payload_a.pod_id, payload_b.pod_id])
    rented = {
        kwargs["pod_id"]: kwargs["container_name"]
        for name, kwargs in harness.redis.calls
        if name == "add_rented_pod"
    }
    assert rented == {
        payload_a.pod_id: f"pod_{payload_a.pod_id}",
        payload_b.pod_id: f"pod_{payload_b.pod_id}",
    }
    run_container_names = {spec.name for spec in harness.docker_client.run_specs}
    assert run_container_names == {f"pod_{payload_a.pod_id}", f"pod_{payload_b.pod_id}"}


# ---------------------------------------------------------------------------
# E3 — delete idempotency (backend delivery is at-least-once)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_container_is_idempotent_when_container_already_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    svc = make_docker_service()
    harness = patch_delete_happy_path(svc, monkeypatch)
    payload = make_delete_request()
    # make_executor_info only reads executor_id, shared by create/delete payloads
    executor_info = make_executor_info(payload)
    keypair = Mock(ss58_address="validator-hotkey")

    # Act: first delete succeeds normally
    first = await svc.delete_container(
        payload=payload, executor_info=executor_info, keypair=keypair, private_key="encrypted"
    )

    # Arrange the retry: the container is now gone. The real SDK wraps docker-py's
    # NotFound into RentalDockerOperationError carrying dockerd's "No such
    # container" text (rental_docker_sdk._call_api); delete detects "missing" by
    # that phrase (_is_missing_docker_container_error), not by exception type.
    harness.docker_client.stop_error = RentalDockerOperationError(
        "Docker SDK stop failed: 404 Client Error: Not Found "
        f'("No such container: {payload.container_name}")'
    )
    harness.docker_client.remove_error = RentalDockerOperationError(
        "Docker SDK remove container failed: 404 Client Error: Not Found "
        f'("No such container: {payload.container_name}")'
    )

    # Act: second delete of the same, now-missing container
    second = await svc.delete_container(
        payload=payload, executor_info=executor_info, keypair=keypair, private_key="encrypted"
    )

    # Assert
    # WHY: backend delivery is at-least-once — a retried delete of an
    # already-removed container must be a successful no-op, never an error
    # (DAH-2345), or the backend retries a doomed request and penalizes the miner.
    assert isinstance(first, ContainerDeleted)
    assert isinstance(second, ContainerDeleted)

    # WHY: idempotency must come from tolerating the missing container, not from
    # skipping the work — both deletes must actually attempt stop + remove, and
    # remove_rented_machine must be safe to repeat.
    assert harness.docker_client.stopped_containers == [payload.container_name] * 2
    removed_names = [kwargs["container_name"] for kwargs in harness.docker_client.removed_containers]
    assert removed_names == [payload.container_name] * 2
    rented_machine_removals = [
        name for name, _ in harness.redis.calls if name == "remove_rented_machine"
    ]
    assert rented_machine_removals == ["remove_rented_machine"] * 2


# ---------------------------------------------------------------------------
# E4 — redis-vs-docker side-effect ordering (crash-recovery contract)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_redis_vs_docker_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: mark the (patched) running check in the shared journal so its
    # position is comparable with the redis.* / docker.* entries.
    svc = make_docker_service()
    harness = patch_create_happy_path(svc, monkeypatch)

    async def _running_check_marker(*args: Any, **kwargs: Any) -> bool:
        harness.journal.append(("check.container_running", {}))
        return True

    harness.mocks["check_container_running"].side_effect = _running_check_marker
    payload = make_create_request()
    executor_info = make_executor_info(payload)

    # Act
    result = await _create(svc, payload, executor_info)

    # Assert
    assert isinstance(result, ContainerCreated)
    names = [name for name, _ in harness.journal]
    ordered_probes = [
        "redis.add_pending_pod",
        "docker.run_container",
        "check.container_running",
        "redis.add_rented_pod",
    ]
    # WHY: index() below must be unambiguous — each probe fires exactly once on
    # this happy path.
    for probe in ordered_probes:
        assert names.count(probe) == 1, f"{probe} recorded {names.count(probe)} times"
    # WHY: this order is invisible in the final state but decides crash-recovery:
    # the pending-pod record must exist BEFORE any docker resource is created,
    # and the rented-pod record only AFTER the container is confirmed running.
    probe_positions = [names.index(probe) for probe in ordered_probes]
    assert probe_positions == sorted(probe_positions), f"journal order broken: {names}"


# ---------------------------------------------------------------------------
# E5 — no orphan docker resources on create failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_failure_at_docker_run_cleans_up_container_and_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    svc = make_docker_service()
    harness = patch_create_happy_path(svc, monkeypatch)
    cleanup_calls = _spy_cleanup(svc, monkeypatch, harness.journal)
    harness.docker_client.run_error = RentalDockerOperationError(
        "Docker SDK run container failed: No such image: daturaai/pytorch:1.0.0"
    )
    payload = make_create_request()
    executor_info = make_executor_info(payload)
    container_name = f"pod_{payload.pod_id}"
    volume_name = f"volume_{payload.pod_id}"

    # Act
    result = await _create(svc, payload, executor_info)

    # Assert
    # WHY: the funnel must report the failure at the docker_run step — and the
    # just-created container/volume must not be stranded on the miner host
    # (disk-exhaustion guard), so cleanup runs before the funnel returns.
    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_run"
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0]["container_name"] == container_name
    assert cleanup_calls[0]["volume_name"] == volume_name
    assert cleanup_calls[0]["remove_volume"] is True

    # WHY: cleanup's real sink is the host shell (docker rm -fv + docker volume
    # rm over SSH), not the SDK client — assert at the seam that actually fires.
    assert any(
        f"/usr/bin/docker rm -fv {container_name}" in command
        for command in harness.ssh_client.commands
    )
    assert any(
        f"/usr/bin/docker volume rm {volume_name}" in command
        for command in harness.ssh_client.commands
    )

    # WHY: cleanup must complete BEFORE the funnel's pending-pod removal — the
    # funnel is the last step before the failure is reported to the backend.
    names = [name for name, _ in harness.journal]
    assert names.index("cleanup.failed_container_creation") < names.index(
        "redis.remove_pending_pod"
    )


@pytest.mark.asyncio
async def test_create_failure_at_health_check_cleans_up_container_and_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    svc = make_docker_service()
    harness = patch_create_happy_path(svc, monkeypatch)
    cleanup_calls = _spy_cleanup(svc, monkeypatch, harness.journal)
    harness.mocks["check_container_running"].return_value = False
    payload = make_create_request()
    executor_info = make_executor_info(payload)
    container_name = f"pod_{payload.pod_id}"
    volume_name = f"volume_{payload.pod_id}"

    # Act
    result = await _create(svc, payload, executor_info)

    # Assert
    # WHY: a container that started but is not running is still an orphan-to-be —
    # the sticky step reports container_health_check (not docker_run) and the
    # same cleanup contract applies.
    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "container_health_check"
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0]["container_name"] == container_name
    assert cleanup_calls[0]["volume_name"] == volume_name
    assert cleanup_calls[0]["remove_volume"] is True
    assert any(
        f"/usr/bin/docker rm -fv {container_name}" in command
        for command in harness.ssh_client.commands
    )
    assert any(
        f"/usr/bin/docker volume rm {volume_name}" in command
        for command in harness.ssh_client.commands
    )


# ---------------------------------------------------------------------------
# E6 — secrets are injected per request, not module-global
# ---------------------------------------------------------------------------


def _token_of(jupyter_url: str) -> str:
    return jupyter_url.split("token=")[1]


@pytest.mark.asyncio
async def test_jupyter_token_fresh_per_request_no_global_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange / Act: two fully fresh service+harness pairs, token source NOT
    # patched — the only nondeterminism left is secrets.token_hex itself.
    svc_a = make_docker_service()
    harness_a = patch_create_happy_path(svc_a, monkeypatch, jupyter_port_map=_JUPYTER_PORT_MAP)
    payload_a = make_create_request(enable_jupyter=True)
    result_a = await _create(svc_a, payload_a, make_executor_info(payload_a))

    svc_b = make_docker_service()
    patch_create_happy_path(svc_b, monkeypatch, jupyter_port_map=_JUPYTER_PORT_MAP)
    payload_b = make_create_request(enable_jupyter=True)
    result_b = await _create(svc_b, payload_b, make_executor_info(payload_b))

    # Assert
    # WHY: each request must draw a fresh token from secrets.token_hex(16) —
    # equal tokens across requests would mean a module-global secret leaking
    # between rentals.
    token_a = _token_of(result_a.jupyter_url)
    token_b = _token_of(result_b.jupyter_url)
    assert token_a != token_b
    for token in (token_a, token_b):
        assert len(token) == 32
        assert all(char in "0123456789abcdef" for char in token)

    # WHY: the URL token and the token handed to run_jupyter must be the same
    # secret — a mismatch would serve a Jupyter the renter cannot open.
    assert harness_a.mocks["run_jupyter"].await_args.kwargs["jupyter_token"] == token_a


@pytest.mark.asyncio
async def test_jupyter_token_source_injectable_for_deterministic_goldens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: pin the sole token source (secrets.token_hex as referenced by
    # services.docker_service) — goldens over the success payload rely on this seam.
    svc = make_docker_service()
    patch_create_happy_path(svc, monkeypatch, jupyter_port_map=_JUPYTER_PORT_MAP)
    monkeypatch.setattr(
        "services.docker_service.secrets.token_hex", lambda nbytes: "ab" * 16
    )
    payload = make_create_request(enable_jupyter=True)
    executor_info = make_executor_info(payload)

    # Act
    result = await _create(svc, payload, executor_info)

    # Assert
    # WHY: with the nondeterminism seam pinned, the jupyter_url is byte-stable —
    # host from executor address, port from jupyter_port_map[1], token injected.
    assert isinstance(result, ContainerCreated)
    assert result.jupyter_url == f"http://127.0.0.1:30888/lab?token={'ab' * 16}"
