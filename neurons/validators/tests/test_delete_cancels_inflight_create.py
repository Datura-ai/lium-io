"""DAH-2728 — a delete must cancel the create it raced, not assume the pod is absent.

A delete arriving mid-create used to see no container yet ("No such container"), report the pod
deleted, and let the create finish minutes later — the orphan then held the host ports and blocked
the next paying rental.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest
import services.docker_service as ds_module
from payload_models.payloads import (
    ContainerCreateRequest,
    ContainerDeleteRequest,
    PayloadPortMapping,
    WorkloadKind,
)
from services.docker_service import DockerService, _CreateCancelledByDelete, inflight_creates
from services.miner_service import MinerService, _bypasses_renting_in_progress


@pytest.fixture
def svc() -> DockerService:
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
    )


@pytest.fixture
def miner_service() -> MinerService:
    return MinerService(
        ssh_service=Mock(),
        task_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        backend_client=MagicMock(),
        file_encrypt_service=MagicMock(),
    )


def _payload(**over) -> ContainerCreateRequest:
    base = dict(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:1.0.0",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=2,
        storage_limit_gb=1,
        available_ports=[PayloadPortMapping(internal_port=20001, external_port=20001)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )
    base.update(over)
    return ContainerCreateRequest(**base)


@pytest.mark.asyncio
async def test_delete_during_create_marks_the_create_cancelled(miner_service: MinerService) -> None:
    payload = _payload()
    seen_by_create: dict[str, bool] = {}

    async def fake_route(routed: ContainerCreateRequest) -> str:
        # A delete for the same pod lands right here, while the create is still running.
        seen_by_create["delete_found_the_create"] = inflight_creates.cancel(routed.pod_id)
        seen_by_create["checkpoint"] = inflight_creates.is_cancelled(routed.pod_id)
        return "created"

    miner_service._route_container = fake_route

    assert await miner_service.handle_container(payload) == "created"
    assert seen_by_create == {"delete_found_the_create": True, "checkpoint": True}
    # The registration must not outlive the create, or a later pod id reuse would abort at once.
    assert inflight_creates.is_cancelled(payload.pod_id) is False


def test_delete_with_no_create_running_cancels_nothing() -> None:
    assert inflight_creates.cancel(str(uuid4())) is False


@pytest.mark.parametrize(
    ("workload_kind", "power_restored"),
    [(WorkloadKind.FILLER, True), (WorkloadKind.CUSTOMER_RENTAL, False)],
)
@pytest.mark.asyncio
async def test_abort_restores_only_filler_power(
    svc: DockerService,
    monkeypatch: pytest.MonkeyPatch,
    workload_kind: WorkloadKind,
    power_restored: bool,
) -> None:
    payload = _payload(workload_kind=workload_kind)
    restore_power_limits = AsyncMock()
    monkeypatch.setattr(ds_module, "restore_filler_pod_gpu_power_limits", restore_power_limits)

    with inflight_creates.track(payload.pod_id):
        inflight_creates.cancel(payload.pod_id)
        with pytest.raises(_CreateCancelledByDelete):
            await svc._abort_if_cancelled_by_delete(Mock(), payload, {})

    assert restore_power_limits.await_count == (1 if power_restored else 0)


@pytest.mark.asyncio
async def test_a_create_nobody_cancelled_runs_on(svc: DockerService) -> None:
    payload = _payload()

    with inflight_creates.track(payload.pod_id):
        assert await svc._abort_if_cancelled_by_delete(Mock(), payload, {}) is None


@pytest.mark.parametrize(
    ("workload_kind", "prefix"),
    [(WorkloadKind.CUSTOMER_RENTAL, "pod_"), (WorkloadKind.FILLER, "filler_")],
)
def test_container_name_is_derived_when_the_delete_carries_none(
    svc: DockerService, workload_kind: WorkloadKind, prefix: str
) -> None:
    # A delete racing an in-flight create arrives before the backend knows the name.
    pod_id = str(uuid4())
    payload = ContainerDeleteRequest(
        miner_hotkey="miner", executor_id=str(uuid4()), pod_id=pod_id,
        workload_kind=workload_kind, container_name="",
    )

    assert svc.get_container_name(payload) == f"{prefix}{pod_id}"


def test_a_customer_delete_passes_the_pending_rent_guard_only_while_its_create_runs() -> None:
    # The guard reads the pending-pod flag the create itself set, so declining the delete would
    # leave that create to finish and orphan its container.
    pod_id = str(uuid4())
    delete = ContainerDeleteRequest(
        miner_hotkey="miner", executor_id=str(uuid4()), pod_id=pod_id,
        container_name=f"pod_{pod_id}",
    )

    assert _bypasses_renting_in_progress(delete) is False
    with inflight_creates.track(pod_id):
        assert _bypasses_renting_in_progress(delete) is True


@pytest.mark.asyncio
async def test_a_retried_create_does_not_untrack_the_one_still_running() -> None:
    # The rent retry can put a second create on the same pod; when it leaves, the first must stay
    # cancellable, or the delete goes back to seeing "no container" and the orphan returns.
    pod_id = str(uuid4())

    with inflight_creates.track(pod_id):
        with inflight_creates.track(pod_id):
            pass
        assert inflight_creates.is_running(pod_id) is True
        assert inflight_creates.cancel(pod_id) is True

    assert inflight_creates.is_running(pod_id) is False


def test_a_second_create_does_not_clear_a_cancel_already_raised() -> None:
    pod_id = str(uuid4())

    with inflight_creates.track(pod_id):
        inflight_creates.cancel(pod_id)
        with inflight_creates.track(pod_id):
            assert inflight_creates.is_cancelled(pod_id) is True


@pytest.mark.asyncio
async def test_wait_returns_at_once_when_no_create_runs() -> None:
    """Nothing to wait for: the delete goes straight to its own teardown."""
    pod_id = str(uuid4())

    finished: bool = await inflight_creates.wait_until_done(pod_id, timeout=0.01)

    assert finished is True


@pytest.mark.asyncio
async def test_wait_returns_when_the_create_leaves() -> None:
    """The delete holds until the cancelled create has torn itself down."""
    pod_id = str(uuid4())

    async def create() -> None:
        with inflight_creates.track(pod_id):
            await asyncio.sleep(0.05)

    create_task: asyncio.Task[None] = asyncio.create_task(create())
    await asyncio.sleep(0)  # let the create register itself

    finished: bool = await inflight_creates.wait_until_done(pod_id, timeout=5)

    assert finished is True
    assert inflight_creates.is_running(pod_id) is False
    await create_task


@pytest.mark.asyncio
async def test_wait_gives_up_on_a_create_that_does_not_stop() -> None:
    """A create stuck between checkpoints must not hold the delete forever."""
    pod_id = str(uuid4())

    async def create() -> None:
        with inflight_creates.track(pod_id):
            await asyncio.sleep(5)

    create_task: asyncio.Task[None] = asyncio.create_task(create())
    await asyncio.sleep(0)

    finished: bool = await inflight_creates.wait_until_done(pod_id, timeout=0.05)

    assert finished is False
    create_task.cancel()
