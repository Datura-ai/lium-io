"""Oracle group B — create_container SUCCESS path (DAH-2382, PR-1a).

Freezes the observable success boundary of DockerService.create_container on
baseline 6be5649f: the full ContainerCreated wire form (B1), the three
ProfilerStep wire shapes as produced by a real run (B2-B4), the ordered
profiler step-name sequence (B5), the Loki "Deployment profile summary"
profile_steps format (B6), the redis operation order incl. the executor lock
(B7), the FILLER-only pending-pod cleanup (B8) and the streaming pub/sub
payload shape (B9). B10/B11 are folded into test_deploy_optimizations.py and
B12 is already covered there (DRAFT §0.5).

Normalization per DRAFT §0: clock-derived durations are masked in goldens,
absent key != null (exact key-set asserts on wire dicts), goldens use the
repo's own --update-snapshot flag.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import services.docker_service as ds_module
from docker_oracle.harness import (
    make_create_request,
    make_docker_service,
    make_executor_info,
    patch_create_happy_path,
)
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    PayloadPortMapping,
    ProfilerStepName,
    WorkloadKind,
)
from services.docker_service import DockerService
from services.redis_service import STREAMING_LOG_CHANNEL

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "docker_oracle"

# Fixed ids so the B1 golden is byte-stable (pod_id must be a valid UUID —
# create_container calls UUID(payload.pod_id) before the port planner).
_B1_EXECUTOR_ID = "00000000-0000-0000-0000-0000000000e1"
_B1_POD_ID = "00000000-0000-0000-0000-0000000000b1"


def _assert_matches_golden(path: Path, serialized: str, update_snapshot: bool) -> None:
    # repo-native golden mechanism (test_custom_dockerfile_build.py::test_A1 pattern):
    # write-on-missing (then fail asking for a re-run), refresh via --update-snapshot,
    # otherwise assert byte-equality.
    path.parent.mkdir(parents=True, exist_ok=True)
    if update_snapshot or not path.exists():
        path.write_text(serialized + "\n", encoding="utf-8")
        if not update_snapshot:
            pytest.fail(
                f"Golden snapshot did not exist; wrote it. Re-run to assert. Path={path}"
            )
        return
    expected = path.read_text(encoding="utf-8").rstrip("\n")
    assert serialized == expected, (
        "docker_service success golden drifted! Refresh with --update-snapshot only for an\n"
        f"INTENDED behavior change. Path={path}\nexpected={expected!r}\nactual={serialized!r}"
    )


async def _create(svc: DockerService, payload: ContainerCreateRequest) -> object:
    # drive the real create_container against the harness seams
    return await svc.create_container(
        payload=payload,
        executor_info=make_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )


# ------------------------------------------------------------------
# B1 — full ContainerCreated golden
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_container_created_full_golden(
    monkeypatch: pytest.MonkeyPatch, update_snapshot: bool
) -> None:
    # Arrange
    svc = make_docker_service()
    patch_create_happy_path(svc, monkeypatch)  # ports [(22,20001,20001)], sizing 10/20
    payload = make_create_request(
        executor_id=_B1_EXECUTOR_ID,
        pod_id=_B1_POD_ID,
        backup_log_id="b1",
        restore_path="/r",
        enable_jupyter=False,
    )

    # Act
    result = await _create(svc, payload)

    # Assert
    # WHY: pin the semantically loaded success fields explicitly (readable failures)
    # before freezing the byte form — incl. the (docker, external) port projection,
    # the fresh-sizing override of the payload limits and the "/root" default path.
    assert isinstance(result, ContainerCreated)
    assert result.workload_kind is WorkloadKind.CUSTOMER_RENTAL
    assert result.container_name == f"pod_{_B1_POD_ID}"
    assert result.volume_name == f"volume_{_B1_POD_ID}"
    assert result.port_maps == [(22, 20001)]
    assert result.backup_log_id == "b1"
    assert result.restore_path == "/r"
    assert result.jupyter_url is None
    assert result.warnings == []
    assert result.storage_limit_gb == 20
    assert result.volume_limit_gb == 10
    assert result.local_volume_path == "/root"
    assert result.profilers, "profilers must be non-empty on success"

    # WHY: the serialized response is what the backend parses — freeze it byte-for-byte
    # with clock-derived durations masked (§0: assert type/relation, never literals).
    data = json.loads(result.model_dump_json())
    for step in data["profilers"]:
        assert isinstance(step["duration"], int) and step["duration"] >= 0, step
        step["duration"] = "<duration_ms>"
    serialized = json.dumps(data, indent=2)
    _assert_matches_golden(
        _FIXTURES_DIR / "container_created_success_golden.json", serialized, update_snapshot
    )


# ------------------------------------------------------------------
# B2/B3/B4 — the three ProfilerStep wire shapes, from a REAL run
# (hand-built instances are shape-tested in test_deploy_optimizations.py:691)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profiler_shape_anchor_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the backend-sent timestamp becomes the anchor step
    svc = make_docker_service()
    patch_create_happy_path(svc, monkeypatch)
    payload = make_create_request(timestamp=1_700_000_000_000)

    # Act
    result = await _create(svc, payload)

    # Assert
    # WHY: the anchor wire shape is exactly {name, timestamp} — no duration/skipped
    # key may appear (absent key != null for the backend parser), and the timestamp
    # is the request value passed through untouched.
    assert isinstance(result, ContainerCreated)
    anchor = result.profilers[0]
    assert anchor.model_dump() == {
        "name": "Requested from backend",
        "timestamp": 1_700_000_000_000,
    }


@pytest.mark.asyncio
async def test_profiler_shape_duration_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: no request timestamp -> the first step is a plain duration step
    svc = make_docker_service()
    patch_create_happy_path(svc, monkeypatch)
    payload = make_create_request()

    # Act
    result = await _create(svc, payload)

    # Assert
    # WHY: a non-skipped step serializes to exactly {name, duration} —
    # skipped=False is omitted from the wire, not sent as false.
    assert isinstance(result, ContainerCreated)
    dumped = result.profilers[0].model_dump()
    assert set(dumped) == {"name", "duration"}
    assert dumped["name"] == "Started in subnet"
    assert isinstance(dumped["duration"], int) and dumped["duration"] >= 0


@pytest.mark.asyncio
async def test_profiler_shape_duration_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: image already present on the executor -> the pull short-circuits
    svc = make_docker_service()
    harness = patch_create_happy_path(svc, monkeypatch)
    payload = make_create_request()
    harness.docker_client.existing_images.add(payload.docker_image)

    # Act
    result = await _create(svc, payload)

    # Assert
    # WHY: a short-circuited step serializes to exactly {name, duration, skipped: true}
    # — and the skip is real: no pull was issued to the docker client.
    assert isinstance(result, ContainerCreated)
    pull = next(p for p in result.profilers if p.name is ProfilerStepName.DOCKER_PULL)
    dumped = pull.model_dump()
    assert set(dumped) == {"name", "duration", "skipped"}
    assert dumped["name"] == "Docker pull step finished"
    assert dumped["skipped"] is True
    assert isinstance(dumped["duration"], int) and dumped["duration"] >= 0
    assert harness.docker_client.pulled_images == []


# ------------------------------------------------------------------
# B5 — ordered profiler name sequence (golden file, not an inline literal)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profiler_name_sequence_golden(
    monkeypatch: pytest.MonkeyPatch, update_snapshot: bool
) -> None:
    # Arrange: happy path, jupyter off, inspector off (harness default), image
    # absent (default), no request timestamp
    svc = make_docker_service()
    patch_create_happy_path(svc, monkeypatch)
    payload = make_create_request()

    # Act
    result = await _create(svc, payload)

    # Assert
    assert isinstance(result, ContainerCreated)
    names = [p.name.value for p in result.profilers]
    # WHY: the anchor is gated on a truthy payload.timestamp — with none sent it
    # must not appear at all (the sequence starts at "Started in subnet").
    assert ProfilerStepName.REQUESTED_FROM_BACKEND.value not in names
    assert names[0] == "Started in subnet"
    # WHY: extraction must not drop, reorder or duplicate a deploy step; the ordered
    # list is frozen as a golden file (a DAH-2440-style added step then shows up as a
    # reviewable golden diff instead of a stale inline literal).
    serialized = json.dumps(names, indent=2)
    _assert_matches_golden(
        _FIXTURES_DIR / "profiler_name_sequence_golden.json", serialized, update_snapshot
    )


# ------------------------------------------------------------------
# B6 — Loki "Deployment profile summary" profile_steps format
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loki_profile_steps_format(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: spy the module logger; request timestamp set so the anchor row
    # exists; inspector off (harness default) so its row reports skipped
    svc = make_docker_service()
    patch_create_happy_path(svc, monkeypatch)
    payload = make_create_request(timestamp=1_700_000_000_000)
    spy_logger = Mock()
    monkeypatch.setattr(ds_module, "logger", spy_logger)

    # Act
    result = await _create(svc, payload)

    # Assert
    assert isinstance(result, ContainerCreated)
    summaries = [
        c
        for c in spy_logger.info.call_args_list
        if c.args and getattr(c.args[0], "message", None) == "Deployment profile summary"
    ]
    # WHY: the Loki summary is the dashboard/classifier surface and is emitted
    # exactly once on the success path.
    assert len(summaries) == 1
    extra = summaries[0].args[0].extra
    profile_steps = extra["profile_steps"]
    assert profile_steps
    # WHY: unlike the ProfilerStep model serializer, the Loki dict-comp emits ALL
    # THREE keys unconditionally — skipped always present, duration_ms nullable.
    for entry in profile_steps:
        assert set(entry) == {"name", "duration_ms", "skipped"}, entry
    # WHY: the anchor row surfaces as duration_ms=null (never dropped) and is not skipped.
    anchor = next(e for e in profile_steps if e["name"] == "Requested from backend")
    assert anchor["duration_ms"] is None
    assert anchor["skipped"] is False
    # WHY: with ENABLE_INSPECTOR off the inspector row must still appear, marked skipped.
    inspector = next(
        e for e in profile_steps if e["name"] == "Inspector collector start step finished"
    )
    assert inspector["skipped"] is True
    # WHY: the total excludes the anchor via `duration or 0` — it must equal the sum
    # of the emitted rows exactly (relation, not literal, per §0 masking rules).
    assert isinstance(extra["total_duration_ms"], int)
    assert extra["total_duration_ms"] == sum(e["duration_ms"] or 0 for e in profile_steps)


# ------------------------------------------------------------------
# B7 — redis operation order for CUSTOMER_RENTAL (executor lock included)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redis_operation_order_customer_rental(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: the executor lock lives INSIDE generate_portMappings, which the
    # harness patches — restore the REAL planner over a small real port set so
    # the lock acquisition lands in the journal (harness fact (b)).
    svc = make_docker_service()
    harness = patch_create_happy_path(svc, monkeypatch)
    monkeypatch.setattr(
        svc, "generate_portMappings", DockerService.generate_portMappings.__get__(svc)
    )
    payload = make_create_request(
        available_ports=[
            PayloadPortMapping(internal_port=port, external_port=port)
            for port in (20000, 20001, 20002)
        ],
    )

    # Act
    result = await _create(svc, payload)

    # Assert
    assert isinstance(result, ContainerCreated)
    tracked = {
        "redis.acquire_executor_lock",
        "redis.add_pending_pod",
        "redis.add_rented_pod",
        "redis.remove_pending_pod",
    }
    redis_ops = [name for name, _ in harness.journal if name in tracked]
    # WHY: crash recovery depends on this exact sequence — lock during port planning,
    # pending marker before any docker work, rented marker only after success; and a
    # successful CUSTOMER_RENTAL never removes its pending pod.
    assert redis_ops == [
        "redis.acquire_executor_lock",
        "redis.add_pending_pod",
        "redis.add_rented_pod",
    ]
    # WHY: the lock is per-executor — it must be taken with this payload's executor_id.
    lock_calls = [
        kwargs for name, kwargs in harness.journal if name == "redis.acquire_executor_lock"
    ]
    assert lock_calls == [{"executor_id": payload.executor_id}]
    # WHY: the pending marker must precede the docker-side run and the rented marker
    # must follow it (the cross-layer ordering the boundary result cannot show).
    op_names = [name for name, _ in harness.journal]
    assert (
        op_names.index("redis.add_pending_pod")
        < op_names.index("docker.run_container")
        < op_names.index("redis.add_rented_pod")
    )
    # WHY: the realtime log drainer task is started exactly once per create, keyed
    # to this pod's identifiers.
    assert harness.mocks["handle_stream_logs"].call_count == 1
    assert harness.mocks["handle_stream_logs"].call_args.kwargs == {
        "miner_hotkey": payload.miner_hotkey,
        "executor_id": payload.executor_id,
        "pod_id": payload.pod_id,
    }


# ------------------------------------------------------------------
# B8 — FILLER-only pending-pod cleanup
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redis_remove_pending_for_filler(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    svc = make_docker_service()
    harness = patch_create_happy_path(svc, monkeypatch)
    payload = make_create_request(workload_kind=WorkloadKind.FILLER)

    # Act
    result = await _create(svc, payload)

    # Assert
    assert isinstance(result, ContainerCreated)
    assert result.container_name == f"filler_{payload.pod_id}"
    removes = [kwargs for name, kwargs in harness.journal if name == "redis.remove_pending_pod"]
    # WHY: only the FILLER success path clears its pending marker — exactly once,
    # with the pod's own identifiers (distinct from CUSTOMER_RENTAL, B7).
    assert removes == [
        {
            "miner_hotkey": payload.miner_hotkey,
            "executor_id": payload.executor_id,
            "pod_id": payload.pod_id,
        }
    ]
    # WHY: the cleanup is a post-success step — it must run only after the rented
    # marker is in place.
    op_names = [name for name, _ in harness.journal]
    assert op_names.index("redis.add_rented_pod") < op_names.index("redis.remove_pending_pod")


# ------------------------------------------------------------------
# B9 — streaming pub/sub payload shape
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_log_publish_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: restore the REAL streaming trio (the harness mocks them) so the
    # drainer publishes through the recording redis; interval 0 keeps the loop fast.
    svc = make_docker_service()
    harness = patch_create_happy_path(svc, monkeypatch)
    for name in ("stream_log", "handle_stream_logs", "finish_stream_logs"):
        monkeypatch.setattr(svc, name, getattr(DockerService, name).__get__(svc))
    monkeypatch.setattr(ds_module, "LOG_STREAM_INTERVAL", 0)

    async def _yielding_check(*args: object, **kwargs: object) -> bool:
        # WHY: every other harness await completes without touching the event loop,
        # so the drainer task would otherwise first run only INSIDE
        # finish_stream_logs' `await log_task` — setting is_realtime_logging back to
        # True after the stop flag was already cleared and spinning forever. One real
        # yield lets the drainer start first (prod always yields on real I/O; the
        # hang is a mock-only artifact, not a production behavior).
        await asyncio.sleep(0)
        return True

    harness.mocks["check_container_running"].side_effect = _yielding_check
    payload = make_create_request()

    # Act (wait_for: a drainer regression must fail loudly, never hang the suite)
    result = await asyncio.wait_for(_create(svc, payload), timeout=30)

    # Assert
    assert isinstance(result, ContainerCreated)
    # WHY: the channel name is the contract with the sole subscriber
    # (compute_client) — pin the constant's byte value, not just identity.
    assert STREAMING_LOG_CHANNEL == "STREAMING_LOG_CHANNEL"
    assert len(harness.redis.published) >= 1
    # WHY: every published message must carry exactly {logs, miner_hotkey,
    # executor_uuid, pod_id} — note the executor key is named executor_uuid on
    # the wire while carrying payload.executor_id.
    for channel, message in harness.redis.published:
        assert channel == STREAMING_LOG_CHANNEL
        assert set(message) == {"logs", "miner_hotkey", "executor_uuid", "pod_id"}
        assert message["miner_hotkey"] == payload.miner_hotkey
        assert message["executor_uuid"] == payload.executor_id
        assert message["pod_id"] == payload.pod_id
    # WHY: each streamed entry is the exact queue record stream_log enqueues —
    # {log_text, log_status, log_tag} tagged container_creation.
    entries = [entry for _, message in harness.redis.published for entry in message["logs"]]
    assert entries
    for entry in entries:
        assert set(entry) == {"log_text", "log_status", "log_tag"}
        assert entry["log_tag"] == "container_creation"
    texts = {entry["log_text"] for entry in entries}
    assert {"Creating docker container", "Created Docker Container"} <= texts
