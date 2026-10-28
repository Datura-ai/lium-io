import json
from types import SimpleNamespace
import time
from unittest.mock import AsyncMock

import neurons.validators.src.services.gpu_power_limit as gpl
import pytest
from helpers import FakeRedis as SharedFakeRedis
from neurons.validators.src.services.gpu_power_limit import (
    POWER_STATE_CMD,
    GpuPowerRestoreRecord,
    GpuPowerRestoreStateError,
    _pod_index_key,
    _restore_key,
    restore_filler_pod_gpu_power_limits,
)


def _record(gpu_uuid: str, watts: int, executor_id: str = "executor-1") -> str:
    return GpuPowerRestoreRecord(
        gpu_uuid=gpu_uuid, watts=watts, pod_id=POD_ID, executor_id=executor_id, capped_at=time.time()
    ).model_dump_json()


# uuid, current, default, min, max
STATE_CSV = "GPU-a, 130, 400, 100, 400\nGPU-b, 250, 250, 100, 250\n"
POD_ID = "pod-1"


class FakeRun:
    def __init__(self, stdout: str = "", stderr: str = "", exit_status: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_status = exit_status


def fake_ssh(*responses: FakeRun) -> AsyncMock:
    ssh = AsyncMock()
    ssh.run.side_effect = responses
    return ssh


def _set_ok(readback_watts: int, persistence: str = "Enabled") -> list[FakeRun]:
    """SSH responses for one successful verified set: -pm 1, -pl, readback confirming the target."""
    return [FakeRun(), FakeRun(), FakeRun(stdout=f"{readback_watts}.00, {persistence}\n")]


class FakeRedis:
    def __init__(self, initial: dict[str, str] | None = None):
        self.store: dict[str, str] = dict(initial or {})

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "timeout", "nonzero", "partial", "redis", "record_redis", "record", "missing_record", "mismatched_record", "index", "delete", "absent"])
async def test_strict_pod_restore_requires_every_record_to_be_restored(failure):
    # Arrange: real restore records and setters; keep one GPU's failure distinct from a partial restore.
    redis = FakeRedis({_pod_index_key(POD_ID): '["GPU-a", "GPU-b"]',
                       _restore_key("GPU-a"): _record("GPU-a", 400),
                       _restore_key("GPU-b"): _record("GPU-b", 250)})
    if failure == "absent":
        redis.store.clear()
    if failure == "missing_record":
        del redis.store[_restore_key("GPU-a")]
    if failure == "mismatched_record":
        redis.store[_restore_key("GPU-a")] = _record("GPU-b", 250)
    if failure in ("record", "index"):
        redis.store[_restore_key("GPU-a") if failure == "record" else _pod_index_key(POD_ID)] = "broken"
    if failure == "redis":
        redis.get = AsyncMock(side_effect=ConnectionError("unreadable"))
    if failure == "record_redis":
        async def get(key):
            if key.startswith("gpu_power_restore:"):
                raise ConnectionError("unreadable record")
            return redis.store.get(key)
        redis.get = get
    if failure == "delete":
        redis.delete = AsyncMock(side_effect=ConnectionError("cannot clear record"))
    initial_records = redis.store.copy()
    async def run(command, **kwargs):
        if command == POWER_STATE_CMD:
            return FakeRun(stdout=STATE_CSV)
        if " -pl " in command and failure in ("timeout", "nonzero", "partial"):
            if failure == "timeout":
                raise TimeoutError("hung")
            if failure != "partial" or "GPU-b" in command:
                return FakeRun(exit_status=1)
        return FakeRun(stdout="400, Enabled\n" if "GPU-a" in command else "250, Enabled\n")
    ssh = AsyncMock(run=AsyncMock(side_effect=run))
    # Act: strict caller cannot succeed with an unknown or remaining cap.
    if failure not in (None, "absent"):
        error_type = RuntimeError if failure in ("timeout", "nonzero", "partial") else GpuPowerRestoreStateError
        with pytest.raises(error_type) as caught:
            await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, require_complete=True)
        assert type(caught.value) is error_type
        # Assert: failed records and the index survive for a later retry.
        assert _pod_index_key(POD_ID) in redis.store
        assert _restore_key("GPU-b") in redis.store
        if failure == "mismatched_record":
            ssh.run.assert_not_awaited()
            assert redis.store == initial_records
    else:
        assert await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, require_complete=True) == (0 if failure else 2)
        assert not redis.store


@pytest.mark.asyncio
@pytest.mark.parametrize("require_complete", [True, False])
async def test_pod_index_delete_failure_after_successful_restore(require_complete):
    # Arrange: only the final index deletion fails; the GPU record can be cleared.
    redis = FakeRedis({_pod_index_key(POD_ID): '["GPU-a"]', _restore_key("GPU-a"): _record("GPU-a", 400)})
    delete_record = redis.delete

    async def delete(key):
        if key == _pod_index_key(POD_ID):
            raise ConnectionError("cannot clear index")
        await delete_record(key)

    redis.delete = delete
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), *_set_ok(400))
    # Act: strict customer checks fail, while existing best-effort cleanup still returns.
    if require_complete:
        with pytest.raises(GpuPowerRestoreStateError):
            await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID, require_complete=True)
    else:
        assert await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID) == 1
    # Assert: the record was deleted after restoration, but the stale index remains visible.
    assert redis.store == {_pod_index_key(POD_ID): '["GPU-a"]'}


def _power(*watts: int) -> dict[str, gpl.GpuPowerState]:
    return {
        f"GPU-{index}": gpl.GpuPowerState(current_watts=w, min_watts=100, max_watts=450, default_watts=350)
        for index, w in enumerate(watts)
    }


async def _rental_start(monkeypatch, redis, states, pod_id: str = "pod-1") -> int:
    monkeypatch.setattr(gpl, "_query_power_state", AsyncMock(return_value=states))
    return await gpl.raise_low_power_limits_to_default(
        AsyncMock(), "ex-1", list(states), redis=redis, pod_id=pod_id
    )


def _ratio(redis, gpu_uuid: str) -> dict:
    return json.loads(redis.store[gpl.rental_start_ratio_key("ex-1", gpu_uuid)])


@pytest.mark.asyncio
async def test_rental_start_raise_keeps_the_80_percent_tier_and_records_each_pod_gpu(monkeypatch) -> None:
    states = _power(298, 200, 298)
    set_side_by_side = AsyncMock(return_value=1)
    monkeypatch.setattr(gpl, "_set_side_by_side", set_side_by_side)
    monkeypatch.setattr(gpl, "_query_power_state", AsyncMock(return_value=states))
    redis = SharedFakeRedis()

    raised = await gpl.raise_low_power_limits_to_default(
        AsyncMock(), "ex-1", ["GPU-0", "GPU-1"], redis=redis, pod_id="pod-1"
    )

    assert raised == 1
    assert [gpu.gpu_uuid for gpu in set_side_by_side.await_args.args[1]] == ["GPU-1"]  # 85 % is left alone
    assert _ratio(redis, "GPU-0") == {"pod_id": "pod-1", "ratio": 0.8514}
    assert _ratio(redis, "GPU-1") == {"pod_id": "pod-1", "ratio": 1.0}
    assert gpl.rental_start_ratio_key("ex-1", "GPU-2") not in redis.store  # another pod's GPU


@pytest.mark.parametrize(
    ("first_pod", "second_pod", "expected"),
    [
        ("pod-1", "pod-1", {"pod_id": "pod-1", "ratio": 1.0}),  # reboot / edit: the start stands
        ("pod-0", "pod-1", {"pod_id": "pod-1", "ratio": 0.8514}),  # a new rental: its own start
    ],
)
@pytest.mark.asyncio
async def test_rental_start_record_survives_a_reboot_of_the_same_rental(
    monkeypatch, first_pod, second_pod, expected
) -> None:
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    redis = SharedFakeRedis()
    await _rental_start(monkeypatch, redis, _power(350), first_pod)
    await _rental_start(monkeypatch, redis, _power(298), second_pod)  # host cut to 85 % meanwhile
    assert _ratio(redis, "GPU-0") == expected


@pytest.mark.asyncio
async def test_rental_start_of_a_rental_from_before_the_records_records_stock(monkeypatch) -> None:
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    redis = SharedFakeRedis()
    redis.store[gpl.rental_legacy_pod_key("pod-1")] = "1"
    await _rental_start(monkeypatch, redis, _power(298))
    assert _ratio(redis, "GPU-0") == {"pod_id": "pod-1", "ratio": 1.0}


@pytest.mark.asyncio
async def test_rental_start_gpu_missing_its_record_on_a_started_rental_records_stock(monkeypatch) -> None:
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    redis = SharedFakeRedis()
    redis.store[gpl.rental_start_ratio_key("ex-1", "GPU-0")] = json.dumps({"pod_id": "pod-1", "ratio": 1.0})
    await _rental_start(monkeypatch, redis, _power(350, 298))  # GPU-1 cut to 85 % before a reboot
    assert _ratio(redis, "GPU-1") == {"pod_id": "pod-1", "ratio": 1.0}


@pytest.mark.parametrize(("pod_on_host", "ratio"), [(True, 1.0), (False, 0.8514)])
@pytest.mark.asyncio
async def test_rental_start_of_a_pod_already_on_the_host_records_stock(monkeypatch, pod_on_host, ratio) -> None:
    # a rental from before the records, cut to 85 % and rebooted before its first check
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    monkeypatch.setattr(gpl, "_query_power_state", AsyncMock(return_value=_power(298)))
    redis = SharedFakeRedis()
    await gpl.raise_low_power_limits_to_default(
        AsyncMock(), "ex-1", ["GPU-0"], redis=redis, pod_id="pod-1", pod_on_host=pod_on_host
    )
    assert _ratio(redis, "GPU-0") == {"pod_id": "pod-1", "ratio": ratio}


@pytest.mark.parametrize(
    ("watts", "failing_gpus"),
    [
        ((280,), {"GPU-0"}),  # a tier rental without its record would be held at 90 %
        ((350, 280), {"GPU-1"}),  # partial: the tier GPU's write failed
        ((350, 350), {"GPU-1"}),  # stock: an unrecorded GPU could be cut and re-recorded at a reboot
    ],
)
@pytest.mark.asyncio
async def test_rental_start_record_write_failure_fails_the_create_then_a_retry_records(
    monkeypatch, watts, failing_gpus
) -> None:
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    redis = SharedFakeRedis()
    redis.fail_next_set_of = {gpl.rental_start_ratio_key("ex-1", u) for u in failing_gpus}
    with pytest.raises(gpl.GpuPowerRestoreStateError):
        await _rental_start(monkeypatch, redis, _power(*watts))
    await _rental_start(monkeypatch, redis, _power(*watts))  # the retried create
    assert all(_ratio(redis, f"GPU-{i}")["pod_id"] == "pod-1" for i in range(len(watts)))


@pytest.mark.asyncio
async def test_rental_start_record_read_outage_fails_the_create(monkeypatch) -> None:
    monkeypatch.setattr(gpl, "_set_side_by_side", AsyncMock(return_value=0))
    with pytest.raises(gpl.GpuPowerRestoreStateError):
        await _rental_start(monkeypatch, SharedFakeRedis(failing=True), _power(280))


@pytest.mark.asyncio
async def test_raise_without_redis_never_records_or_raises(monkeypatch) -> None:
    monkeypatch.setattr(gpl, "_query_power_state", AsyncMock(side_effect=RuntimeError("nvidia-smi hung")))
    assert await gpl.raise_low_power_limits_to_default(AsyncMock(), "ex-1", None) == 0


@pytest.mark.parametrize(("container_names", "expected"), [(("pod_pod-1",), True), (("pod_pod-2",), False), (None, False)])
@pytest.mark.asyncio
async def test_create_passes_pod_on_host_from_the_prerun_listing(monkeypatch, container_names, expected) -> None:
    import services.docker_service as ds

    raise_mock = AsyncMock(return_value=0)
    monkeypatch.setattr(ds, "restore_tracked_gpu_power_limits", AsyncMock())
    monkeypatch.setattr(ds, "raise_low_power_limits_to_default", raise_mock)
    payload = SimpleNamespace(gpu_uuids=["GPU-0"], executor_id="ex-1", pod_id="pod-1")
    probe = SimpleNamespace(container_names=container_names)
    await ds.DockerService._restore_gpu_power_for_uncapped_pod(
        SimpleNamespace(redis_service=object()), AsyncMock(), payload, probe, {}, record_rental_start=True
    )
    assert raise_mock.await_args.kwargs["pod_on_host"] is expected
