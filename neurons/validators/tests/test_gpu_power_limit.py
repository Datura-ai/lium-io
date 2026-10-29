import time
from unittest.mock import AsyncMock

import pytest
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

    async def set(self, key: str, value: str) -> None:
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
