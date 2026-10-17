import json
import logging
import time

import pytest

from core.config import settings

from neurons.validators.src.services.redis_service import DUPLICATED_MACHINE_SET, GPU_UUID_HOTKEYS_SEEN
from neurons.validators.src.services.task.checks.duplicate_executor import DuplicateExecutorCheck
from neurons.validators.src.services.task.messages import DuplicateExecutorMessages as Msg
from neurons.validators.src.services.task.result_handler import ResultHandler

from tests.helpers import build_context_config, build_services, build_state, default_executor


class FakeRedis:
    """Redis double backed by per-key sets.

    Mirrors how compute_client writes duplicate pairs under DUPLICATED_MACHINE_SET
    and how the check reads them back, so a key-name divergence between writer and
    reader makes the dual-registration test fail.
    """

    def __init__(self) -> None:
        self.sets: dict[str, set[str]] = {}
        self.calls: list[tuple[str, str]] = []

    async def sadd(self, key: str, elem: str) -> None:
        self.sets.setdefault(key, set()).add(elem)

    async def is_elem_exists_in_set(self, key: str, elem: str) -> bool:
        self.calls.append((key, elem))
        return elem in self.sets.get(key, set())


def _make_ctx(redis_service, context_factory, miner_hotkey: str = "miner-hotkey"):
    return context_factory(
        services=build_services(redis=redis_service),
        config=build_context_config(),
        state=build_state(),
        miner_hotkey=miner_hotkey,
    )


@pytest.mark.asyncio
async def test_unique_executor_passes(context_factory):
    redis_service = FakeRedis()
    ctx = _make_ctx(redis_service, context_factory)

    result = await DuplicateExecutorCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.UNIQUE.reason
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_duplicate_observe_mode_passes_without_penalty(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "DUPLICATE_EXECUTOR_DRY_RUN", True)
    redis_service = FakeRedis()
    ctx = _make_ctx(redis_service, context_factory)
    await redis_service.sadd(DUPLICATED_MACHINE_SET, f"{ctx.miner_hotkey}:{ctx.executor.uuid}")

    result = await DuplicateExecutorCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DUPLICATE_OBSERVED.reason
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_duplicate_enforce_mode_fails_and_clears(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "DUPLICATE_EXECUTOR_DRY_RUN", False)
    redis_service = FakeRedis()
    ctx = _make_ctx(redis_service, context_factory)
    elem = f"{ctx.miner_hotkey}:{ctx.executor.uuid}"
    await redis_service.sadd(DUPLICATED_MACHINE_SET, elem)

    result = await DuplicateExecutorCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.DUPLICATE.reason
    assert result.updates.get("clear_verified_job_info") is True
    # the check must read the same Redis key the writer (compute_client) populates
    assert redis_service.calls == [(DUPLICATED_MACHINE_SET, elem)]


@pytest.mark.asyncio
async def test_dual_registration_both_fail_in_enforce_mode(context_factory, monkeypatch):
    # writer side: one executor announced by two hotkeys -> both pairs land in the set
    monkeypatch.setattr(settings, "DUPLICATE_EXECUTOR_DRY_RUN", False)
    redis_service = FakeRedis()
    ctx_a = _make_ctx(redis_service, context_factory, miner_hotkey="hotkey-a")
    ctx_b = _make_ctx(redis_service, context_factory, miner_hotkey="hotkey-b")
    await redis_service.sadd(DUPLICATED_MACHINE_SET, f"hotkey-a:{ctx_a.executor.uuid}")
    await redis_service.sadd(DUPLICATED_MACHINE_SET, f"hotkey-b:{ctx_b.executor.uuid}")

    result_a = await DuplicateExecutorCheck().run(ctx_a)
    result_b = await DuplicateExecutorCheck().run(ctx_b)

    assert result_a.passed is False
    assert result_b.passed is False
    assert result_a.event.reason_code == Msg.DUPLICATE.reason
    assert result_b.event.reason_code == Msg.DUPLICATE.reason


# DAH-4546: GPU UUIDs seen under another miner hotkey within the window.
class GpuSeenRedis(FakeRedis):
    def __init__(self) -> None:
        super().__init__()
        self.hashes: dict[str, dict[str, str]] = {}

    async def hget(self, key: str, field: str):
        return self.hashes.get(key, {}).get(field)

    async def hset(self, key: str, field: str, value: str):
        self.hashes.setdefault(key, {})[field] = value

    # the verified-record writes of a failed cycle; not under test here
    async def set_verified_job_info(self, **kwargs):
        pass

    async def clear_verified_job_info(self, **kwargs):
        pass


def _gpu_ctx(redis_service, context_factory, miner_hotkey: str, executor_uuid: str = "executor-123"):
    executor = default_executor().model_copy(update={"uuid": executor_uuid})
    return context_factory(
        executor=executor,
        services=build_services(redis=redis_service),
        config=build_context_config(),
        state=build_state(gpu_uuids="GPU-1,GPU-2"),
        miner_hotkey=miner_hotkey,
    )


def _seen(redis_service, hotkey: str, days_ago: float):
    ts = time.time() - days_ago * 86400
    for uuid in ("GPU-1", "GPU-2"):
        redis_service.hashes.setdefault(GPU_UUID_HOTKEYS_SEEN, {})[uuid] = json.dumps({hotkey: ts})


async def _run_cycle(redis_service, ctx, cycle_passes: bool = True):
    """The check, then ResultHandler persisting the cycle: the hash is written only when the cycle passes."""
    result = await DuplicateExecutorCheck().run(ctx)
    await ResultHandler(redis_service)._persist_verification_data(
        miner_hotkey=ctx.miner_hotkey,
        executor_id=ctx.executor.uuid,
        verified_job_info={},
        context=ctx.model_copy(update=result.updates),
        success=result.passed and cycle_passes,
    )
    return result


@pytest.mark.asyncio
async def test_same_hotkey_new_executor_id_passes(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "CROSS_HOTKEY_GPU_UUID_ENFORCEMENT_ENABLED", True)
    redis_service = GpuSeenRedis()
    first = await _run_cycle(redis_service, _gpu_ctx(redis_service, context_factory, "hotkey-a", "exec-old"))
    again = await _run_cycle(redis_service, _gpu_ctx(redis_service, context_factory, "hotkey-a", "exec-new"))

    assert first.passed is True and again.passed is True
    assert again.event.reason_code == Msg.UNIQUE.reason
    assert set(json.loads(redis_service.hashes[GPU_UUID_HOTKEYS_SEEN]["GPU-1"])) == {"hotkey-a"}


@pytest.mark.asyncio
async def test_other_hotkey_in_window_logged_only_when_off(context_factory, monkeypatch, caplog):
    monkeypatch.setattr(settings, "CROSS_HOTKEY_GPU_UUID_ENFORCEMENT_ENABLED", False)
    redis_service = GpuSeenRedis()
    _seen(redis_service, "hotkey-a", days_ago=1)

    with caplog.at_level(logging.WARNING):
        result = await _run_cycle(redis_service, _gpu_ctx(redis_service, context_factory, "hotkey-b"))

    assert result.passed is True
    assert result.event.reason_code == Msg.UNIQUE.reason
    assert "GPU UUIDs seen under another miner hotkey" in caplog.text
    # the relister is not recorded, so the first hotkey keeps passing
    assert set(json.loads(redis_service.hashes[GPU_UUID_HOTKEYS_SEEN]["GPU-1"])) == {"hotkey-a"}


@pytest.mark.asyncio
async def test_other_hotkey_in_window_fails_when_on(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "CROSS_HOTKEY_GPU_UUID_ENFORCEMENT_ENABLED", True)
    redis_service = GpuSeenRedis()
    _seen(redis_service, "hotkey-a", days_ago=1)

    result = await _run_cycle(redis_service, _gpu_ctx(redis_service, context_factory, "hotkey-b"))

    assert result.passed is False
    assert result.event.reason_code == Msg.CROSS_HOTKEY_GPU_UUID.reason == "GPU_UUID_SEEN_UNDER_OTHER_HOTKEY"
    assert result.updates.get("clear_verified_job_info") is True
    assert result.fatal is True


@pytest.mark.asyncio
async def test_other_hotkey_older_than_window_passes(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "CROSS_HOTKEY_GPU_UUID_ENFORCEMENT_ENABLED", True)
    monkeypatch.setattr(settings, "CROSS_HOTKEY_GPU_UUID_WINDOW_DAYS", 7)
    redis_service = GpuSeenRedis()
    _seen(redis_service, "hotkey-a", days_ago=8)

    result = await _run_cycle(redis_service, _gpu_ctx(redis_service, context_factory, "hotkey-b"))

    assert result.passed is True
    # the stale entry is dropped and the new hotkey recorded
    assert set(json.loads(redis_service.hashes[GPU_UUID_HOTKEYS_SEEN]["GPU-1"])) == {"hotkey-b"}


@pytest.mark.asyncio
async def test_node_failing_a_later_check_records_nothing(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "CROSS_HOTKEY_GPU_UUID_ENFORCEMENT_ENABLED", True)
    redis_service = GpuSeenRedis()

    result = await _run_cycle(redis_service, _gpu_ctx(redis_service, context_factory, "hotkey-a"), cycle_passes=False)

    assert result.passed is True
    assert GPU_UUID_HOTKEYS_SEEN not in redis_service.hashes
