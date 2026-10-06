import pytest

from core.config import settings

from neurons.validators.src.services.redis_service import DUPLICATED_MACHINE_SET
from neurons.validators.src.services.task.checks.duplicate_executor import DuplicateExecutorCheck
from neurons.validators.src.services.task.messages import DuplicateExecutorMessages as Msg

from tests.helpers import build_context_config, build_services, build_state


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
