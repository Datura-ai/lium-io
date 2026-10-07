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


