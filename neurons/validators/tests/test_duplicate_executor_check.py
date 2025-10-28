import pytest

from neurons.validators.src.services.task.checks.duplicate_executor import DuplicateExecutorCheck
from neurons.validators.src.services.task.messages import DuplicateExecutorMessages as Msg

from tests.helpers import build_context_config, build_services, build_state


class DummyRedis:
    def __init__(self, duplicates: set[str]):
        self.duplicates = duplicates
        self.calls: list[tuple[str, str]] = []

    async def is_elem_exists_in_set(self, key: str, elem: str) -> bool:
        self.calls.append((key, elem))
        return elem in self.duplicates


@pytest.mark.parametrize(
    "duplicates,expected_pass,expected_reason,expect_clear",
    [
        (set(), True, Msg.UNIQUE.reason, False),
        ({"miner-hotkey:executor-123"}, False, Msg.DUPLICATE.reason, True),
    ],
)
@pytest.mark.asyncio
async def test_duplicate_executor_check(duplicates, expected_pass, expected_reason, expect_clear, context_factory):
    redis_service = DummyRedis(duplicates)
    services = build_services(redis=redis_service)
    config = build_context_config()
    state = build_state()

    ctx = context_factory(services=services, config=config, state=state)
    result = await DuplicateExecutorCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    expected_elem = f"{ctx.miner_hotkey}:{ctx.executor.uuid}"
    assert redis_service.calls == [("duplicated_machine:set", expected_elem)]

    if expect_clear:
        assert result.updates.get("clear_verified_job_info") is True
    else:
        assert "clear_verified_job_info" not in result.updates
