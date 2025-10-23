import pytest

from neurons.validators.src.services.task.checks.banned_gpu import BannedGpuCheck

from tests.helpers import build_context_config, build_services, build_state


class DummyRedis:
    def __init__(self, banned: list[str]):
        self._banned = banned

    async def get_banned_guids(self) -> list[str]:
        return self._banned


@pytest.mark.parametrize(
    "gpu_uuids,banned_list,expected_pass,expected_reason,expect_clear",
    [
        (None, [], True, "GPU_UUID_EMPTY", False),
        ("abc123", ["abc123"], False, "GPU_BANNED", True),
        ("abc123,def456", ["zzz"], True, "GPU_BANNED_CHECK_OK", False),
    ],
)
@pytest.mark.asyncio
async def test_banned_gpu_check(
    gpu_uuids,
    banned_list,
    expected_pass,
    expected_reason,
    expect_clear,
    context_factory,
):
    services = build_services(redis=DummyRedis(banned_list))
    config = build_context_config()
    state = build_state(specs={}, gpu_uuids=gpu_uuids)

    ctx = context_factory(services=services, config=config, state=state)
    result = await BannedGpuCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason
    if expect_clear:
        assert result.updates.get("clear_verified_job_info") is True
    else:
        assert "clear_verified_job_info" not in result.updates
