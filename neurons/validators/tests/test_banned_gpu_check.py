import pytest

from neurons.validators.src.services.task.checks.banned_gpu import BannedGpuCheck
from neurons.validators.src.services.task.messages import BannedGpuMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

from tests.helpers import build_context_config, build_services, build_state


def build_rented_data_with_banned_guids(banned_guids: list[str]) -> RentedExecutorsResponse:
    """Create RentedExecutorsResponse with specified banned GUIDs."""
    return RentedExecutorsResponse(
        executors={},
        banned_guids=banned_guids,
    )


@pytest.mark.parametrize(
    "gpu_uuids,banned_list,expected_pass,expected_reason,expect_clear",
    [
        (None, [], True, Msg.UUID_EMPTY.reason, False),
        ("abc123", ["abc123"], False, Msg.GPU_BANNED.reason, True),
        ("abc123,def456", ["zzz"], True, Msg.GPU_ALLOWED.reason, False),
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
    services = build_services()
    config = build_context_config()
    rented_data = build_rented_data_with_banned_guids(banned_list)
    state = build_state(specs={}, gpu_uuids=gpu_uuids, rented_data=rented_data)

    ctx = context_factory(services=services, config=config, state=state)
    result = await BannedGpuCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason
    if expect_clear:
        assert result.updates.get("clear_verified_job_info") is True
    else:
        assert "clear_verified_job_info" not in result.updates
