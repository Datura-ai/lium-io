from unittest.mock import AsyncMock

import pytest

from neurons.validators.src.services.task.checks.banned_provider import BannedProviderCheck
from neurons.validators.src.services.task.messages import BannedProviderMessages as Msg
from neurons.validators.src.services.task.pipeline import Pipeline
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)

from tests.helpers import (
    build_context_config,
    build_services,
    build_state,
    default_executor,
)


def build_rented_data(
    *,
    banned_hotkeys: list[str] | None = None,
    banned_coldkeys: list[str] | None = None,
    banned_provider_guids: list[str] | None = None,
) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={},
        banned_hotkeys=banned_hotkeys or [],
        banned_coldkeys=banned_coldkeys or [],
        banned_provider_guids=banned_provider_guids or [],
    )


@pytest.mark.asyncio
async def test_hotkey_match_fails(context_factory):
    state = build_state(
        gpu_uuids="gpu-1",
        rented_data=build_rented_data(banned_hotkeys=["miner-hotkey"]),
    )
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(),
        state=state,
        miner_hotkey="miner-hotkey",
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.PROVIDER_BANNED.reason
    assert result.updates["is_provider_banned"] is True


@pytest.mark.asyncio
async def test_gpu_uuid_match_fails(context_factory):
    state = build_state(
        gpu_uuids="gpu-1,gpu-2",
        rented_data=build_rented_data(banned_provider_guids=["gpu-2"]),
    )
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(),
        state=state,
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.PROVIDER_BANNED.reason
    assert result.updates["is_provider_banned"] is True


@pytest.mark.asyncio
async def test_rented_provider_ban_is_not_fatal(context_factory):
    """A banned node with a live pod must keep being verified.

    Fatal here would abort the run, the node would stop being upserted, go stale and be
    flipped inactive by the hourly sweep, which stops renter billing and its payouts.
    """
    executor = default_executor()
    rented_data = RentedExecutorsResponse(
        executors={
            executor.uuid: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address=executor.address,
                executor_ip_port=str(executor.port),
                pods=[RentedPod(pod_id="pod-1", container_name="pod_p1")],
            )
        },
        banned_hotkeys=["miner-hotkey"],
    )
    state = build_state(gpu_uuids="gpu-1", rented_data=rented_data)
    ctx = context_factory(
        executor=executor,
        services=build_services(),
        config=build_context_config(),
        state=state,
        miner_hotkey="miner-hotkey",
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is False
    assert result.fatal is False

    pipeline = Pipeline(checks=[BannedProviderCheck()], sink=AsyncMock())
    success, _, final_ctx = await pipeline.run(ctx)
    assert success is True
    assert final_ctx.is_provider_banned is True


