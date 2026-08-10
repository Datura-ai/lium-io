from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from neurons.validators.src.services.task.checks.banned_provider import BannedProviderCheck
from neurons.validators.src.services.task.messages import BannedProviderMessages as Msg
from neurons.validators.src.services.task.pipeline import Pipeline
from neurons.validators.src.services.task.result_handler import ResultHandler
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

from tests.helpers import build_context_config, build_services, build_state


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
    assert result.updates == {"is_provider_banned": True}


@pytest.mark.asyncio
async def test_coldkey_match_fails(context_factory):
    state = build_state(
        gpu_uuids="gpu-1",
        rented_data=build_rented_data(banned_coldkeys=["miner-coldkey"]),
    )
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(),
        state=state,
        miner_hotkey="fresh-hotkey",
        miner_coldkey="miner-coldkey",
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.PROVIDER_BANNED.reason
    assert result.updates == {"is_provider_banned": True}


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
    assert result.updates == {"is_provider_banned": True}


@pytest.mark.asyncio
async def test_clean_provider_passes(context_factory):
    state = build_state(
        gpu_uuids="gpu-1",
        rented_data=build_rented_data(
            banned_hotkeys=["other-hotkey"],
            banned_coldkeys=["other-coldkey"],
            banned_provider_guids=["other-gpu"],
        ),
    )
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(),
        state=state,
        miner_hotkey="miner-hotkey",
        miner_coldkey="miner-coldkey",
    )

    result = await BannedProviderCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.PROVIDER_ALLOWED.reason
    assert result.updates == {}


@pytest.mark.asyncio
async def test_provider_ban_preserves_verified_job_info(context_factory):
    state = build_state(
        rented_data=build_rented_data(banned_hotkeys=["miner-hotkey"]),
    )
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(),
        state=state,
        miner_hotkey="miner-hotkey",
    )

    result = await BannedProviderCheck().run(ctx)

    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_provider_ban_reaches_job_result(context_factory):
    state = build_state(
        rented_data=build_rented_data(banned_hotkeys=["miner-hotkey"]),
    )
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(),
        state=state,
        miner_hotkey="miner-hotkey",
    )
    pipeline = Pipeline(checks=[BannedProviderCheck()], sink=AsyncMock())

    success, _, final_ctx = await pipeline.run(ctx)
    job_result = await ResultHandler(redis_service=None, dry_run=True).handle_result(
        context=final_ctx,
        miner_info=SimpleNamespace(miner_hotkey="miner-hotkey", job_batch_id="batch-1"),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="provider banned",
        success=success,
    )

    assert success is False
    assert final_ctx.clear_verified_job_info is False
    assert job_result.is_provider_banned is True
