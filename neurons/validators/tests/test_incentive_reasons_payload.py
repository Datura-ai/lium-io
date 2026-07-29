"""DAH-2340: the published MACHINE_SPEC_CHANNEL payload carries structured
zero-incentive reasons as DATA, and the validator ExecutorSpecRequest that goes
over the WebSocket validates them into typed IncentiveReason objects — the reason
never has to be parsed back out of log_text downstream.
"""
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from incentive.miner_incentive_log import MinerLogLine
from protocol.vc_protocol.validator_requests import ExecutorSpecRequest, IncentiveReason
from services.miner_service import MinerService
from services.redis_service import MACHINE_SPEC_CHANNEL

pytest_plugins = ["fixtures.incentive_fixtures"]


def _spec(reasons: list[dict[str, Any]] | None = None) -> ExecutorSpecRequest:
    """Build the ExecutorSpecRequest exactly as the redis→WS bridge does; omit reasons like an old payload would."""
    request_kwargs: dict[str, Any] = dict(
        miner_hotkey="hk",
        miner_coldkey="ck",
        validator_hotkey="vk",
        executor_uuid="exec-1",
        executor_ip="10.0.0.1",
        executor_port=8080,
        specs={},
        score=0.0,
        synthetic_job_score=0.0,
        log_text="ok",
        log_status="success",
        job_batch_id="2026-07-03 12:00:00",
        collateral_deposited=False,
    )
    if reasons is not None:
        request_kwargs["incentive_reasons"] = reasons
    return ExecutorSpecRequest(**request_kwargs)


@pytest.mark.asyncio
async def test_publisher_carries_reason_codes_as_data_not_just_log_text(
    create_job_result, mock_settings
) -> None:
    job = create_job_result()
    job.record_incentive_log(MinerLogLine.no_payout_because_spot_tier(job))
    redis_service = MagicMock()
    redis_service.publish = AsyncMock()
    service = MinerService(
        ssh_service=MagicMock(),
        task_service=MagicMock(),
        redis_service=redis_service,
        attestation_service=MagicMock(),
    )

    await service.publish_machine_specs([job], miner_hotkey="hk", miner_coldkey="ck")

    channel, payload = redis_service.publish.await_args.args
    assert channel == MACHINE_SPEC_CHANNEL
    spec = _spec(payload["incentive_reasons"])

    assert [reason.reason for reason in spec.incentive_reasons] == ["spot_tier"]
    assert isinstance(spec.incentive_reasons[0], IncentiveReason)
    assert spec.incentive_reasons[0].message_for_miner  # miner-facing text present


@pytest.mark.asyncio
async def test_publisher_reports_empty_list_when_no_catalogued_reason_exists(
    create_job_result, mock_settings
) -> None:
    job = create_job_result()
    redis_service = MagicMock()
    redis_service.publish = AsyncMock()
    service = MinerService(
        ssh_service=MagicMock(),
        task_service=MagicMock(),
        redis_service=redis_service,
        attestation_service=MagicMock(),
    )

    await service.publish_machine_specs([job], miner_hotkey="hk", miner_coldkey="ck")

    _, payload = redis_service.publish.await_args.args
    assert payload["incentive_reasons"] == []


def test_miner_log_fields_survive_in_the_context_field(create_job_result) -> None:
    job = create_job_result()
    job.executor_info = job.executor_info.model_copy(update={"price_per_gpu": 4.23})
    job.record_incentive_log(
        MinerLogLine.no_payout_because_price_above_market_soft_limit(job, market_p90=3.842, rate=1.1)
    )
    spec = _spec([reason.model_dump() for reason in job.zero_incentive_reasons])

    reason = spec.incentive_reasons[0]
    assert reason.reason == "price_above_market_p90_soft_limit"
    # the per-reason details ride in the explicit context field
    assert reason.context["soft_limit_threshold"] == 4.2262


def test_payload_without_the_key_parses_as_none_meaning_not_reported() -> None:
    assert _spec().incentive_reasons is None
