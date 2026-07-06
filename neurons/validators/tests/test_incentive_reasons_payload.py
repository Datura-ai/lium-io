"""DAH-2340: the published MACHINE_SPEC_CHANNEL payload carries structured
zero-incentive reasons as DATA, and the validator ExecutorSpecRequest that goes
over the WebSocket validates them into typed IncentiveReason objects — the reason
never has to be parsed back out of log_text downstream.
"""
from typing import Any

from incentive.miner_incentive_log import MinerLogLine
from protocol.vc_protocol.validator_requests import ExecutorSpecRequest, IncentiveReason

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


def test_published_payload_carries_reason_codes_as_data_not_just_log_text(create_job_result) -> None:
    job = create_job_result()
    job.record_incentive_log(MinerLogLine.no_payout_because_spot_tier(job))

    # model_dump per reason is exactly what the publisher (miner_service) sends.
    spec = _spec([reason.model_dump() for reason in job.zero_incentive_reasons])

    assert [reason.reason for reason in spec.incentive_reasons] == ["spot_tier"]
    assert isinstance(spec.incentive_reasons[0], IncentiveReason)
    assert spec.incentive_reasons[0].message_for_miner  # miner-facing text present


def test_extra_miner_log_fields_survive_on_the_typed_model(create_job_result) -> None:
    job = create_job_result()
    job.executor_info = job.executor_info.model_copy(update={"price_per_gpu": 4.23})
    job.record_incentive_log(
        MinerLogLine.no_payout_because_price_above_market_soft_limit(job, market_p90=3.842, rate=1.1)
    )
    spec = _spec([reason.model_dump() for reason in job.zero_incentive_reasons])

    reason = spec.incentive_reasons[0]
    assert reason.reason == "price_above_market_p90_soft_limit"
    # extra='allow' keeps the structured context next to the stable code
    assert reason.model_dump()["soft_limit_threshold"] == 4.2262


def test_payload_without_the_key_parses_as_none_meaning_not_reported() -> None:
    assert _spec().incentive_reasons is None
