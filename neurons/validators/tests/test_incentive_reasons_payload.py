"""DAH-2340: the published MACHINE_SPEC_CHANNEL payload carries structured
zero-incentive reasons as DATA, and the validator ExecutorSpecRequest that goes
over the WebSocket validates them into typed IncentiveReason objects — the reason
never has to be parsed back out of log_text downstream.
"""
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.miner_incentive_log import MinerLogLine
from protocol.vc_protocol.validator_requests import ExecutorSpecRequest, IncentiveReason
from services.task_service import JobResult


def _job(**overrides) -> JobResult:
    base = dict(
        executor_info=ExecutorSSHInfo(
            uuid="exec-1",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
        ),
        score=0.0,
        job_score=0.0,
        job_batch_id="2026-07-03 12:00:00",
        log_status="success",
        log_text="ok",
        gpu_model="NVIDIA H200",
        gpu_count=8,
    )
    base.update(overrides)
    return JobResult(**base)


def _spec(reasons: list[dict]) -> ExecutorSpecRequest:
    """Build the ExecutorSpecRequest exactly as the redis→WS bridge does."""
    return ExecutorSpecRequest(
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
        incentive_reasons=reasons,
    )


def test_published_payload_carries_reason_codes_as_data_not_just_log_text():
    job = _job()
    job.record_incentive_log(MinerLogLine.no_payout_because_spot_tier(job))

    # This is the exact expression the publisher (miner_service.publish_machine_specs) sends.
    published_reasons = [reason.to_reason_payload() for reason in job.zero_incentive_reasons]
    spec = _spec(published_reasons)

    assert [r.reason for r in spec.incentive_reasons] == ["spot_tier"]
    assert isinstance(spec.incentive_reasons[0], IncentiveReason)
    assert spec.incentive_reasons[0].message_for_miner  # miner-facing text present


def test_extra_miner_log_fields_survive_on_the_typed_model():
    job = _job()
    job.executor_info = job.executor_info.model_copy(update={"price_per_gpu": 4.23})
    job.record_incentive_log(
        MinerLogLine.no_payout_because_price_above_market_soft_limit(job, market_p90=3.842, rate=1.1)
    )
    spec = _spec([reason.to_reason_payload() for reason in job.zero_incentive_reasons])

    reason = spec.incentive_reasons[0]
    assert reason.reason == "price_above_market_p90_soft_limit"
    # extra='allow' keeps the structured context next to the stable code
    assert reason.model_dump()["soft_limit_threshold"] == 4.2262


def test_missing_key_defaults_to_empty_list_for_backward_compat():
    spec = ExecutorSpecRequest(
        miner_hotkey="hk",
        miner_coldkey="ck",
        validator_hotkey="vk",
        executor_uuid="exec-1",
        executor_ip="10.0.0.1",
        executor_port=8080,
        specs={},
        score=1.0,
        synthetic_job_score=1.0,
        log_text="ok",
        log_status="success",
        job_batch_id="2026-07-03 12:00:00",
        collateral_deposited=False,
    )
    assert spec.incentive_reasons == []
