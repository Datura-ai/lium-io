"""Unit tests for the miner_incentive_log catalog (DAH-2327).

Direct coverage of every builder in the single catalog: the machine-readable
`reason` code, the plain-English message shown to the miner, the structured
fields, and that a line actually lands in JobResult.incentive_logs.
"""
import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive import miner_incentive_log as miner_log
from services.task_service import JobResult

H200 = "NVIDIA H200"


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
        score=1.0,
        job_score=1.0,
        job_batch_id="batch",
        log_status="success",
        log_text="ok",
        gpu_model=H200,
        gpu_count=8,
    )
    base.update(overrides)
    return JobResult(**base)


# ── Zero-incentive reason builders ───────────────────────────────────────────

@pytest.mark.parametrize(
    "reason, expected_code, message_fragment",
    [
        (miner_log.no_payout_because_spot_tier(), "spot_tier", "spot tier"),
        (
            miner_log.no_payout_because_discord_not_connected(False),
            "provider_discord_not_connected",
            "Discord",
        ),
        (
            miner_log.no_payout_because_paused_for_new_rentals(),
            "new_rentals_paused",
            "paused for new rentals",
        ),
        (
            miner_log.no_payout_because_running_own_default_job(),
            "miner_default_job",
            "own default job",
        ),
        (
            miner_log.no_payout_because_gpu_model_not_in_unrented_program(H200),
            "gpu_model_not_eligible_for_unrented_incentive",
            H200,
        ),
        (
            miner_log.no_payout_because_price_above_market_soft_limit(4.23, 3.842, 1.1),
            "price_above_market_p90_soft_limit",
            "soft price limit",
        ),
        (
            miner_log.no_payout_because_no_unrented_capacity_for_gpu_count(8, H200, 8, 0, 0.0, 0.0),
            "no_unrented_capacity_for_gpu_count",
            "no unrented-incentive capacity",
        ),
    ],
)
def test_zero_reason_builder_code_and_message(reason, expected_code, message_fragment):
    assert reason.reason == expected_code
    assert message_fragment.lower() in reason.message_for_miner.lower()


def test_spot_tier_carries_internal_log_message():
    reason = miner_log.no_payout_because_spot_tier()
    assert reason.internal_log_message == "Executor excluded from both pools - spot tier"


def test_discord_reason_carries_connected_flag_for_internal_log():
    reason = miner_log.no_payout_because_discord_not_connected(is_connected=False)
    assert reason.internal_log_fields == {"provider_discord_connected": False}


def test_soft_limit_reason_computes_threshold_and_fields():
    # threshold = p90 3.842 * rate 1.1 = 4.2262
    reason = miner_log.no_payout_because_price_above_market_soft_limit(4.23, 3.842, 1.1)
    assert reason.miner_log_fields["soft_limit_threshold"] == 4.2262
    assert reason.miner_log_fields["machine_price_p90"] == 3.842
    assert "4.2262" in reason.message_for_miner
    assert "4.23" in reason.message_for_miner


def test_no_capacity_reason_carries_bucket_fields():
    reason = miner_log.no_payout_because_no_unrented_capacity_for_gpu_count(
        gpu_count=8, gpu_model=H200, count_bucket=8, max_cap=0, cap_multiplier=0.0, total_rental_cost=0.0
    )
    assert reason.miner_log_fields["max_cap"] == 0
    assert reason.miner_log_fields["count_bucket"] == 8


def test_to_log_line_renders_message_and_reason_code():
    job = _job()
    entry = miner_log.no_payout_because_spot_tier().to_log_line(job)

    assert "spot tier" in entry            # human message
    assert "spot_tier" in entry            # machine reason code in the structured payload
    assert str(job.executor_info.uuid) in entry


# ── Calculation-report builders ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "line, message_fragment, expected_keys",
    [
        (
            miner_log.mining_score_calculated(_job(mining_score=1.0), False),
            "Mining score is calculated",
            ["mining_score", "gpu_portion", "driver_multiplier"],
        ),
        (
            miner_log.mining_incentive_calculated("hk", _job(incentive=0.5), 0.83, 0.13),
            "Incentive score is calculated",
            ["mining_score", "total_mining_score", "mining_share", "incentive"],
        ),
        (
            miner_log.rental_incentive_calculated("hk", _job(incentive=0.2), 8),
            "Rental price incentive",
            ["effective_rate", "rental_share", "count_bucket", "incentive"],
        ),
        (
            miner_log.mining_score_missing("hk", _job()),
            "should not happen",
            ["score", "job_score", "gpu_model"],
        ),
    ],
)
def test_report_builder_message_and_keys(line, message_fragment, expected_keys):
    assert message_fragment.lower() in line.message.lower()
    for key in expected_keys:
        assert key in line.fields


def test_report_line_renders_string_and_builds_internal_log():
    job = _job(mining_score=1.0)
    line = miner_log.mining_score_calculated(job, is_rented_after_cutoff=False)

    assert "Mining score is calculated" in line.to_log_line()

    # The same content is also available as an `_m` object for the internal logger.
    assert "Mining score is calculated" in line.as_internal_log().to_full_string()
