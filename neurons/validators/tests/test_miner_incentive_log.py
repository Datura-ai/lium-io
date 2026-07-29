"""Unit tests for the miner_incentive_log catalog (DAH-2327).

Direct coverage of every builder in the single catalog: the machine-readable
`reason` code, the plain-English message shown to the miner, the structured
fields, and that a line actually lands in JobResult.incentive_logs.
"""
import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.miner_incentive_log import MinerLogLine, ZeroIncentiveReason
from incentive.rental_price import InsufficientDisk
from services.task_service import JobResult

H200 = "NVIDIA H200"


def test_reason_enum_pins_the_stable_code_contract():
    # Append-only contract: dashboards and the backend (DAH-2340) key off these
    # exact strings. Renaming any value is a breaking change.
    assert {r.value for r in ZeroIncentiveReason} == {
        "spot_tier",
        "provider_discord_not_connected",
        "new_rentals_paused",
        "miner_default_job",
        "gpu_model_not_eligible_for_unrented_incentive",
        "price_above_market_p90_soft_limit",
        "no_unrented_capacity_for_gpu_count",
        "nvidia_driver_below_minimum",
        "sysbox_not_enabled",
        "insufficient_disk_for_vram",
    }


def test_constructor_returns_enum_reason():
    line = MinerLogLine.no_payout_because_spot_tier(_job())
    assert line.reason is ZeroIncentiveReason.SPOT_TIER
    assert line.reason == "spot_tier"  # StrEnum: JSON/string behaviour unchanged


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
    "build, expected_code, message_fragment",
    [
        (lambda job: MinerLogLine.no_payout_because_spot_tier(job), "spot_tier", "spot tier"),
        (
            lambda job: MinerLogLine.no_payout_because_discord_not_connected(job),
            "provider_discord_not_connected",
            "Discord",
        ),
        (
            lambda job: MinerLogLine.no_payout_because_paused_for_new_rentals(job),
            "new_rentals_paused",
            "paused for new rentals",
        ),
        (
            lambda job: MinerLogLine.no_payout_because_running_own_default_job(job),
            "miner_default_job",
            "own default job",
        ),
        (
            lambda job: MinerLogLine.no_payout_because_gpu_model_not_in_unrented_program(job),
            "gpu_model_not_eligible_for_unrented_incentive",
            H200,
        ),
        (
            lambda job: MinerLogLine.no_payout_because_price_above_market_soft_limit(job, 3.842, 1.1),
            "price_above_market_p90_soft_limit",
            "soft price limit",
        ),
        (
            lambda job: MinerLogLine.no_payout_because_insufficient_disk_for_vram(
                job, InsufficientDisk(vram_gb=1128.0, disk_gb=500.0, rate=1.5)
            ),
            "insufficient_disk_for_vram",
            "GPU VRAM",
        ),
        (
            lambda job: MinerLogLine.no_payout_because_no_unrented_capacity_for_gpu_count(job, 8),
            "no_unrented_capacity_for_gpu_count",
            "no unrented-incentive capacity",
        ),
        (
            lambda job: MinerLogLine.no_payout_because_nvidia_driver_below_minimum(job),
            "nvidia_driver_below_minimum",
            "driver",
        ),
        (
            lambda job: MinerLogLine.no_payout_because_sysbox_not_enabled(job),
            "sysbox_not_enabled",
            "sysbox",
        ),
    ],
)
def test_zero_reason_constructor_code_and_message(build, expected_code, message_fragment):
    line = build(_job())
    assert line.reason == expected_code
    assert message_fragment.lower() in line.message.lower()
    assert line.fields["reason"] == expected_code
    assert line.fields["incentive"] == 0.0


def test_spot_tier_carries_internal_log_message():
    line = MinerLogLine.no_payout_because_spot_tier(_job())
    assert line.internal_message == "Executor excluded from both pools - spot tier"


def test_discord_reason_carries_connected_flag_for_internal_log():
    job = _job()
    job.provider_discord_connected = False
    line = MinerLogLine.no_payout_because_discord_not_connected(job)
    assert line.internal_fields["provider_discord_connected"] is False
    assert line.internal_fields["pool"] == "none"


def test_to_internal_log_renders_internal_message_with_pool_fields():
    line = MinerLogLine.no_payout_because_spot_tier(_job())
    rendered: str = line.to_internal_log().to_full_string()

    assert "Executor excluded from both pools - spot tier" in rendered
    assert "spot_tier" in rendered
    assert '"pool": "none"' in rendered


def test_soft_limit_reason_computes_threshold_and_fields():
    # threshold = p90 3.842 * rate 1.1 = 4.2262
    job = _job()
    job.executor_info = job.executor_info.model_copy(update={"price_per_gpu": 4.23})
    line = MinerLogLine.no_payout_because_price_above_market_soft_limit(job, 3.842, 1.1)
    assert line.fields["soft_limit_threshold"] == 4.2262
    assert line.fields["machine_price_p90"] == 3.842
    assert "4.2262" in line.message
    assert "4.23" in line.message


def test_insufficient_disk_reason_keeps_the_two_numbers_apart():
    # The builder takes the measurement whole, but the message still has to name which
    # number is which: reading them the wrong way advises the miner to add disk he has.
    line = MinerLogLine.no_payout_because_insufficient_disk_for_vram(
        _job(), InsufficientDisk(vram_gb=1128.0, disk_gb=500.0, rate=1.5)
    )

    assert line.fields["total_vram_gb"] == 1128.0
    assert line.fields["total_disk_gb"] == 500.0
    assert "500.0 GB of total disk but 1128.0 GB of GPU VRAM" in line.message
    assert "at least 1692.0 GB" in line.message


def test_no_capacity_reason_carries_bucket_fields():
    job = _job(max_cap=0, unrented_cap_multiplier=0.0, total_rental_cost=0.0)
    line = MinerLogLine.no_payout_because_no_unrented_capacity_for_gpu_count(job, count_bucket=8)
    assert line.fields["max_cap"] == 0
    assert line.fields["count_bucket"] == 8


def test_to_log_line_renders_message_and_reason_code():
    job = _job()
    entry = MinerLogLine.no_payout_because_spot_tier(job).to_log_line()

    assert "spot tier" in entry            # human message
    assert "spot_tier" in entry            # machine reason code in the structured payload
    assert str(job.executor_info.uuid) in entry


# ── Calculation-report builders ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "line, message_fragment, expected_keys",
    [
        (
            MinerLogLine.mining_score_calculated(_job(mining_score=1.0), False),
            "Mining score is calculated",
            ["mining_score", "gpu_portion", "driver_multiplier"],
        ),
        (
            MinerLogLine.mining_incentive_calculated("hk", _job(incentive=0.5), 0.83, 0.13),
            "Incentive score is calculated",
            ["mining_score", "total_mining_score", "mining_share", "incentive"],
        ),
        (
            MinerLogLine.rental_incentive_calculated("hk", _job(incentive=0.2), 8),
            "Rental price incentive",
            ["effective_rate", "rental_share", "count_bucket", "incentive"],
        ),
        (
            MinerLogLine.mining_score_missing("hk", _job()),
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
    line = MinerLogLine.mining_score_calculated(job, is_rented_after_cutoff=False)

    assert "Mining score is calculated" in line.to_log_line()

    # The same content is also available as an `_m` object for the internal logger.
    assert "Mining score is calculated" in line.as_internal_log().to_full_string()
