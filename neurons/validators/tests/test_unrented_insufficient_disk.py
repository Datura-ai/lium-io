"""DAH-2520 — unrented incentive disk/VRAM gate.

An unrented executor whose total disk is below its total GPU VRAM times
MIN_DISK_TO_VRAM_RATE forfeits the unrented rental incentive while staying active.
Enforcement is gated by ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT; while the flag is
off the breach is only logged (shadow mode) and the payout is unchanged.
"""

import logging
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult

H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default

MB_PER_GB = 1024
KB_PER_GB = 1024 ** 2


def _build_incentive() -> RentalPriceIncentive:
    return RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})


def _make_job(
    *,
    vram_gb_per_gpu: float | None = 141.0,
    gpu_count: int = 8,
    disk_gb: float | None = 500.0,
    is_rented: bool = False,
    spec: dict | None = None,
) -> JobResult:
    if spec is None:
        spec = {}
        if vram_gb_per_gpu is not None:
            spec["gpu"] = {
                "details": [{"capacity": vram_gb_per_gpu * MB_PER_GB} for _ in range(gpu_count)]
            }
        if disk_gb is not None:
            spec["hard_disk"] = {"total": disk_gb * KB_PER_GB}

    return JobResult(
        spec=spec,
        executor_info=ExecutorSSHInfo(
            uuid="exec-1",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
            price_per_gpu=1.0,
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="batch",
        log_status="success",
        log_text="ok",
        gpu_model=H200,
        gpu_count=gpu_count,
        is_rented=is_rented,
        collateral_deposited=True,
        sysbox_runtime=True,
    )


def test_insufficient_disk_detects_machine_short_on_disk():
    # Arrange — 8 x 141 GB = 1128 GB VRAM on a 500 GB disk
    incentive = _build_incentive()

    # Act
    measured = incentive._insufficient_disk(_make_job())

    # Assert
    assert measured is not None
    assert measured.vram_gb == 1128.0
    assert measured.disk_gb == 500.0


def test_insufficient_disk_ignores_machine_with_enough_disk():
    # Arrange — same VRAM, 2 TB disk
    incentive = _build_incentive()

    # Act
    measured = incentive._insufficient_disk(_make_job(disk_gb=2048.0))

    # Assert
    assert measured is None


def test_insufficient_disk_at_ratio_threshold_is_not_flagged():
    # Arrange — disk exactly equals VRAM * MIN_DISK_TO_VRAM_RATE; only strictly more is flagged
    incentive = _build_incentive()

    # Act
    measured = incentive._insufficient_disk(_make_job(vram_gb_per_gpu=100.0, gpu_count=2, disk_gb=300.0))

    # Assert
    assert measured is None


def test_insufficient_disk_equal_totals_is_flagged():
    # Arrange — equal VRAM and disk no longer clears the 1.5x margin
    incentive = _build_incentive()

    # Act
    measured = incentive._insufficient_disk(_make_job(vram_gb_per_gpu=100.0, gpu_count=2, disk_gb=200.0))

    # Assert
    assert measured is not None
    assert measured.vram_gb == 200.0
    assert measured.disk_gb == 200.0


@pytest.mark.parametrize(
    "job_kwargs",
    [
        {"disk_gb": None},                          # scrape has no hard_disk block
        {"vram_gb_per_gpu": None},                  # scrape has no gpu details
        {"disk_gb": 0.0},                           # unreadable disk reported as 0
        {"spec": {"hard_disk": {"total": 1}, "gpu": {"details": []}}},  # gpu list came back empty
        {"vram_gb_per_gpu": None, "disk_gb": None},  # no scrape at all
    ],
)
def test_insufficient_disk_partial_scrape_is_never_flagged(job_kwargs):
    # A missing number must not cost a miner the incentive — fail open.
    incentive = _build_incentive()

    measured = incentive._insufficient_disk(_make_job(**job_kwargs))

    assert measured is None


def _logged_reasons(caplog) -> list[str]:
    # the reason codes of the structured lines captured so far; _m keeps them off the message
    return [r.msg.extra.get("reason") for r in caplog.records if hasattr(r.msg, "extra")]


def test_estimated_job_result_is_not_logged_as_unmeasured(caplog):
    # estimate_executor builds a spec-less JobResult for every GPU model every cycle;
    # warning on those would drown the shadow measurement this log exists to feed.
    incentive = _build_incentive()

    with caplog.at_level(logging.WARNING):
        incentive._insufficient_disk(_make_job(vram_gb_per_gpu=None, disk_gb=None))

    assert "insufficient_disk_unmeasured" not in _logged_reasons(caplog)


def test_unreadable_scrape_is_logged_as_unmeasured(caplog):
    # A machine that reports no disk must be distinguishable from one that passed the gate.
    incentive = _build_incentive()

    with caplog.at_level(logging.WARNING):
        incentive._insufficient_disk(_make_job(disk_gb=None))

    assert "insufficient_disk_unmeasured" in _logged_reasons(caplog)


MALFORMED_SPECS: list[dict] = [
    {"gpu": {"details": [{"capacity": "141 GB"}]}, "hard_disk": {"total": 1}},  # unparsable number
    {"gpu": {"details": [{"capacity": {"total": 1}}]}, "hard_disk": {"total": 1}},  # nested object
    {"gpu": ["details"], "hard_disk": {"total": 1}},                            # gpu is a list
    {"gpu": {"details": 8}, "hard_disk": {"total": 1}},                         # details not iterable
    {"gpu": {"details": ["141"]}, "hard_disk": {"total": 1}},                   # entry is not a dict
    {"gpu": {"details": [{"capacity": 1}]}, "hard_disk": "500"},                # hard_disk is a string
]


@pytest.mark.parametrize("spec", MALFORMED_SPECS)
def test_insufficient_disk_survives_a_malformed_scrape(spec):
    # The scrape is produced on the miner's machine and calculate_mining_scores has no
    # per-result guard, so a malformed value must fail open instead of raising.
    incentive = _build_incentive()

    measured = incentive._insufficient_disk(_make_job(spec=spec))

    assert measured is None


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", MALFORMED_SPECS)
async def test_malformed_scrape_never_breaks_scoring(monkeypatch, spec):
    # Raising out of calculate_executor_score would cost EVERY miner this cycle's weights.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job(spec=spec))

    assert result.eligible_for_rental_share is True


def test_insufficient_disk_reports_the_numbers_it_compared():
    # Rounding happens before the comparison: raw 200.03 * 1.5 = 300.045 GB would exceed
    # 300.0 GB disk, but rounded to 200.0 GB VRAM it clears the 1.5x margin exactly.
    incentive = _build_incentive()

    measured = incentive._insufficient_disk(
        _make_job(vram_gb_per_gpu=200.03, gpu_count=1, disk_gb=300.0)
    )

    assert measured is None


def test_enforcement_defaults_to_shadow_mode():
    # Rollout contract: first deploy must be shadow-only, so the flag default
    # (not the env-resolved value) must stay False.
    default = type(settings).model_fields["ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT"].default

    assert default is False


@pytest.mark.asyncio
async def test_shadow_mode_keeps_rental_eligibility(monkeypatch):
    # Arrange — VRAM over disk but flag off → shadow only
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT", False)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job())

    # Assert — still eligible for the unrented rental pool, nothing told to the miner
    assert result.eligible_for_rental_share is True
    assert "insufficient_disk_for_vram" not in "\n".join(result.incentive_logs)


@pytest.mark.asyncio
async def test_shadow_mode_emits_the_measurement_log(monkeypatch, caplog):
    # The shadow log is the whole deliverable of the first deploy: the rollout decision
    # is made from these fields, so they are a dashboard contract.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT", False)
    incentive = _build_incentive()

    with caplog.at_level(logging.INFO):
        await incentive.calculate_executor_score(_make_job())

    breach = next(r.msg for r in caplog.records if r.msg.extra.get("reason") == "insufficient_disk_for_vram")
    assert "shadow only - flag off" in breach.message
    assert breach.extra["total_vram_gb"] == 1128.0
    assert breach.extra["total_disk_gb"] == 500.0
    assert breach.extra["enforced"] is False
    assert breach.extra["pool"] == "rental_kept_shadow"


@pytest.mark.asyncio
async def test_enforced_drops_rental_eligibility(monkeypatch):
    # Arrange — VRAM over disk and flag on
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job())

    # Assert — excluded from the rental pool, no mining either (active but no incentive)
    assert result.eligible_for_rental_share is False
    assert result.mining_score == 0


@pytest.mark.asyncio
async def test_enforced_appends_customer_facing_incentive_log(monkeypatch):
    # DAH-2327: the zero-incentive reason must reach the customer-facing incentive log
    # with both numbers, so the miner knows how much disk to add.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job())

    log = "\n".join(result.incentive_logs)
    assert "insufficient_disk_for_vram" in log
    assert "1128.0" in log  # total VRAM
    assert "500.0" in log   # total disk
    assert [reason.reason for reason in result.zero_incentive_reasons] == ["insufficient_disk_for_vram"]


@pytest.mark.asyncio
async def test_enough_disk_keeps_eligibility_when_enforced(monkeypatch):
    # Arrange — flag on but the machine has more disk than VRAM
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job(disk_gb=2048.0))

    # Assert
    assert result.eligible_for_rental_share is True


@pytest.mark.asyncio
async def test_rented_executor_is_not_gated(monkeypatch):
    # A rented machine earns from the rented path and must not be touched by this gate.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job(is_rented=True))

    assert "insufficient_disk_for_vram" not in "\n".join(result.incentive_logs)
