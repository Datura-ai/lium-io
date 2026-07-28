"""DAH-2520 — unrented incentive VRAM/disk gate.

An unrented executor whose total GPU VRAM exceeds the machine's total disk
forfeits the unrented rental incentive while staying active. Enforcement is
gated by ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT; while the flag is off the breach
is only logged (shadow mode) and the payout is unchanged.
"""

from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult

H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default

GB_IN_MB = 1024
GB_IN_KB = 1024 ** 2


def _build_incentive() -> RentalPriceIncentive:
    return RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})


def _make_job(
    *,
    vram_gb_per_gpu: float | None = 141.0,
    gpu_count: int = 8,
    disk_gb: float | None = 500.0,
    is_rented: bool = False,
) -> JobResult:
    spec: dict = {}
    if vram_gb_per_gpu is not None:
        spec["gpu"] = {
            "details": [{"capacity": vram_gb_per_gpu * GB_IN_MB} for _ in range(gpu_count)]
        }
    if disk_gb is not None:
        spec["hard_disk"] = {"total": disk_gb * GB_IN_KB}

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


def test_vram_over_disk_detects_machine_short_on_disk():
    # Arrange — 8 x 141 GB = 1128 GB VRAM on a 500 GB disk
    incentive = _build_incentive()

    # Act
    measured = incentive._vram_over_disk(_make_job())

    # Assert
    assert measured is not None
    assert measured.vram_gb == 1128.0
    assert measured.disk_gb == 500.0


def test_vram_over_disk_ignores_machine_with_enough_disk():
    # Arrange — same VRAM, 2 TB disk
    incentive = _build_incentive()

    # Act
    measured = incentive._vram_over_disk(_make_job(disk_gb=2048.0))

    # Assert
    assert measured is None


def test_vram_over_disk_equal_is_not_flagged():
    # Arrange — disk exactly equals VRAM; only strictly more VRAM is flagged
    incentive = _build_incentive()

    # Act
    measured = incentive._vram_over_disk(_make_job(vram_gb_per_gpu=100.0, gpu_count=2, disk_gb=200.0))

    # Assert
    assert measured is None


@pytest.mark.parametrize(
    "job_kwargs",
    [
        {"disk_gb": None},          # scrape has no hard_disk block
        {"vram_gb_per_gpu": None},  # scrape has no gpu details
        {"disk_gb": 0.0},           # unreadable disk reported as 0
    ],
)
def test_vram_over_disk_partial_scrape_is_never_flagged(job_kwargs):
    # A missing number must not cost a miner the incentive — fail open.
    incentive = _build_incentive()

    measured = incentive._vram_over_disk(_make_job(**job_kwargs))

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
    assert "vram_exceeds_disk" not in "\n".join(result.incentive_logs)


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
    assert "vram_exceeds_disk" in log
    assert "1128.0" in log  # total VRAM
    assert "500.0" in log   # total disk


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

    assert "vram_exceeds_disk" not in "\n".join(result.incentive_logs)
