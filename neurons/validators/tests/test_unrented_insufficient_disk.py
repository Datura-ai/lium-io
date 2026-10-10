"""DAH-2520 — unrented incentive disk/VRAM gate.

An unrented executor whose total disk is below its total GPU VRAM times
MIN_DISK_TO_VRAM_RATE forfeits the unrented rental incentive while staying active.
Enforcement is gated by ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT; while the flag is
off the breach is only logged (shadow mode) and the payout is unchanged.
"""

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


def _logged_reasons(caplog) -> list[str]:
    # the reason codes of the structured lines captured so far; _m keeps them off the message
    return [r.msg.extra.get("reason") for r in caplog.records if hasattr(r.msg, "extra")]


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


