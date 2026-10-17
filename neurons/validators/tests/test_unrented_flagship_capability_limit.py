"""DAH-2546 / DAH-2594 — unrented incentive flagship capability gate.

An unrented 8x H200/B200/B300 executor that has none of NCU profiling counters
open on the host (ncu_profiling_access == "unrestricted"), real GPU splitting
enabled (min_gpu_count below the full node size), or a verified TDX quote
forfeits the unrented rental incentive while staying active. Enforcement is gated by
ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT; while the flag is off the breach is
only logged (shadow mode) and the payout is unchanged.
"""

from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive

from core.config import settings
from services.task_service import JobResult

H200 = "NVIDIA H200"
B200 = "NVIDIA B200"
B300 = "NVIDIA B300 SXM6 AC"
H100 = "NVIDIA H100 80GB HBM3"

# a scrape carrying no NCU observation at all: a machine not re-scraped since DAH-2182 shipped
SCRAPE_WITHOUT_NCU: dict[str, str] = {}


def _build_incentive() -> RentalPriceIncentive:
    return RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})


def _make_job(
    *,
    gpu_model: str = H200,
    gpu_count: int = 8,
    # spec=None models the spec-less synthetic JobResult the estimate path builds
    spec: dict[str, str] | None = SCRAPE_WITHOUT_NCU,
    supports_gpu_splitting: bool = False,
    gpu_splitting_min_count: int | None = None,
    is_rented: bool = False,
    attestation_digest: str | None = None,
    gpu_attestation_passed: bool | None = None,
    tdx_quote: str | None = None,
) -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid="exec-1",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
            price_per_gpu=1.0,
            tdx_quote=tdx_quote,
        ),
        spec=spec,
        score=1.0,
        job_score=1.0,
        job_batch_id="batch",
        log_status="success",
        log_text="ok",
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        is_rented=is_rented,
        supports_gpu_splitting=supports_gpu_splitting,
        attestation_digest=attestation_digest,
        gpu_attestation_passed=gpu_attestation_passed,
        gpu_splitting_min_count=gpu_splitting_min_count,
        collateral_deposited=True,
        sysbox_runtime=True,
    )


@pytest.mark.asyncio
async def test_enforced_drops_rental_eligibility(monkeypatch):
    # Arrange — neither capability and flag on
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job())

    # Assert — excluded from the rental pool, no mining either (active but no incentive)
    assert result.eligible_for_rental_share is False
    assert result.mining_score == 0


@pytest.mark.asyncio
async def test_rented_flagship_without_capabilities_untouched(monkeypatch):
    # The gate governs only the unrented pool: a rented machine keeps normal scoring
    # and is never told about the missing capabilities.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job(is_rented=True))

    assert "flagship_without_ncu_or_split" not in "\n".join(result.incentive_logs)
