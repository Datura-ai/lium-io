"""DAH-2250 — unrented incentive soft price limit.

An unrented executor priced above the market p90 ceiling
(machine_prices_p90[gpu] * the shared config's soft_limit_price_rate) forfeits the unrented rental
incentive while staying active. Enforcement is gated by
ENABLE_UNRENTED_SOFT_PRICE_LIMIT; while the flag is off the breach is only
logged (shadow mode) and the payout is unchanged.
"""

from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings, shared_client
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult

H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default


def _build_incentive() -> RentalPriceIncentive:
    return RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})


def _make_job(
    price_per_gpu: float | None,
    *,
    gpu_model: str = H200,
    is_rented: bool = False,
    is_spot: bool = False,
    is_new_rentals_paused: bool = False,
    provider_discord_connected: bool = True,
    default_job_owner: str | None = None,
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
            price_per_gpu=price_per_gpu,
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="batch",
        log_status="success",
        log_text="ok",
        gpu_model=gpu_model,
        gpu_count=1,
        is_rented=is_rented,
        is_spot=is_spot,
        is_new_rentals_paused=is_new_rentals_paused,
        provider_discord_connected=provider_discord_connected,
        default_job_owner=default_job_owner,
        collateral_deposited=True,
        sysbox_runtime=True,
    )


def _set_p90(monkeypatch, mapping: dict[str, float]) -> None:
    new_cfg = shared_client.config.model_copy(update={"machine_prices_p90": mapping})
    monkeypatch.setattr(shared_client, "_config", new_cfg)


def test_is_over_soft_price_limit_above_threshold(monkeypatch):
    # Arrange — threshold = 2.0 * 1.1 = 2.2
    _set_p90(monkeypatch, {H200: 2.0})
    incentive = _build_incentive()

    # Act
    over = incentive._is_over_soft_price_limit(_make_job(2.3))

    # Assert
    assert over is True


def test_is_over_soft_price_limit_at_threshold_is_not_over(monkeypatch):
    # Arrange — exactly at the threshold (2.2) is allowed
    _set_p90(monkeypatch, {H200: 2.0})
    incentive = _build_incentive()

    # Act
    over = incentive._is_over_soft_price_limit(_make_job(2.2))

    # Assert
    assert over is False


@pytest.mark.asyncio
async def test_shadow_mode_keeps_rental_eligibility(monkeypatch):
    # Arrange — over the limit but flag off → shadow only
    _set_p90(monkeypatch, {H200: 2.0})
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_SOFT_PRICE_LIMIT", False)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job(2.3))

    # Assert — still eligible for the unrented rental pool
    assert result.eligible_for_rental_share is True


@pytest.mark.asyncio
async def test_enforced_drops_rental_eligibility(monkeypatch):
    # Arrange — over the limit and flag on
    _set_p90(monkeypatch, {H200: 2.0})
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_SOFT_PRICE_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job(2.3))

    # Assert — excluded from rental pool, no mining either (active but no incentive)
    assert result.eligible_for_rental_share is False
    assert result.mining_score == 0


# ── DAH-2327: every zero-incentive exit surfaces its reason to the miner ──────


