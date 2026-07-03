"""DAH-2250 — unrented incentive soft price limit.

An unrented executor priced above the market p90 ceiling
(machine_prices_p90[gpu] * SOFT_LIMIT_PRICE_RATE) forfeits the unrented rental
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


def test_is_over_soft_price_limit_no_market_data(monkeypatch):
    # Arrange — H200 absent from p90 map → cannot gate
    _set_p90(monkeypatch, {})
    incentive = _build_incentive()

    # Act
    over = incentive._is_over_soft_price_limit(_make_job(99.0))

    # Assert
    assert over is False


def test_is_over_soft_price_limit_no_price(monkeypatch):
    # Arrange — miner price unknown → cannot gate
    _set_p90(monkeypatch, {H200: 2.0})
    incentive = _build_incentive()

    # Act
    over = incentive._is_over_soft_price_limit(_make_job(None))

    # Assert
    assert over is False


def test_enforcement_defaults_to_shadow_mode():
    # Arrange / Act — rollout contract: first deploy must be shadow-only, so the
    # flag default (not the env-resolved value) must stay False.
    default = type(settings).model_fields["ENABLE_UNRENTED_SOFT_PRICE_LIMIT"].default

    # Assert
    assert default is False


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


@pytest.mark.asyncio
async def test_enforced_under_threshold_keeps_eligibility(monkeypatch):
    # Arrange — flag on but price within the p90 ceiling (2.1 < 2.2)
    _set_p90(monkeypatch, {H200: 2.0})
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_SOFT_PRICE_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job(2.1))

    # Assert
    assert result.eligible_for_rental_share is True


@pytest.mark.asyncio
async def test_enforced_appends_customer_facing_incentive_log(monkeypatch):
    # DAH-2327: the zero-incentive reason must reach the customer-facing incentive
    # log (JobResult.incentive_logs -> "Incentive Scores Calculation Logs") with the
    # numbers and the price to set. Threshold = 2.0 * 1.1 = 2.2.
    _set_p90(monkeypatch, {H200: 2.0})
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_SOFT_PRICE_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job(2.3))

    log = "\n".join(result.incentive_logs)
    assert "soft price limit" in log
    assert "2.3" in log        # miner's price
    assert "2.2" in log        # ceiling / price to set


@pytest.mark.asyncio
async def test_shadow_mode_does_not_append_incentive_log(monkeypatch):
    # Flag off — payout unchanged, so no zero-incentive reason should be logged.
    _set_p90(monkeypatch, {H200: 2.0})
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_SOFT_PRICE_LIMIT", False)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job(2.3))

    log = "\n".join(result.incentive_logs)
    assert "soft price limit" not in log
