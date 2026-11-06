"""Tests for RentalPriceIncentive.get_snapshot() and standalone estimate_executor()."""

from unittest.mock import AsyncMock

import pytest

from incentive import rental_price as rental_price_module
from incentive.config import DEFAULT_PRICE, IncentiveConfig
from incentive.rental_price import (
    ExecutorEstimateParams,
    RentalPriceEstimate,
    RentalPriceIncentive,
)
from services.task_service import JobResult

pytest_plugins = ["fixtures.incentive_fixtures"]

ALGORITHM = "rental_price"

BASE_GPU_MAP = {
    "H100": "H100",
    "H100 NVL": "H100",
    "H200": "H200",
    "A100": "A100",
    "L4": "L4",
}


def _bucket_caps(cap: int) -> dict[int, int]:
    return {i: cap for i in range(1, 33)}


MAX_UNRENTED_GPUS_BY_TYPE: dict[str, dict[int, int]] = {
    "H100": _bucket_caps(16),
    "H200": _bucket_caps(8),
    "A100": {},   # ineligible
    "L4": {},     # ineligible
}

RENTAL_PRICES_PER_HOUR = {
    "H100": 3.50,
    "H200": 4.00,
    "A100": 2.00,
    "L4": 0.80,
}

TAO_PRICE = 500.0
ALPHA_RATE = 0.5


@pytest.fixture
def rental_config():
    return IncentiveConfig(
        algorithm=ALGORITHM,
        rental_incentive_gpu_types=[
            k for k, v in MAX_UNRENTED_GPUS_BY_TYPE.items() if any(c > 0 for c in v.values())
        ],
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
        rental_prices_per_hour=RENTAL_PRICES_PER_HOUR,
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
    )


@pytest.fixture
def mock_redis():
    svc = AsyncMock()
    svc.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    svc.get_executor_uptime = AsyncMock(return_value=9999)
    return svc


@pytest.fixture
def mock_price_provider():
    provider = AsyncMock()
    provider.get_tao_price.return_value = TAO_PRICE
    provider.get_alpha_rate.return_value = ALPHA_RATE
    return provider


def _make_incentive(rental_config, mock_redis, mock_price_provider, job_results, monkeypatch):
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", BASE_GPU_MAP)
    incentive = RentalPriceIncentive(
        rental_config, mock_redis,
        job_results,
        total_gpu_model_count_map={k: 10 for k in BASE_GPU_MAP},
    )
    incentive.price_provider = mock_price_provider
    return incentive


def _make_job(executor_id: str, gpu_model: str, gpu_count: int, is_rented: bool, sysbox_runtime: bool = True) -> JobResult:
    from datura.requests.miner_requests import ExecutorSSHInfo

    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid=executor_id, address="10.0.0.1", port=8080,
            ssh_username="root", ssh_port=22, python_path="/usr/bin/python3", root_dir="/tmp",
        ),
        score=1.0, job_score=1.0, job_batch_id="test-batch",
        log_status="success", log_text="ok",
        gpu_model=gpu_model, gpu_count=gpu_count, is_rented=is_rented,
        collateral_deposited=True,
        sysbox_runtime=sysbox_runtime,
    )


# ── get_snapshot() ────────────────────────────────────────────────────────────


# ── estimate_executor() — unrented path ──────────────────────────────────────


@pytest.mark.asyncio
async def test_estimate_executor_unrented_ineligible_gpu_returns_zero(
    rental_config, mock_redis, mock_price_provider, monkeypatch
):
    """GPU with empty cap dict (e.g. A100, L4) is ineligible and returns usd_per_epoch=0."""
    # Arrange
    job_results = {"miner_a": [_make_job("exec-a", "H100", 8, is_rented=False)]}
    incentive = _make_incentive(rental_config, mock_redis, mock_price_provider, job_results, monkeypatch)
    await incentive.calculate_mining_scores()
    snapshot = incentive.get_snapshot()

    # Act — A100 has empty cap dict, ineligible
    estimator = RentalPriceIncentive(
        rental_config,
        mock_redis,
        jobs_results={},
        total_gpu_model_count_map={},
        snapshot=snapshot,
    )
    result = await estimator.estimate_executor(ExecutorEstimateParams(gpu_model="A100", gpu_count=8, is_rented=False))

    # Assert — ineligible flag set, no reward
    assert result.eligible_for_rental_share is False
    assert result.usd_per_epoch == 0.0


# ── estimate_executor() — rented path ────────────────────────────────────────

@pytest.mark.asyncio
async def test_estimate_executor_rented_returns_tao(
    rental_config, mock_redis, mock_price_provider, monkeypatch
):
    """Rented path returns usd_per_epoch based on mining share."""
    # Arrange — some existing rented executors set mining_score baseline
    job_results = {
        "miner_a": [_make_job("exec-a", "H100", 8, is_rented=True)],
    }
    incentive = _make_incentive(rental_config, mock_redis, mock_price_provider, job_results, monkeypatch)
    await incentive.calculate_mining_scores()
    snapshot = incentive.get_snapshot()

    # Act
    estimator = RentalPriceIncentive(
        rental_config,
        mock_redis,
        jobs_results={},
        total_gpu_model_count_map={},
        snapshot=snapshot,
    )
    result = await estimator.estimate_executor(ExecutorEstimateParams(gpu_model="H100", gpu_count=8, is_rented=True))

    # Assert
    assert isinstance(result, RentalPriceEstimate)
    assert result.is_rented is True
    assert result.usd_per_epoch >= 0
    assert result.mining_score is not None


# ── snapshot seeding ──────────────────────────────────────────────────────────


# ── DAH-2528: estimate follows the occupancy-aware bucket fallback ───────────

