"""Tests for RentalPriceIncentive.get_snapshot() and standalone estimate_executor()."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from incentive import rental_price as rental_price_module
from incentive.config import DEFAULT_PRICE, IncentiveConfig
from incentive.rental_price import (
    ExecutorEstimateParams,
    GpuTypeRentalState,
    RentalPriceEstimate,
    RentalPriceIncentive,
    RentalPriceSnapshot,
    precompute_all_estimates,
)
from services.const import TEMPO, SECONDS_PER_BLOCK, FIXED_RATIO, TOTAL_BURN_EMISSION
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

MAX_UNRENTED_GPUS_BY_TYPE = {
    "H100": 16,
    "H200": 8,
    "A100": 0,   # ineligible
    "L4": 0,     # ineligible
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
        rental_incentive_gpu_types=[k for k, v in MAX_UNRENTED_GPUS_BY_TYPE.items() if v > 0],
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

@pytest.mark.asyncio
async def test_get_snapshot_fields_after_calculate_mining_scores(
    rental_config, mock_redis, mock_price_provider, monkeypatch
):
    """Snapshot captures epoch state after calculate_mining_scores runs."""
    # Arrange
    job_results = {
        "miner_a": [_make_job("exec-a1", "H100", 8, is_rented=False)],
        "miner_b": [_make_job("exec-b1", "H200", 4, is_rented=False)],
    }
    incentive = _make_incentive(rental_config, mock_redis, mock_price_provider, job_results, monkeypatch)

    # Act
    await incentive.calculate_mining_scores()
    snapshot = incentive.get_snapshot()

    # Assert — snapshot is a valid RentalPriceSnapshot
    assert isinstance(snapshot, RentalPriceSnapshot)

    # epoch_subnet_emission = TEMPO * TAO_PRICE * ALPHA_RATE
    expected_emission = TEMPO * TAO_PRICE * ALPHA_RATE
    assert snapshot.epoch_subnet_emission == pytest.approx(expected_emission)

    # H100 and H200 both unrented → both tracked in by_gpu_type
    assert "H100" in snapshot.rental.by_gpu_type
    assert "H200" in snapshot.rental.by_gpu_type

    h100_state = snapshot.rental.by_gpu_type["H100"]
    assert h100_state.unrented_count == 8
    assert h100_state.max_cap == MAX_UNRENTED_GPUS_BY_TYPE["H100"]
    # weighted_rate_sum = gpu_count * hourly_rate = 8 * 3.50
    assert h100_state.weighted_rate_sum == pytest.approx(8 * RENTAL_PRICES_PER_HOUR["H100"])

    h200_state = snapshot.rental.by_gpu_type["H200"]
    assert h200_state.unrented_count == 4
    assert h200_state.weighted_rate_sum == pytest.approx(4 * RENTAL_PRICES_PER_HOUR["H200"])

    # total_gpu_count counts all successful results
    assert snapshot.mining.total_gpu_count == 12  # 8 + 4
    assert snapshot.mining.total_gpu_model_count_map == {"H100": 8, "H200": 4}


@pytest.mark.asyncio
async def test_get_snapshot_cap_multiplier_stored(
    rental_config, mock_redis, mock_price_provider, monkeypatch
):
    """Snapshot stores computed cap_multiplier even when unrented count exceeds cap."""
    # Arrange — H100 cap is 16; 20 GPUs exceeds it
    job_results = {
        "miner_a": [_make_job("exec-a", "H100", 20, is_rented=False)],
    }
    incentive = _make_incentive(rental_config, mock_redis, mock_price_provider, job_results, monkeypatch)

    # Act
    await incentive.calculate_mining_scores()
    snapshot = incentive.get_snapshot()

    # Assert — cap_multiplier = min(20, 16) / 20
    h100_state = snapshot.rental.by_gpu_type["H100"]
    assert h100_state.cap_multiplier == pytest.approx(16 / 20)
    assert snapshot.mining.total_gpu_model_count_map == {"H100": 20}


# ── estimate_executor() — unrented path ──────────────────────────────────────

@pytest.mark.asyncio
async def test_estimate_executor_unrented_h100_eligible(
    rental_config, mock_redis, mock_price_provider, monkeypatch
):
    """Eligible GPU (H100) unrented path returns positive usd_per_epoch."""
    # Arrange — two existing unrented H100 executors already in epoch
    job_results = {
        "miner_a": [_make_job("exec-a", "H100", 8, is_rented=False)],
        "miner_b": [_make_job("exec-b", "H100", 8, is_rented=False)],
    }
    incentive = _make_incentive(rental_config, mock_redis, mock_price_provider, job_results, monkeypatch)
    await incentive.calculate_mining_scores()
    snapshot = incentive.get_snapshot()

    # Act — estimate a new hypothetical 8xH100 unrented executor
    estimator = RentalPriceIncentive(
        rental_config,
        mock_redis,
        jobs_results={},
        total_gpu_model_count_map={},
        snapshot=snapshot,
    )
    result = await estimator.estimate_executor(ExecutorEstimateParams(gpu_model="H100", gpu_count=8, is_rented=False))

    # Assert
    assert isinstance(result, RentalPriceEstimate)
    assert result.gpu_model == "H100"
    assert result.base_model == "H100"
    assert result.is_rented is False
    assert result.eligible_for_rental_share is True
    # With rental cost present, usd_per_epoch should be positive
    assert result.usd_per_epoch > 0
    assert result.rental_share is not None
    assert result.effective_rate is not None
    assert result.unrented_cap_multiplier is not None


@pytest.mark.asyncio
async def test_estimate_executor_unrented_ineligible_gpu_returns_zero(
    rental_config, mock_redis, mock_price_provider, monkeypatch
):
    """GPU with cap=0 (e.g. A100, L4) is ineligible and returns usd_per_epoch=0."""
    # Arrange
    job_results = {"miner_a": [_make_job("exec-a", "H100", 8, is_rented=False)]}
    incentive = _make_incentive(rental_config, mock_redis, mock_price_provider, job_results, monkeypatch)
    await incentive.calculate_mining_scores()
    snapshot = incentive.get_snapshot()

    # Act — A100 has cap=0, ineligible
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

@pytest.mark.asyncio
async def test_estimate_executor_unrented_gpu_splitting(
    rental_config, mock_redis, mock_price_provider, monkeypatch
):
    """gpu_splitting changes the fake JobResult path used by estimate_executor()."""
    # Arrange
    rental_config.gpu_count_custom_prices = {
        "H100": {
            "8": 1.0,
            "1": 3.5,
        }
    }
    job_results = {"miner_a": [_make_job("exec-a", "H100", 8, is_rented=False)]}
    incentive = _make_incentive(rental_config, mock_redis, mock_price_provider, job_results, monkeypatch)
    await incentive.calculate_mining_scores()
    snapshot = incentive.get_snapshot()

    # Act — gpu_splitting should cause estimate_executor() to use the higher min-count rate.
    estimator_split = RentalPriceIncentive(
        rental_config,
        mock_redis,
        jobs_results={},
        total_gpu_model_count_map={},
        snapshot=snapshot,
    )
    result_split = await estimator_split.estimate_executor(
        ExecutorEstimateParams(gpu_model="H100", gpu_count=8, is_rented=False, gpu_splitting=True, gpu_splitting_min_count=1)
    )

    estimator_no_split = RentalPriceIncentive(
        rental_config,
        mock_redis,
        jobs_results={},
        total_gpu_model_count_map={},
        snapshot=snapshot,
    )
    result_no_split = await estimator_no_split.estimate_executor(ExecutorEstimateParams(gpu_model="H100", gpu_count=8, is_rented=False))

    # Assert — splitting takes the better 1-GPU rate, proving the fake JobResult received the new fields.
    assert result_no_split.hourly_rate == pytest.approx(1.0)
    assert result_split.hourly_rate == pytest.approx(3.5)
    assert result_split.effective_rate is not None
    assert result_no_split.effective_rate is not None
    assert result_split.effective_rate > result_no_split.effective_rate
    assert result_split.usd_per_epoch > 0
    assert result_no_split.usd_per_epoch > 0
    assert result_split.usd_per_epoch > result_no_split.usd_per_epoch


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

@pytest.mark.asyncio
async def test_snapshot_seeding_preserves_accumulated_state(
    rental_config, mock_redis, mock_price_provider, monkeypatch
):
    """New estimator seeded from snapshot starts with correct unrented counts."""
    # Arrange — run real epoch first
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", BASE_GPU_MAP)
    job_results = {
        "miner_a": [_make_job("exec-a", "H100", 8, is_rented=False)],
    }
    incentive = _make_incentive(rental_config, mock_redis, mock_price_provider, job_results, monkeypatch)
    await incentive.calculate_mining_scores()
    snapshot = incentive.get_snapshot()

    # Act — create a snapshot-seeded estimator
    seeded = RentalPriceIncentive(
        rental_config, mock_redis, {},
        total_gpu_model_count_map={},
        snapshot=snapshot,
    )

    # Assert — accumulated state from original epoch is present in seeded instance.
    # H100 cap is `int` (16) here, so the bucket sentinel is 0.
    assert seeded.unrented_count_by_bucket.get(("H100", 0)) == 8
    assert seeded._weighted_rate_sum_by_bucket.get(("H100", 0)) == pytest.approx(
        8 * RENTAL_PRICES_PER_HOUR["H100"]
    )
    assert seeded.epoch_subnet_emission == pytest.approx(TEMPO * TAO_PRICE * ALPHA_RATE)
    assert seeded.total_mining_score == pytest.approx(incentive.total_mining_score)


@pytest.mark.asyncio
async def test_estimate_executor_does_not_fetch_prices_when_snapshot_provided(
    rental_config, mock_redis, mock_price_provider, monkeypatch
):
    """estimate_executor() must NOT call get_tao_price/get_alpha_rate when seeded from a snapshot."""
    # Arrange — build a snapshot via a real scoring run
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", BASE_GPU_MAP)
    job_results = {"miner_a": [_make_job("exec-a", "H100", 8, is_rented=False)]}
    incentive = _make_incentive(rental_config, mock_redis, mock_price_provider, job_results, monkeypatch)
    await incentive.calculate_mining_scores()
    snapshot = incentive.get_snapshot()

    # Reset call counts after the initial scoring run
    mock_price_provider.get_tao_price.reset_mock()
    mock_price_provider.get_alpha_rate.reset_mock()

    # Act — estimate using the snapshot
    estimator = RentalPriceIncentive(
        rental_config, mock_redis, {},
        total_gpu_model_count_map={},
        snapshot=snapshot,
    )
    await estimator.estimate_executor(ExecutorEstimateParams(gpu_model="H100", gpu_count=8, is_rented=False))

    # Assert — price provider must not have been called during estimation
    mock_price_provider.get_tao_price.assert_not_called()
    mock_price_provider.get_alpha_rate.assert_not_called()


@pytest.mark.asyncio
async def test_precompute_all_estimates_returns_both_rented_and_unrented_without_gather(
    rental_config, mock_redis, monkeypatch
):
    """Precompute should fill both estimate slots sequentially without relying on asyncio.gather."""
    snapshot = RentalPriceSnapshot(
        epoch_subnet_emission=1.0,
        rental_share=0.2,
        burn_share=0.1,
        mining={
            "total_gpu_count": 1,
            "total_mining_score": 1.0,
            "total_gpu_model_count_map": {"H100": 1},
        },
        rental={
            "total_rental_cost": 10.0,
            "by_gpu_type": {
                "H100": GpuTypeRentalState(
                    unrented_count=1,
                    max_cap=16,
                    cap_multiplier=1.0,
                    weighted_rate_sum=3.5,
                )
            },
        },
    )
    estimates_by_key = {
        ("H100", False): RentalPriceEstimate(
            gpu_model="H100",
            base_model="H100",
            gpu_count=1,
            is_rented=False,
            usd_per_epoch=1.0,
        ),
        ("H100", True): RentalPriceEstimate(
            gpu_model="H100",
            base_model="H100",
            gpu_count=1,
            is_rented=True,
            usd_per_epoch=2.0,
        ),
        ("H100 NVL", False): RentalPriceEstimate(
            gpu_model="H100 NVL",
            base_model="H100",
            gpu_count=1,
            is_rented=False,
            usd_per_epoch=3.0,
        ),
        ("H100 NVL", True): RentalPriceEstimate(
            gpu_model="H100 NVL",
            base_model="H100",
            gpu_count=1,
            is_rented=True,
            usd_per_epoch=4.0,
        ),
        ("H200", False): RentalPriceEstimate(
            gpu_model="H200",
            base_model="H200",
            gpu_count=1,
            is_rented=False,
            usd_per_epoch=5.0,
        ),
        ("H200", True): RentalPriceEstimate(
            gpu_model="H200",
            base_model="H200",
            gpu_count=1,
            is_rented=True,
            usd_per_epoch=6.0,
        ),
        ("A100", False): RentalPriceEstimate(
            gpu_model="A100",
            base_model="A100",
            gpu_count=1,
            is_rented=False,
            usd_per_epoch=7.0,
        ),
        ("A100", True): RentalPriceEstimate(
            gpu_model="A100",
            base_model="A100",
            gpu_count=1,
            is_rented=True,
            usd_per_epoch=8.0,
        ),
        ("L4", False): RentalPriceEstimate(
            gpu_model="L4",
            base_model="L4",
            gpu_count=1,
            is_rented=False,
            usd_per_epoch=9.0,
        ),
        ("L4", True): RentalPriceEstimate(
            gpu_model="L4",
            base_model="L4",
            gpu_count=1,
            is_rented=True,
            usd_per_epoch=10.0,
        ),
    }
    estimate_executor_mock = AsyncMock(
        side_effect=lambda _config, _redis, _snapshot, params: estimates_by_key[(params.gpu_model, params.is_rented)]
    )

    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", BASE_GPU_MAP)
    monkeypatch.setattr(
        rental_price_module,
        "asyncio",
        SimpleNamespace(gather=AsyncMock(side_effect=AssertionError("asyncio.gather should not be used"))),
        raising=False,
    )
    monkeypatch.setattr(rental_price_module, "estimate_executor", estimate_executor_mock)

    estimates = await precompute_all_estimates(
        config=rental_config,
        snapshot=snapshot,
        redis_service=mock_redis,
    )

    assert set(estimates) == set(BASE_GPU_MAP)
    for gpu_model in BASE_GPU_MAP:
        assert estimates[gpu_model]["unrented"] == estimates_by_key[(gpu_model, False)]
        assert estimates[gpu_model]["rented"] == estimates_by_key[(gpu_model, True)]

    assert estimate_executor_mock.await_count == len(BASE_GPU_MAP) * 2
    assert estimate_executor_mock.await_args_list[0].args[3].gpu_model == "H100"
    assert estimate_executor_mock.await_args_list[0].args[3].is_rented is False
    assert estimate_executor_mock.await_args_list[1].args[3].gpu_model == "H100"
    assert estimate_executor_mock.await_args_list[1].args[3].is_rented is True
