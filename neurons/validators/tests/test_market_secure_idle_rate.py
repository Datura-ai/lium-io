"""DAH-5934: the backend sends an idle Secure rate per full GPU name (GPU market x 0.6); the validator pays at it in
place of its own table for that name, keeps the table for every other name, and estimates price the same way."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core import validator as validator_module
from incentive import rental_price as rental_price_module
from incentive.config import DEFAULT_PRICE, IncentiveConfig
from incentive.rental_price import ExecutorEstimateParams, RentalPriceIncentive, estimate_executor
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from test_rental_price_estimate import BASE_GPU_MAP, _make_incentive, _make_job

pytest_plugins = ["fixtures.incentive_fixtures"]


@pytest.fixture
def config() -> IncentiveConfig:
    return IncentiveConfig(
        algorithm="rental_price",
        rental_incentive_gpu_types=["H100", "H200"],
        max_unrented_gpus={"H100": {1: 16, 8: 16}, "H200": {1: 8, 8: 8}},
        rental_prices_per_hour={"H100": 3.5, "H200": 4.0},
        gpu_count_custom_prices={"*": {"*": 0, "1": DEFAULT_PRICE, "8": DEFAULT_PRICE}},
    )


@pytest.fixture
def price_provider():
    provider = AsyncMock()
    provider.get_tao_price.return_value = 500.0
    provider.get_alpha_rate.return_value = 0.5
    return provider


def test_a_bad_rate_is_dropped_and_the_reply_still_reads() -> None:
    reply = RentedExecutorsResponse.model_validate(
        {"executors": {}, "secure_idle_rate_by_gpu_model": {"H100": 2.35, "H200": -1, "A100": "x", "L4": float("nan")}}
    )
    assert reply.secure_idle_rate_by_gpu_model == {"H100": 2.35}
    assert RentedExecutorsResponse.model_validate({"executors": {}}).secure_idle_rate_by_gpu_model == {}


def test_the_market_rate_replaces_the_table_for_its_name_only(config, monkeypatch) -> None:
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", BASE_GPU_MAP)
    market = config.model_copy(update={"market_rental_prices_per_hour": {"H100": 2.35}})
    incentive = RentalPriceIncentive(market, AsyncMock(), {}, total_gpu_model_count_map={})

    assert incentive._listed_hourly_rate(_make_job("a", "H100", 8, is_rented=False)) == 2.35
    assert incentive._listed_hourly_rate(_make_job("b", "H200", 8, is_rented=False)) == 4.0
    # the table's per-count eligibility still holds: a 3-GPU node has no idle rate
    assert incentive._listed_hourly_rate(_make_job("c", "H100", 3, is_rented=False)) == 0.0


@pytest.mark.asyncio
async def test_an_estimate_prices_at_the_rates_its_snapshot_carries(config, price_provider, monkeypatch) -> None:
    market = config.model_copy(update={"market_rental_prices_per_hour": {"H100": 2.35}})
    incentive = _make_incentive(
        market, AsyncMock(), price_provider, {"m": [_make_job("a", "H100", 8, is_rented=False)]}, monkeypatch
    )
    await incentive.calculate_mining_scores()
    snapshot = incentive.get_snapshot()
    assert snapshot.market_rental_prices_per_hour == {"H100": 2.35}

    # another process holds only the table config; the snapshot brings the cycle's market rates
    estimate = await estimate_executor(
        config, AsyncMock(), snapshot, ExecutorEstimateParams(gpu_model="H100", gpu_count=8, is_rented=False)
    )
    assert estimate.hourly_rate == 2.35


def test_each_cycle_takes_the_backends_rates_for_names_the_validator_knows(config, monkeypatch) -> None:
    monkeypatch.setattr(validator_module, "BASE_GPU_MAP", BASE_GPU_MAP)
    fake = SimpleNamespace(incentive=config, default_extra={})
    reply = RentedExecutorsResponse.model_validate(
        {"executors": {}, "secure_idle_rate_by_gpu_model": {"H100": 2.35, "Unknown GPU": 9.0}}
    )
    validator_module.Validator.apply_market_secure_rates(fake, reply)
    assert config.market_rental_prices_per_hour == {"H100": 2.35}

    # a backend that sends none puts every name back on the table
    validator_module.Validator.apply_market_secure_rates(fake, RentedExecutorsResponse(executors={}))
    assert config.market_rental_prices_per_hour == {}


def test_the_payout_log_names_the_rate_source() -> None:
    from incentive.miner_incentive_log import MinerLogLine

    job = _make_job("a", "H100", 8, is_rented=False)
    job.hourly_rate = 2.35
    line = MinerLogLine.rental_incentive_calculated("hk", job, 8, hourly_rate_source="market")
    assert line.fields["hourly_rate_source"] == "market"
