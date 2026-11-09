"""DAH-6030: total idle pay at most k times realized Default Job revenue; the rest stays in the burn share."""

import logging
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from core.config import settings
import json
import time

from incentive.idle_pay_budget import DJ_REVENUE_KEY, MAX_READING_AGE_HOURS, store_default_job_revenue
from incentive.miner_incentive_log import ZeroIncentiveReason
from incentive.rental_price import RentalPriceIncentive
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

from tests.test_spot_node_pay import ALPHA_RATE, TAO_PRICE, _config, _node, _spot


def _redis() -> AsyncMock:
    store: dict = {}
    redis = AsyncMock()
    redis.get = AsyncMock(side_effect=lambda key: store.get(key))
    redis.set = AsyncMock(side_effect=lambda key, value, ex=None: store.__setitem__(key, value))
    return redis


async def _run(redis: AsyncMock) -> RentalPriceIncentive:
    # one Secure node (bucket pay) and one Spot node (unbucketed pay)
    incentive = RentalPriceIncentive(_config(), redis, {"hk": [_node("secure-1"), _spot(filler_revenue_per_gpu_hour=2.0)]}, {})
    incentive.price_provider = AsyncMock()
    incentive.price_provider.get_tao_price.return_value = TAO_PRICE
    incentive.price_provider.get_alpha_rate.return_value = ALPHA_RATE
    await incentive.calculate_mining_scores()
    return incentive


def _paid(incentive: RentalPriceIncentive) -> float:
    return sum(r.incentive for r in incentive.job_results["hk"])


def _usd_per_hour(incentive: RentalPriceIncentive) -> float:
    """USD/hour after weight normalization: with no mining score the vector sums to the burn emission only."""
    mass = incentive.total_burn_emission + (incentive.mining_share if incentive.total_mining_score > 0 else 0.0)
    return _paid(incentive) / incentive._share_for_rental_cost(1.0) / mass


@pytest_asyncio.fixture
async def uncapped(monkeypatch) -> RentalPriceIncentive:
    monkeypatch.setattr(settings, "ENABLE_SPOT_RENTAL_INCENTIVE", False)
    monkeypatch.setattr(settings, "SPOT_IDLE_PAY_SECURE_DECISION_ENABLED", False)
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", None)
    incentive = await _run(_redis())
    assert incentive.rental_share > 0 and incentive.unbucketed_share > 0
    return incentive


@pytest.mark.asyncio
async def test_budget_below_uncapped_scales_every_idle_payee_and_burns_the_rest(uncapped, monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.5)
    redis = _redis()
    await store_default_job_revenue(redis, _usd_per_hour(uncapped))  # budget = half of uncapped

    capped = await _run(redis)

    assert capped.idle_pay_scale == pytest.approx(0.5)
    for before, after in zip(uncapped.job_results["hk"], capped.job_results["hk"]):
        assert after.incentive == pytest.approx(before.incentive * 0.5)
    assert _usd_per_hour(capped) == pytest.approx(0.5 * _usd_per_hour(uncapped))
    assert capped.burn_share == pytest.approx(uncapped.burn_share + _paid(uncapped) / 2)


@pytest.mark.asyncio
async def test_budget_above_uncapped_changes_nothing(uncapped, monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    redis = _redis()
    await store_default_job_revenue(redis, 10 * _usd_per_hour(uncapped))

    capped = await _run(redis)

    assert capped.idle_pay_scale == 1.0
    assert _paid(capped) == pytest.approx(_paid(uncapped))
    assert capped.burn_share == pytest.approx(uncapped.burn_share)


@pytest.mark.asyncio
async def test_unreadable_revenue_keeps_the_last_good_budget(uncapped, monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 1.0)
    redis = _redis()
    await store_default_job_revenue(redis, 0.25 * _usd_per_hour(uncapped))
    await store_default_job_revenue(redis, None)  # this cycle's reply had no reading

    capped = await _run(redis)

    assert capped.idle_pay_scale == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_cold_start_without_any_reading_pays_unscaled_and_says_so(uncapped, monkeypatch, caplog):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)

    with caplog.at_level(logging.WARNING):
        capped = await _run(_redis())

    assert capped.idle_pay_scale == 1.0
    assert _paid(capped) == pytest.approx(_paid(uncapped))
    assert "no Default Job revenue reading" in caplog.text


@pytest.mark.asyncio
async def test_outage_zeros_keep_the_last_good_budget(uncapped, monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    redis = _redis()
    budget = 0.25 * _usd_per_hour(uncapped)
    for reading in (budget / 0.9, 0.0, 0.0, 0.0):  # good, then zeros from a revenue outage
        await store_default_job_revenue(redis, reading)

    capped = await _run(redis)

    assert _usd_per_hour(capped) == pytest.approx(budget)


@pytest.mark.asyncio
async def test_k_zero_pays_no_idle_incentive(uncapped, monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.0)
    redis = _redis()
    await store_default_job_revenue(redis, _usd_per_hour(uncapped))

    capped = await _run(redis)

    assert capped.idle_pay_scale == 0.0
    assert _paid(capped) == 0.0
    assert capped.burn_share == pytest.approx(capped.total_burn_emission)
    spot = next(r for r in capped.job_results["hk"] if r.spot_pay_candidate)
    assert spot.zero_incentive_reasons[0].reason == ZeroIncentiveReason.IDLE_PAY_BUDGET_ZERO


@pytest.mark.asyncio
async def test_a_single_zero_between_good_readings_does_not_stop_idle_pay(uncapped, monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    redis = _redis()
    budget = 0.25 * _usd_per_hour(uncapped)
    for reading in (budget / 0.9, 0.0, None):  # positive, then one zero, then unreadable
        await store_default_job_revenue(redis, reading)

    capped = await _run(redis)

    assert _usd_per_hour(capped) == pytest.approx(budget)


@pytest.mark.asyncio
async def test_a_positive_reading_older_than_a_day_keeps_capping(uncapped, monkeypatch):
    monkeypatch.setattr(settings, "IDLE_PAY_BUDGET_K", 0.9)
    redis = _redis()
    budget = 0.25 * _usd_per_hour(uncapped)
    old = time.time() - (MAX_READING_AGE_HOURS + 1) * 3600
    await redis.set(DJ_REVENUE_KEY, json.dumps({"usd_per_hour": budget / 0.9, "at": old}))

    capped = await _run(redis)

    assert _usd_per_hour(capped) == pytest.approx(budget)


@pytest.mark.asyncio
async def test_k_unset_is_off_whatever_the_revenue(uncapped):
    redis = _redis()
    await store_default_job_revenue(redis, 0.0)

    capped = await _run(redis)

    assert capped.idle_pay_scale == 1.0
    assert _paid(capped) == pytest.approx(_paid(uncapped))


@pytest.mark.parametrize("value", [None, "x", -1.0, float("nan"), float("inf"), True])
def test_a_bad_revenue_value_reads_as_unreadable_without_failing_the_reply(value):
    reply = RentedExecutorsResponse.model_validate({"executors": {}, "default_job_revenue_usd_per_hour": value})

    assert reply.default_job_revenue_usd_per_hour is None
