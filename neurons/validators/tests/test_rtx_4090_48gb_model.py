"""DAH-5508: the RTX 4090 48GB is its own model, only with RTX_4090_48GB_MODEL_ENABLED on (default off)."""

import pytest

from core.config import settings
from incentive.config import BASE_GPU_MAP, MAX_UNRENTED_GPUS_BY_TYPE, RENTAL_PRICES_PER_HOUR
from services.const import GPU_MODEL_RATES
from services.gpu_precheck import VramRangeMismatchError, precheck_gpu_spec
from services.gpu_spec_table import RTX_4090, RTX_4090_48GB, classify_gpu_model
from services.task.checks.machine_spec_scrape import _normalize_gpu_details
from services.task.result_handler import _canonicalize_published_gpu_names


def test_the_setting_defaults_off():
    assert type(settings).model_fields["RTX_4090_48GB_MODEL_ENABLED"].default is False


@pytest.mark.parametrize("vram_mb", [49140, 46068])
def test_off_a_48gb_4090_is_still_a_plain_4090_and_rejected(vram_mb):
    model = classify_gpu_model(RTX_4090, vram_mb, False)
    assert model == RTX_4090
    with pytest.raises(VramRangeMismatchError):
        precheck_gpu_spec(model, vram_mb)


@pytest.mark.parametrize("vram_mb", [49140, 46068])
def test_on_a_48gb_4090_is_the_48gb_model_and_passes(vram_mb):
    model = classify_gpu_model(RTX_4090, vram_mb, True)
    assert model == RTX_4090_48GB
    precheck_gpu_spec(model, vram_mb)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("vram_mb", [24564, 23028])
def test_a_24gb_4090_is_unchanged_in_both_modes(enabled, vram_mb):
    model = classify_gpu_model(RTX_4090, vram_mb, enabled)
    assert model == RTX_4090
    precheck_gpu_spec(model, vram_mb)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("vram_mb", [36864, 32768, 65536])
def test_an_odd_capacity_4090_is_rejected_in_both_modes(enabled, vram_mb):
    with pytest.raises(VramRangeMismatchError):
        precheck_gpu_spec(classify_gpu_model(RTX_4090, vram_mb, enabled), vram_mb)


@pytest.mark.parametrize("enabled", [False, True])
def test_a_card_reporting_the_48gb_name_is_judged_by_its_vram(enabled):
    assert classify_gpu_model(RTX_4090_48GB, 24564, enabled) == RTX_4090
    assert classify_gpu_model(RTX_4090_48GB, 49140, enabled) == (RTX_4090_48GB if enabled else RTX_4090)


@pytest.mark.parametrize("enabled", [False, True])
def test_other_48gb_cards_are_untouched(enabled):
    assert classify_gpu_model("NVIDIA L40", 46068, enabled) == "NVIDIA L40"
    assert classify_gpu_model("NVIDIA GeForce RTX 4090 D", 49140, enabled) == "NVIDIA GeForce RTX 4090 D"


def test_the_48gb_model_has_its_own_economics():
    assert GPU_MODEL_RATES[RTX_4090_48GB] == 0.0
    assert RENTAL_PRICES_PER_HOUR[RTX_4090_48GB] == 0.48  # the backend MACHINE_PRICES
    assert BASE_GPU_MAP[RTX_4090_48GB] == "RTX 4090 48GB" != BASE_GPU_MAP[RTX_4090]
    assert MAX_UNRENTED_GPUS_BY_TYPE["RTX 4090 48GB"] == {1: 10, 8: 64}  # the default tier
    # the standard card keeps its numbers
    assert GPU_MODEL_RATES[RTX_4090] == 0.05
    assert MAX_UNRENTED_GPUS_BY_TYPE["RTX 4090"] == {1: 10, 8: 8}  # DAH-5971 demand cap


@pytest.mark.parametrize("enabled", [False, True])
def test_scrape_state_and_published_specs_follow_the_setting(monkeypatch, enabled):
    monkeypatch.setattr(settings, "RTX_4090_48GB_MODEL_ENABLED", enabled)
    details = [{"name": RTX_4090, "capacity": 49140}, {"name": RTX_4090, "capacity": 24564}]
    expected = [RTX_4090_48GB if enabled else RTX_4090, RTX_4090]
    assert [d["name"] for d in _normalize_gpu_details(details)] == expected
    published = _canonicalize_published_gpu_names({"gpu": {"details": details}})
    assert [d["name"] for d in published["gpu"]["details"]] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_published_estimates_follow_the_setting(monkeypatch, enabled):
    from incentive import rental_price

    monkeypatch.setattr(settings, "RTX_4090_48GB_MODEL_ENABLED", enabled)

    async def fake_estimate(config, redis_service, snapshot, params):
        return {"gpu_model": params.gpu_model}

    monkeypatch.setattr(rental_price, "estimate_executor", fake_estimate)
    estimates = await rental_price.precompute_all_estimates(None, None, None)
    assert (RTX_4090_48GB in estimates) is enabled
    assert RTX_4090 in estimates
    assert len(estimates) == len(BASE_GPU_MAP) - (0 if enabled else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_rented", [False, True])
async def test_off_an_on_demand_48gb_estimate_is_unavailable(monkeypatch, is_rented):
    # Arrange
    from unittest.mock import AsyncMock

    from incentive.rental_price import ExecutorEstimateParams, RentalPriceIncentive

    monkeypatch.setattr(settings, "RTX_4090_48GB_MODEL_ENABLED", False)
    estimator = RentalPriceIncentive(settings.incentive, AsyncMock(), jobs_results={}, total_gpu_model_count_map={})
    estimator._seed_snapshot = object()  # never read: the request stops before any computation

    # Act
    result = await estimator.estimate_executor(ExecutorEstimateParams(gpu_model=RTX_4090_48GB, is_rented=is_rented))
    unknown = await estimator.estimate_executor(ExecutorEstimateParams(gpu_model="NVIDIA Unknown", is_rented=is_rented))

    # Assert
    assert result.usd_per_epoch is None
    assert result.eligible_for_rental_share is False
    assert unknown.usd_per_epoch == 0.0  # other unknown models are unchanged
