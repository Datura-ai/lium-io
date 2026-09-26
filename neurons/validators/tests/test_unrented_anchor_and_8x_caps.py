"""The unrented incentive pays half the market anchor per idle GPU and the 8x bucket caps are doubled.

Twice the idle 8x slots at half the hourly pay each: the most a full 8x bucket can earn
(anchor x cap) is unchanged, a fleet at or under the old cap earns half, and a fleet
above the old cap is diluted less.
"""

from unittest.mock import AsyncMock

import pytest
from incentive.config import (
    MARKET_PRICES_PER_HOUR,
    MAX_UNRENTED_GPUS_BY_TYPE,
    UNRENTED_ANCHOR_MULTIPLIER,
    IncentiveConfig,
)
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult

H200 = "NVIDIA H200"
IDLE_8X_H200_NODES = 12  # 96 GPUs: over the old 64-GPU cap, under the new 128


def test_unrented_anchor_multiplier_is_one_half() -> None:
    assert UNRENTED_ANCHOR_MULTIPLIER == 0.5
    prices = IncentiveConfig().rental_prices_per_hour
    assert prices["NVIDIA B300 SXM6 AC"] == 3.2
    assert prices[H200] == MARKET_PRICES_PER_HOUR[H200] / 2


def test_8x_caps_are_doubled_and_1x_caps_are_not() -> None:
    assert MAX_UNRENTED_GPUS_BY_TYPE["B300"] == {1: 4, 8: 64}
    eligible = {gpu: cap for gpu, cap in MAX_UNRENTED_GPUS_BY_TYPE.items() if cap and gpu != "B300"}
    assert eligible
    assert all(cap == {1: 10, 8: 128} for cap in eligible.values())


def test_full_8x_bucket_can_earn_the_same_hourly_total_as_before_the_change() -> None:
    config = IncentiveConfig()
    for gpu_model, base_model in (("NVIDIA B300 SXM6 AC", "B300"), (H200, "H200")):
        new_max = config.rental_prices_per_hour[gpu_model] * config.max_unrented_gpus[base_model][8]
        old_max = MARKET_PRICES_PER_HOUR[gpu_model] * (config.max_unrented_gpus[base_model][8] // 2)
        assert new_max == pytest.approx(old_max)


@pytest.mark.asyncio
async def test_96_idle_8x_h200_gpus_are_no_longer_diluted_and_earn_half_the_market_anchor(make_pcc_job) -> None:
    jobs: dict[str, list[JobResult]] = {
        f"miner_{i}": [make_pcc_job(f"exec-8x-{i:02d}", H200, 8)] for i in range(IDLE_8X_H200_NODES)
    }
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)
    incentive = RentalPriceIncentive(
        IncentiveConfig(), redis, jobs,
        total_gpu_model_count_map={H200: 8 * IDLE_8X_H200_NODES},
    )
    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 500.0
    price_provider.get_alpha_rate.return_value = 0.5
    incentive.price_provider = price_provider

    await incentive.calculate_mining_scores()

    assert incentive.unrented_count_by_bucket[("H200", 8)] == 96
    assert incentive.cap_multiplier_by_bucket[("H200", 8)] == pytest.approx(1.0)
    assert incentive.total_rental_cost == pytest.approx(96 * MARKET_PRICES_PER_HOUR[H200] / 2)
    for jobs_of_miner in jobs.values():
        result = jobs_of_miner[0]
        assert result.max_cap == 128
        assert result.cap_dilution_applied is False
        assert result.effective_rate == pytest.approx(MARKET_PRICES_PER_HOUR[H200] / 2)
