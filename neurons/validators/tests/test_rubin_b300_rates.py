"""Vera Rubin (DAH-4506) launches like RUBIN: its own base model and idle tier at the B300 rate and caps
(1x: 4, 8x: 32), so an idle Rubin never fills the B300 tier. The rate is pending the art-18 ruling.
"""

from unittest.mock import AsyncMock

import pytest
from incentive import config as incentive_config
from incentive.config import BASE_GPU_MAP, MAX_UNRENTED_GPUS_BY_TYPE, RENTAL_PRICES_PER_HOUR, IncentiveConfig
from incentive.rental_price import FLAGSHIP_CAPABILITY_BASE_MODELS, RentalPriceIncentive
from services import const, gpu_spec_table
from services.gpu_precheck import precheck_gpu_spec

B300 = "NVIDIA B300 SXM6 AC"
RUBIN = "NVIDIA Rubin"


@pytest.mark.parametrize(
    "table",
    [RENTAL_PRICES_PER_HOUR, const.GPU_MODEL_RATES, gpu_spec_table.GPU_VRAM_SIZES_MB],
    ids=["RENTAL_PRICES_PER_HOUR", "GPU_MODEL_RATES", "GPU_VRAM_SIZES_MB"],
)
def test_rubin_carries_the_b300_value(table) -> None:
    assert table[RUBIN] == table[B300]


def test_rubin_is_its_own_family_at_the_b300_caps() -> None:
    assert BASE_GPU_MAP[RUBIN] == "Rubin"
    assert MAX_UNRENTED_GPUS_BY_TYPE["Rubin"] == MAX_UNRENTED_GPUS_BY_TYPE["B300"] == {1: 4, 8: 32}
    assert MAX_UNRENTED_GPUS_BY_TYPE["Rubin"] is not MAX_UNRENTED_GPUS_BY_TYPE["B300"]
    assert "Rubin" in IncentiveConfig().rental_incentive_gpu_types
    assert "Rubin" in FLAGSHIP_CAPABILITY_BASE_MODELS
    assert incentive_config.RENTAL_PRICES_PER_HOUR[RUBIN] == 6.4


def test_rubin_passes_the_precheck_at_the_b300_observed_total() -> None:
    assert precheck_gpu_spec(RUBIN, 275040) is None


@pytest.mark.asyncio
async def test_an_idle_rubin_gets_the_b300_rate_and_cap_in_its_own_tier(make_pcc_job) -> None:
    jobs = {
        "miner_rubin": [make_pcc_job("exec-rubin-1x", RUBIN, 1)],
        "miner_b300": [make_pcc_job("exec-b300-1x", B300, 1)],
    }
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)
    incentive = RentalPriceIncentive(
        IncentiveConfig(), redis, jobs, total_gpu_model_count_map={RUBIN: 1, B300: 1},
    )
    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 500.0
    price_provider.get_alpha_rate.return_value = 0.5
    incentive.price_provider = price_provider

    await incentive.calculate_mining_scores()

    rubin, b300 = jobs["miner_rubin"][0], jobs["miner_b300"][0]
    assert incentive.unrented_count_by_bucket[("Rubin", 1)] == 1
    assert incentive.unrented_count_by_bucket[("B300", 1)] == 1
    assert rubin.max_cap == b300.max_cap == 4
    assert rubin.hourly_rate == b300.hourly_rate == pytest.approx(6.4)
    assert rubin.effective_rate == pytest.approx(b300.effective_rate)
