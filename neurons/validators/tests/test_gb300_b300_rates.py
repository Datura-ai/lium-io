"""GB300 is priced like B300 for now, until it has its own market data.

GB300 is a different card, so it is its own base model with its own idle tier at the B300 caps (1x: 4, 8x: 32); an
idle GB300 never fills the B300 tier. Its idle rate, score rate and VRAM size are the B300 AC entries.
"""

from unittest.mock import AsyncMock

import pytest
from incentive import config as incentive_config
from incentive.config import BASE_GPU_MAP, MAX_UNRENTED_GPUS_BY_TYPE, RENTAL_PRICES_PER_HOUR, IncentiveConfig
from incentive.rental_price import FLAGSHIP_CAPABILITY_BASE_MODELS, RentalPriceIncentive
from services import const, gpu_spec_table
from services.gpu_precheck import precheck_gpu_spec

B300 = "NVIDIA B300 SXM6 AC"
GB300 = "NVIDIA GB300"


@pytest.mark.parametrize(
    "table",
    [RENTAL_PRICES_PER_HOUR, const.GPU_MODEL_RATES, gpu_spec_table.GPU_VRAM_SIZES_MB],
    ids=["RENTAL_PRICES_PER_HOUR", "GPU_MODEL_RATES", "GPU_VRAM_SIZES_MB"],
)
def test_gb300_carries_the_b300_value(table) -> None:
    assert table[GB300] == table[B300]


def test_gb300_is_its_own_family_at_the_b300_caps() -> None:
    assert BASE_GPU_MAP[GB300] == "GB300"
    assert MAX_UNRENTED_GPUS_BY_TYPE["GB300"] == MAX_UNRENTED_GPUS_BY_TYPE["B300"] == {1: 4, 8: 32}
    assert MAX_UNRENTED_GPUS_BY_TYPE["GB300"] is not MAX_UNRENTED_GPUS_BY_TYPE["B300"]
    assert "GB300" in IncentiveConfig().rental_incentive_gpu_types
    assert "GB300" in FLAGSHIP_CAPABILITY_BASE_MODELS
    assert incentive_config.RENTAL_PRICES_PER_HOUR[GB300] == 6.4


def test_gb300_passes_the_precheck_at_the_b300_observed_total() -> None:
    assert precheck_gpu_spec(GB300, 275040) is None


@pytest.mark.asyncio
async def test_an_idle_gb300_gets_the_b300_rate_and_cap_in_its_own_tier(make_pcc_job) -> None:
    jobs = {
        "miner_gb300": [make_pcc_job("exec-gb300-1x", GB300, 1)],
        "miner_b300": [make_pcc_job("exec-b300-1x", B300, 1)],
    }
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)
    incentive = RentalPriceIncentive(
        IncentiveConfig(), redis, jobs, total_gpu_model_count_map={GB300: 1, B300: 1},
    )
    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 500.0
    price_provider.get_alpha_rate.return_value = 0.5
    incentive.price_provider = price_provider

    await incentive.calculate_mining_scores()

    gb300, b300 = jobs["miner_gb300"][0], jobs["miner_b300"][0]
    assert incentive.unrented_count_by_bucket[("GB300", 1)] == 1
    assert incentive.unrented_count_by_bucket[("B300", 1)] == 1
    assert gb300.max_cap == b300.max_cap == 4
    assert gb300.hourly_rate == b300.hourly_rate == pytest.approx(6.4)
    assert gb300.effective_rate == pytest.approx(b300.effective_rate)
