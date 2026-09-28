"""An idle B300 earns 1.25 USD/hour per GPU, at or under what Lium's filler workload earns on it.

The installed lium-core (0.1.8 per pdm.lock) anchors B300 at 5.10, so
`incentive.config.RENTAL_PRICES_PER_HOUR` pins the idle rate in the validator.
"""

import pytest

from incentive.config import IncentiveConfig
from incentive.utils import get_hourly_rate

B300_AC = "NVIDIA B300 SXM6 AC"
B300_PC = "NVIDIA B300 SXM6 PC"


@pytest.mark.parametrize("gpu_model", [B300_AC, B300_PC])
def test_idle_b300_hourly_rate_is_1_25_for_single_and_full_chassis(gpu_model: str) -> None:
    config: IncentiveConfig = IncentiveConfig()

    b300_hourly_rates_for_1_and_8_gpus: list[float] = [
        get_hourly_rate(gpu_model, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour)
        for gpu_count in (1, 8)
    ]

    assert b300_hourly_rates_for_1_and_8_gpus == [1.25, 1.25]
