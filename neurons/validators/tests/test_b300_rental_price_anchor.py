"""DAH-3542: an idle B300 earns 6.40 USD/hour per GPU.

The installed lium-core (0.1.8 per pdm.lock) still anchors B300 at 5.10, so
`incentive.config.RENTAL_PRICES_PER_HOUR` pins it until a lium-core release carries 6.40.
"""

from incentive.config import IncentiveConfig
from incentive.utils import get_hourly_rate

B300 = "NVIDIA B300 SXM6 AC"


def test_idle_b300_hourly_rate_is_6_40_for_single_and_full_chassis() -> None:
    config: IncentiveConfig = IncentiveConfig()

    b300_hourly_rates_for_1_and_8_gpus: list[float] = [
        get_hourly_rate(B300, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour)
        for gpu_count in (1, 8)
    ]

    assert b300_hourly_rates_for_1_and_8_gpus == [6.4, 6.4]
