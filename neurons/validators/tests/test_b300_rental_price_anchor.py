"""An idle B300 earns 3.20 USD/hour per GPU: the 6.40 market anchor times the unrented anchor multiplier.

The installed lium-core (0.1.8 per pdm.lock) still anchors B300 at 5.10, so
`incentive.config.MARKET_PRICES_PER_HOUR` pins it at 6.40 until a lium-core release carries it.
"""

from incentive.config import MARKET_PRICES_PER_HOUR, IncentiveConfig
from incentive.utils import get_hourly_rate

B300 = "NVIDIA B300 SXM6 AC"


def test_b300_market_anchor_is_pinned_at_6_40() -> None:
    assert MARKET_PRICES_PER_HOUR[B300] == 6.4


def test_idle_b300_hourly_rate_is_3_20_for_single_and_full_chassis() -> None:
    config: IncentiveConfig = IncentiveConfig()

    b300_hourly_rates_for_1_and_8_gpus: list[float] = [
        get_hourly_rate(B300, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour)
        for gpu_count in (1, 8)
    ]

    assert b300_hourly_rates_for_1_and_8_gpus == [3.2, 3.2]
