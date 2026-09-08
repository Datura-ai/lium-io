"""DAH-3230: the RTX PRO 6000 Server Edition anchors the unrented incentive at the Workstation price.

The validator's `rental_prices_per_hour` comes from the pinned lium-core table, which still says
Server 0.86 vs Workstation 1.0; `incentive.config.RENTAL_PRICES_PER_HOUR` pins the two editions to
parity so an idle Server node earns the same subsidy as an idle Workstation node.
"""

from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG

from incentive.config import RENTAL_PRICES_PER_HOUR, IncentiveConfig
from incentive.utils import get_hourly_rate

SERVER = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
WORKSTATION = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"


def test_incentive_config_anchors_server_edition_at_workstation_price():
    prices = IncentiveConfig().rental_prices_per_hour

    assert prices[SERVER] == prices[WORKSTATION] == 1.0


def test_hourly_rate_is_the_same_for_both_editions_through_the_price_resolver():
    config = IncentiveConfig()

    for gpu_count in (1, 8):
        server_rate = get_hourly_rate(
            SERVER, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour
        )
        workstation_rate = get_hourly_rate(
            WORKSTATION, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour
        )
        assert server_rate == workstation_rate == 1.0


def test_parity_override_changes_only_the_server_edition_entry():
    # The algorithm asserts every key is in BASE_GPU_MAP, so the override must not add or drop a GPU.
    upstream = DEFAULT_SHARED_CONFIG.machine_prices

    assert RENTAL_PRICES_PER_HOUR.keys() == upstream.keys()
    differing = {gpu for gpu in upstream if RENTAL_PRICES_PER_HOUR[gpu] != upstream[gpu]}
    assert differing <= {SERVER}
