"""DAH-3230: the RTX PRO 6000 Server Edition anchors the unrented incentive at the Workstation price.

The validator's `rental_prices_per_hour` comes from the installed lium-core table — `lium-core 0.1.8`
per `pdm.lock`, where Server is 0.86 and Workstation 1.0; `incentive.config.RENTAL_PRICES_PER_HOUR`
pins the two editions to parity so an idle Server node earns the same subsidy as an idle Workstation
node until the 1.0 table in this PR is released and picked up by the pin.
"""

from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG

from incentive.config import RENTAL_PRICES_PER_HOUR, IncentiveConfig
from incentive.utils import get_hourly_rate

SERVER = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
WORKSTATION = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"


def test_incentive_config_anchors_server_edition_at_workstation_price():
    """Fails when the parity override is dropped or mistyped: the Server entry then falls back to the
    installed lium-core value (0.86 in 0.1.8) and no longer equals the Workstation entry."""
    prices = IncentiveConfig().rental_prices_per_hour
    upstream = DEFAULT_SHARED_CONFIG.machine_prices

    assert prices[SERVER] == prices[WORKSTATION]
    assert prices[SERVER] == upstream[WORKSTATION]


def test_hourly_rate_is_the_same_for_both_editions_through_the_price_resolver():
    """Fails when the resolver stops reading the anchored table for one edition (a custom-price entry
    or a base-model mapping that treats the two editions differently)."""
    config = IncentiveConfig()
    workstation_anchor = config.rental_prices_per_hour[WORKSTATION]

    for gpu_count in (1, 8):
        server_rate = get_hourly_rate(
            SERVER, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour
        )
        workstation_rate = get_hourly_rate(
            WORKSTATION, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour
        )
        assert server_rate == workstation_rate == workstation_anchor


def test_parity_override_changes_only_the_server_edition_entry():
    """Holds by construction for today's `{**upstream, SERVER: upstream[WORKSTATION]}`; it guards a
    future hand-edit of `RENTAL_PRICES_PER_HOUR` that adds, drops or re-prices another GPU — the
    algorithm asserts every key is in BASE_GPU_MAP, and any other override belongs in lium-core."""
    upstream = DEFAULT_SHARED_CONFIG.machine_prices

    assert RENTAL_PRICES_PER_HOUR.keys() == upstream.keys()
    differing = {gpu for gpu in upstream if RENTAL_PRICES_PER_HOUR[gpu] != upstream[gpu]}
    assert differing <= {SERVER}
