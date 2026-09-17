"""DAH-3623: no idle rate above 0.8 x the median price renters paid for the model, and every pinned
model sits on that mark, rounded down to the cent.

Owner rule, 17 Sep 2026: "idle pay should never be higher than rental rates"; 21:41Z: "pin means raise
too". The fixture below is the 7-day paid median per GPU model read from prod `rental_history`
(per-rental `price_per_gpu`, rentals active 10 to 17 Sep 2026, read 2026-09-17 22:23Z) for every model
that received idle pay in the 24 h before the read and had at least 5 rentals in the window (fewer than
5 rentals is not a median to pin on).

A rate raised above the cap fails here; a pin that drifts off the mark fails here. When the market
moves, the commit that changes a rate also updates the fixture row (median, rentals,
`PAID_MEDIAN_AS_OF`) and says where the new number came from.
"""

import math

import pytest
from incentive.config import BASE_GPU_MAP, MAX_UNRENTED_GPUS_BY_TYPE, IncentiveConfig
from incentive.utils import get_hourly_rate

CAP_SHARE_OF_PAID_MEDIAN: float = 0.8
MIN_RENTALS_FOR_A_MEDIAN: int = 5
PAID_MEDIAN_AS_OF: str = "2026-09-17T22:23Z"

# gpu_model -> (median USD per GPU-hour renters paid over 7 days, rentals in the window)
PAID_MEDIAN_7D_USD_PER_GPU_HOUR: dict[str, tuple[float, int]] = {
    "NVIDIA A100 80GB PCIe": (0.30, 332),
    "NVIDIA H100 80GB HBM3": (1.30, 310),
    "NVIDIA GeForce RTX 5090": (0.60, 856),
    "NVIDIA RTX 6000 Ada Generation": (0.69, 195),
    "NVIDIA GeForce RTX 3090": (0.16, 495),
    "NVIDIA A100-SXM4-80GB": (0.70, 64),
    "NVIDIA H200": (3.00, 327),
    "NVIDIA L40S": (0.38, 261),
    "NVIDIA GeForce RTX 4090": (0.35, 876),
    "NVIDIA B300 SXM6 AC": (8.00, 128),
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": (1.25, 45),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": (1.25, 481),
    "NVIDIA B200": (5.60, 98),
    "NVIDIA RTX A6000": (0.42, 120),
    "NVIDIA H100 PCIe": (2.24, 34),
}

# Pinned at the cap by DAH-3623: commit 1 the three models whose lium-core rate was above the rental
# price itself, commit 2 (owner, 17 Sep 2026 16:56Z) the six that sat between 80 % and 100 % of it,
# commit 3 (owner, 21:41Z: "pin means raise too") the three that sat under 80 % of it, raised.
PINNED_AT_CAP: tuple[str, ...] = (
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA H200",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L40S",
    "NVIDIA B200",
    "NVIDIA RTX A6000",
    "NVIDIA H100 PCIe",
)


def _cap_usd_per_gpu_hour(gpu_model: str) -> float:
    median, rentals = PAID_MEDIAN_7D_USD_PER_GPU_HOUR[gpu_model]
    assert rentals >= MIN_RENTALS_FOR_A_MEDIAN, (
        f"{gpu_model}: a median on {rentals} rentals is not a cap; use a wider window or drop the row"
    )
    return CAP_SHARE_OF_PAID_MEDIAN * median


def _idle_rates_for_1_and_8_gpus(gpu_model: str) -> list[float]:
    config = IncentiveConfig()
    return [
        get_hourly_rate(
            gpu_model, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour
        )
        for gpu_count in (1, 8)
    ]


@pytest.mark.parametrize("gpu_model", sorted(PAID_MEDIAN_7D_USD_PER_GPU_HOUR))
def test_idle_rate_is_at_most_80_percent_of_the_paid_median(gpu_model: str) -> None:
    """Fails when a configured idle rate (in either count bucket) is above 0.8 x the paid median.
    On main before DAH-3623 it failed for nine models: A100 PCIe (0.36 > 0.24), H100 HBM3 (1.494 > 1.04),
    RTX 5090 (0.65 > 0.48), H200 (2.85 > 2.40), RTX 3090 (0.16 > 0.128), RTX 6000 Ada (0.69 > 0.552),
    A100 SXM (0.6923 > 0.56), RTX 4090 (0.30 > 0.28) and L40S (0.35 > 0.304)."""
    cap = _cap_usd_per_gpu_hour(gpu_model)

    for rate in _idle_rates_for_1_and_8_gpus(gpu_model):
        assert rate <= cap + 1e-9, (
            f"{gpu_model}: idle rate {rate} USD/GPU-h is above the cap {cap:.4f} "
            f"= {CAP_SHARE_OF_PAID_MEDIAN} x paid median {PAID_MEDIAN_7D_USD_PER_GPU_HOUR[gpu_model][0]} "
            f"(as of {PAID_MEDIAN_AS_OF})"
        )


@pytest.mark.parametrize("gpu_model", PINNED_AT_CAP)
def test_dah_3623_pins_sit_exactly_on_the_cap_rounded_down_to_the_cent(gpu_model: str) -> None:
    """Fails when a pin is mistyped (1.4 for 1.04) or drifts from the fixture it was derived from:
    the pinned rate must equal 0.8 x the paid median, rounded down to the cent, in both buckets.
    With main's config.py the three raised models fail this case and only this case (under the cap
    is legal, off the pin is not): B200 4.25 for 4.48, RTX A6000 0.32 for 0.33, H100 PCIe 1.1988
    for 1.79."""
    expected = math.floor(_cap_usd_per_gpu_hour(gpu_model) * 100 + 1e-9) / 100

    assert _idle_rates_for_1_and_8_gpus(gpu_model) == [expected, expected]


def test_fixture_names_only_models_that_can_receive_idle_pay() -> None:
    """Fails when a fixture key is misspelt or names a model with no idle-pay bucket: such a row
    would resolve to rate 0 and the cap check above would pass without checking anything."""
    for gpu_model in PAID_MEDIAN_7D_USD_PER_GPU_HOUR:
        base_model = BASE_GPU_MAP[gpu_model]
        assert any(cap > 0 for cap in MAX_UNRENTED_GPUS_BY_TYPE[base_model].values()), gpu_model
        assert all(rate > 0 for rate in _idle_rates_for_1_and_8_gpus(gpu_model)), gpu_model
