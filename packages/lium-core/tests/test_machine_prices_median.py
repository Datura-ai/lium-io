"""DAH-3648: the base listing price of every rental-backed model is its 30-day paid median.

Regression: machine_prices was anchored on the 30-day median on 2026-06-26 (DAH-2250) and then left
alone while the market moved. On 2026-09-18 B200 listed at a 4.25 base against a 5.60 median and
H100 HBM3 at 1.494 against 1.30, so the listing band (machine_min_price_rate x .. machine_max_price_rate x)
was centred on a price nobody paid. The fixture below is the tsdb read the table was set from
(rental_history.price_per_gpu, one row per rental, rentals active in the trailing 30 days,
percentile_cont(0.5), floored to the cent). A price edit that leaves a model off its dated median
fails here; re-anchoring means re-reading the medians and replacing the fixture with the new date.
"""

import math
from typing import NamedTuple

import pytest

from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG

MEDIAN_SNAPSHOT_DATE = "2026-09-18"
MIN_RENTALS_FOR_ANCHOR = 50

MACHINE_PRICES = DEFAULT_SHARED_CONFIG.machine_prices


class PaidMedian(NamedTuple):
    median_usd_per_gpu_hour: float
    rentals: int


# 30-day paid median per model as read on MEDIAN_SNAPSHOT_DATE 09:08Z.
PAID_MEDIAN_30D: dict[str, PaidMedian] = {
    "NVIDIA B300 SXM6 AC": PaidMedian(8.00, 345),
    "NVIDIA B200": PaidMedian(5.60, 634),
    "NVIDIA H200": PaidMedian(3.25, 912),
    "NVIDIA H100 80GB HBM3": PaidMedian(1.30, 699),
    "NVIDIA H100 PCIe": PaidMedian(1.50, 333),
    "NVIDIA GeForce RTX 5090": PaidMedian(0.60, 1951),
    "NVIDIA GeForce RTX 4090": PaidMedian(0.32, 2679),
    "NVIDIA RTX 6000 Ada Generation": PaidMedian(0.69, 401),
    "NVIDIA L40S": PaidMedian(0.38, 676),
    "NVIDIA L40": PaidMedian(0.33, 169),
    "NVIDIA A100 80GB PCIe": PaidMedian(0.30, 399),
    "NVIDIA A100-SXM4-80GB": PaidMedian(0.68, 127),
    "NVIDIA RTX A6000": PaidMedian(0.42, 369),
    "NVIDIA GeForce RTX 3090": PaidMedian(0.18, 1235),
    # thin: under MIN_RENTALS_FOR_ANCHOR rentals, the base is not anchored on them
    "NVIDIA H200 NVL": PaidMedian(3.90, 20),
    "NVIDIA L4": PaidMedian(0.27, 13),
}

# The two RTX PRO 6000 editions are one card for a renter and stay at parity (DAH-3230, owner
# 2026-09-08); the anchor is the pooled 30-day median of both editions: 1,105 rentals.
RTX_PRO_6000_SERVER = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
RTX_PRO_6000_WORKSTATION = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
RTX_PRO_6000_POOLED_MEDIAN = PaidMedian(1.25, 1105)

# Base values the thin models keep (the value before DAH-3648).
THIN_MODELS_KEEP: dict[str, float] = {
    "NVIDIA H200 NVL": 2.90,
    "NVIDIA L4": 0.11,
}


def floor_cent(value: float) -> float:
    return math.floor(value * 100 + 1e-9) / 100


ANCHORED = sorted(model for model, m in PAID_MEDIAN_30D.items() if m.rentals >= MIN_RENTALS_FOR_ANCHOR)
THIN = sorted(model for model, m in PAID_MEDIAN_30D.items() if m.rentals < MIN_RENTALS_FOR_ANCHOR)


@pytest.mark.parametrize("model", ANCHORED)
def test_base_price_is_the_30_day_paid_median(model: str) -> None:
    median = PAID_MEDIAN_30D[model]

    assert MACHINE_PRICES[model] == pytest.approx(floor_cent(median.median_usd_per_gpu_hour)), (
        f"{model}: base {MACHINE_PRICES[model]} is off its {MEDIAN_SNAPSHOT_DATE} paid median "
        f"{median.median_usd_per_gpu_hour} ({median.rentals} rentals)"
    )


@pytest.mark.parametrize("model", THIN)
def test_thin_models_keep_their_base(model: str) -> None:
    assert MACHINE_PRICES[model] == pytest.approx(THIN_MODELS_KEEP[model])


def test_rtx_pro_6000_editions_sit_at_the_pooled_median_in_parity() -> None:
    assert MACHINE_PRICES[RTX_PRO_6000_SERVER] == MACHINE_PRICES[RTX_PRO_6000_WORKSTATION]
    assert MACHINE_PRICES[RTX_PRO_6000_SERVER] == pytest.approx(
        floor_cent(RTX_PRO_6000_POOLED_MEDIAN.median_usd_per_gpu_hour)
    )
