"""DAH-3648: the base listing price of every rental-backed model is its 30-day paid median.

Regression: machine_prices was anchored on the 30-day median on 2026-06-26 (DAH-2250) and then left
alone while the market moved. On 2026-09-18 B200 listed at a 4.25 base against a 5.60 median and
H100 HBM3 at 1.494 against 1.39, so the listing band (machine_min_price_rate x .. machine_max_price_rate x)
was centred on a price nobody paid. The median is the GPU-hour-weighted one the platform publishes as
`gpu_price_stat.lium_median_30d` (rental_history.price_per_gpu over rentals started in the trailing
30 days, each row weighted gpu_count x rental_hours, lower weighted median, floored to the cent) -- the
same definition DAH-2250 anchored on. The fixture below is the prod-replica read of 2026-09-19 03:24Z
that the table was set from; a per-rental (unweighted) median is NOT the same number (RTX 5090 0.60
per rental against 0.40 per GPU-hour; A100 PCIe 0.30 against 0.45). A price edit that leaves a model
off its dated median fails here; re-anchoring means re-reading the medians and replacing the fixture
with the new date.
"""

import math
from typing import NamedTuple

import pytest

from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG

MEDIAN_SNAPSHOT_DATE = "2026-09-19"
MIN_RENTALS_FOR_ANCHOR = 50

MACHINE_PRICES = DEFAULT_SHARED_CONFIG.machine_prices


class PaidMedian(NamedTuple):
    median_usd_per_gpu_hour: float
    rentals: int


# 30-day GPU-hour-weighted paid median per model (gpu_price_stat.lium_median_30d) and the rentals started
# in the window, as read on MEDIAN_SNAPSHOT_DATE 03:24Z.
PAID_MEDIAN_30D: dict[str, PaidMedian] = {
    "NVIDIA B300 SXM6 AC": PaidMedian(8.00, 335),
    "NVIDIA B200": PaidMedian(5.60, 615),
    "NVIDIA H200": PaidMedian(3.65, 881),
    "NVIDIA H100 80GB HBM3": PaidMedian(1.39, 679),
    "NVIDIA H100 PCIe": PaidMedian(1.30, 328),
    "NVIDIA GeForce RTX 5090": PaidMedian(0.40, 1953),
    "NVIDIA GeForce RTX 4090": PaidMedian(0.30, 2647),
    "NVIDIA RTX 6000 Ada Generation": PaidMedian(0.75, 371),
    "NVIDIA L40S": PaidMedian(0.38, 648),
    "NVIDIA L40": PaidMedian(0.33, 151),
    "NVIDIA A100 80GB PCIe": PaidMedian(0.45, 384),
    "NVIDIA A100-SXM4-80GB": PaidMedian(0.70, 91),
    "NVIDIA RTX A6000": PaidMedian(0.42, 329),
    "NVIDIA GeForce RTX 3090": PaidMedian(0.16, 1212),
    # thin: under MIN_RENTALS_FOR_ANCHOR rentals, the base is not anchored on them. The weighted read
    # covered the 16 anchored models only; these two rows carry the 2026-09-18 per-rental read, and it
    # is the rental count that classifies them.
    "NVIDIA H200 NVL": PaidMedian(3.90, 20),
    "NVIDIA L4": PaidMedian(0.27, 13),
}

# The two RTX PRO 6000 editions are one card for a renter and stay at parity (DAH-3230, owner
# 2026-09-08). Weighted medians read separately: Server Edition 1.19 over 942 rentals, Workstation
# Edition 1.00 over 99. The parity anchor is the Server Edition's median (942 of the 1,041 rentals);
# a pooled weighted median of both editions was not part of the read.
RTX_PRO_6000_SERVER = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
RTX_PRO_6000_WORKSTATION = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
RTX_PRO_6000_SERVER_MEDIAN = PaidMedian(1.19, 942)
RTX_PRO_6000_WORKSTATION_MEDIAN = PaidMedian(1.00, 99)

# Base values the thin models keep (the value before DAH-3648).
THIN_MODELS_KEEP: dict[str, float] = {
    "NVIDIA H200 NVL": 2.90,
    "NVIDIA L4": 0.11,
}

# The per-rental (unweighted) medians of the 2026-09-18 read that the table must NOT sit on where the
# two definitions differ (negative control for the definition, not only for the numbers).
UNWEIGHTED_MEDIAN_WHERE_IT_DIFFERS: dict[str, float] = {
    "NVIDIA H200": 3.25,
    "NVIDIA H100 80GB HBM3": 1.30,
    "NVIDIA H100 PCIe": 1.50,
    "NVIDIA GeForce RTX 5090": 0.60,
    "NVIDIA GeForce RTX 4090": 0.32,
    "NVIDIA RTX 6000 Ada Generation": 0.69,
    "NVIDIA A100 80GB PCIe": 0.30,
    "NVIDIA A100-SXM4-80GB": 0.68,
    "NVIDIA GeForce RTX 3090": 0.18,
}


def floor_cent(value: float) -> float:
    return math.floor(value * 100 + 1e-9) / 100


ANCHORED = sorted(model for model, m in PAID_MEDIAN_30D.items() if m.rentals >= MIN_RENTALS_FOR_ANCHOR)
THIN = sorted(model for model, m in PAID_MEDIAN_30D.items() if m.rentals < MIN_RENTALS_FOR_ANCHOR)


@pytest.mark.parametrize("model", ANCHORED)
def test_base_price_is_the_30_day_paid_median(model: str) -> None:
    median = PAID_MEDIAN_30D[model]

    assert MACHINE_PRICES[model] == pytest.approx(floor_cent(median.median_usd_per_gpu_hour)), (
        f"{model}: base {MACHINE_PRICES[model]} is off its {MEDIAN_SNAPSHOT_DATE} GPU-hour-weighted paid "
        f"median {median.median_usd_per_gpu_hour} ({median.rentals} rentals)"
    )


@pytest.mark.parametrize("model", sorted(UNWEIGHTED_MEDIAN_WHERE_IT_DIFFERS))
def test_base_price_is_not_the_per_rental_median(model: str) -> None:
    """The definition is GPU-hour-weighted; a table set from the per-rental median fails here."""
    assert MACHINE_PRICES[model] != pytest.approx(UNWEIGHTED_MEDIAN_WHERE_IT_DIFFERS[model]), (
        f"{model}: base {MACHINE_PRICES[model]} is the per-rental median, not the GPU-hour-weighted one"
    )


@pytest.mark.parametrize("model", THIN)
def test_thin_models_keep_their_base(model: str) -> None:
    assert MACHINE_PRICES[model] == pytest.approx(THIN_MODELS_KEEP[model])


def test_rtx_pro_6000_editions_sit_at_the_server_edition_median_in_parity() -> None:
    assert RTX_PRO_6000_SERVER_MEDIAN.rentals >= MIN_RENTALS_FOR_ANCHOR
    assert MACHINE_PRICES[RTX_PRO_6000_SERVER] == MACHINE_PRICES[RTX_PRO_6000_WORKSTATION]
    assert MACHINE_PRICES[RTX_PRO_6000_SERVER] == pytest.approx(
        floor_cent(RTX_PRO_6000_SERVER_MEDIAN.median_usd_per_gpu_hour)
    )
