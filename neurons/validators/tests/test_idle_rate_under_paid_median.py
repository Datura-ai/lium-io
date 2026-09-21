"""DAH-3623: no idle rate above the median price renters paid for the model, and every pinned model
sits on that median, rounded down to the cent.

Owner rule, 17 Sep 2026: "idle pay should never be higher than rental rates"; 21:41Z: "pin means raise
too"; 18 Sep 2026 09:14Z: "pin the idle to median now" (the pin was 0.8 x the median until then).
Window: 30 days (Rustam, 18 Sep 2026 06:20Z on #1401: "fixed rates from a 7-day window are too
unstable"). Definition: the GPU-hour-weighted median the platform publishes as
`gpu_price_stat.lium_median_30d` (DAH-2250's anchor; PRICE-MEDIAN-DEFINITION default (B), 20 Sep 2026
12:00Z, P144): `rental_history.price_per_gpu` over the rentals started in the trailing 30 days, each
row weighted `gpu_count x rental_hours`, lower weighted median. The fixture below is the prod-replica
read of 2026-09-19 03:24Z (lium-io#1407, review thread r4052066252) for every model that received idle
pay in the 24 h before the 18 Sep read and had at least 5 rentals in the window (fewer than 5 rentals
is not a median to pin on). A per-rental (unweighted) median is a different number for nine of the
twelve pinned models (RTX 5090 0.60 per rental against 0.40 per GPU-hour; A100 PCIe 0.30 against 0.45).

Three fixture models are not pinned and sit at or under their median by an earlier decision: B300
(6.40 against an 8.00 median, DAH-3542) and the two RTX PRO 6000 editions (1.00 against 1.19 Server /
1.00 Workstation -- the Workstation Edition sits exactly on its own median -- held at parity by
DAH-3230, test_rental_price_anchor_parity.py). They pass the cap and are named in
NOT_PINNED_UNDER_THE_MEDIAN so a new fixture row has to be classified one way or the other.

The paid median is the only cap. A second cap at the model's base price (lium-core `machine_prices`)
was tried in commit 5 and withdrawn in commit 6 on the owner's word (Fish, 18 Sep 2026 08:53Z: "no
don't do that"); the base listing prices themselves move to the median under P186 instead.

A rate raised above the median fails here; a pin that drifts off the median fails here. When the
market moves, the commit that changes a rate also updates the fixture row (median, rentals,
`PAID_MEDIAN_AS_OF`) and says where the new number came from.
"""

import math
from typing import NamedTuple

import pytest
from incentive.config import BASE_GPU_MAP, MAX_UNRENTED_GPUS_BY_TYPE, IncentiveConfig
from incentive.utils import get_hourly_rate

CAP_SHARE_OF_PAID_MEDIAN: float = 1.0  # owner, 18 Sep 2026 09:14Z: the pin IS the median (was 0.8)
MIN_RENTALS_FOR_A_MEDIAN: int = 5
PAID_MEDIAN_AS_OF: str = "2026-09-19T03:24Z"


class PaidMedian(NamedTuple):
    """One fixture row: the two quantities are named so a swapped pair (399 USD, 0.30 rentals) cannot
    pass the sample-size check and set a 399 USD cap."""

    median_usd_per_gpu_hour: float
    rentals: int


# gpu_model -> the GPU-hour-weighted median USD per GPU-hour renters paid over 30 days
# (gpu_price_stat.lium_median_30d) and the rentals started in the window
PAID_MEDIAN_30D: dict[str, PaidMedian] = {
    "NVIDIA A100 80GB PCIe": PaidMedian(median_usd_per_gpu_hour=0.45, rentals=384),
    "NVIDIA H100 80GB HBM3": PaidMedian(median_usd_per_gpu_hour=1.39, rentals=679),
    "NVIDIA GeForce RTX 5090": PaidMedian(median_usd_per_gpu_hour=0.40, rentals=1953),
    "NVIDIA RTX 6000 Ada Generation": PaidMedian(median_usd_per_gpu_hour=0.75, rentals=371),
    "NVIDIA GeForce RTX 3090": PaidMedian(median_usd_per_gpu_hour=0.16, rentals=1212),
    "NVIDIA A100-SXM4-80GB": PaidMedian(median_usd_per_gpu_hour=0.70, rentals=91),
    "NVIDIA H200": PaidMedian(median_usd_per_gpu_hour=3.65, rentals=881),
    "NVIDIA L40S": PaidMedian(median_usd_per_gpu_hour=0.38, rentals=648),
    "NVIDIA GeForce RTX 4090": PaidMedian(median_usd_per_gpu_hour=0.30, rentals=2647),
    "NVIDIA B300 SXM6 AC": PaidMedian(median_usd_per_gpu_hour=8.00, rentals=335),
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": PaidMedian(median_usd_per_gpu_hour=1.00, rentals=99),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": PaidMedian(median_usd_per_gpu_hour=1.19, rentals=942),
    "NVIDIA B200": PaidMedian(median_usd_per_gpu_hour=5.60, rentals=615),
    "NVIDIA RTX A6000": PaidMedian(median_usd_per_gpu_hour=0.42, rentals=329),
    "NVIDIA H100 PCIe": PaidMedian(median_usd_per_gpu_hour=1.30, rentals=328),
}

# Models that can receive idle pay but have no fixture row. The 30-day read (2026-09-18 08:06Z, prod
# replica, read-only; query and output in the loop's private folder, on request) found: H100 NVL and
# RTX 4090 D with zero rentals and zero idle pay; H200 NVL with 20 rentals (per-rental median 3.90,
# rate 2.90, under the median, outside DAH-3623's twelve; the 19 Sep weighted read covered the 16
# anchored models only). A model added here without a reason in this comment is a mistake.
IDLE_ELIGIBLE_MODELS_WITHOUT_A_MEDIAN_ROW: tuple[str, ...] = (
    "NVIDIA H200 NVL",
    "NVIDIA H100 NVL",
    "NVIDIA GeForce RTX 4090 D",
)

# The recorded exception: idle-eligible models whose rate sits ABOVE their 30-day paid median and are
# not pinned by DAH-3623 (outside the twelve the ticket names). L40: 151 rentals, weighted median 0.33
# (0.33 per rental too), rate 0.36, 14 idle-pay ledger rows in 14 days, about 0.1 USD/day of idle pay
# over the median. Pinning it
# is Rustam's call on #1401; the day it is pinned the row moves to PAID_MEDIAN_30D + PINNED_AT_CAP and
# leaves this dict, or the test below fails.
ABOVE_THE_MEDIAN_NOT_YET_PINNED: dict[str, PaidMedian] = {
    "NVIDIA L40": PaidMedian(median_usd_per_gpu_hour=0.33, rentals=151),
}

# Fixture models that are NOT pinned to their median and sit at or under it by an earlier decision
# (module docstring): B300 6.40 (DAH-3542) and the two RTX PRO 6000 editions at Workstation parity 1.00
# (DAH-3230; the Workstation Edition's own median is 1.00).
NOT_PINNED_UNDER_THE_MEDIAN: tuple[str, ...] = (
    "NVIDIA B300 SXM6 AC",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
)

# Pinned at the median by DAH-3623: commit 1 the three models whose lium-core rate was above the rental
# price itself, commit 2 (owner, 17 Sep 2026 16:56Z) the six that sat between 88 % and 102 % of it
# on the 30-day medians (A100 SXM 102 %), commit 3 (owner, 21:41Z: "pin means raise too") the three
# that sat under 80 % of it; commit 6 (owner, 18 Sep 09:14Z) moves all twelve from 0.8 x to 1.0 x;
# commit 8 (PRICE-MEDIAN-DEFINITION default (B), 20 Sep 12:00Z) re-reads the twelve GPU-hour-weighted.
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
    row = PAID_MEDIAN_30D[gpu_model]
    assert row.rentals >= MIN_RENTALS_FOR_A_MEDIAN, (
        f"{gpu_model}: a median on {row.rentals} rentals is not a cap; use a wider window or drop the row"
    )
    return CAP_SHARE_OF_PAID_MEDIAN * row.median_usd_per_gpu_hour


def _configured_gpu_counts(gpu_model: str, config: IncentiveConfig) -> list[int]:
    """The GPU counts the model's price config names (its own entry, else the "*" fallback), so an
    override added at another count is checked too; "*" itself is a count of 0 and is skipped."""
    gpu_config = config.gpu_count_custom_prices.get(gpu_model) or config.gpu_count_custom_prices["*"]
    counts = sorted(int(count) for count in gpu_config if count != "*")
    assert counts, f"{gpu_model}: no GPU count in gpu_count_custom_prices"
    return counts


def _idle_rates_per_configured_gpu_count(gpu_model: str) -> list[float]:
    """One rate per GPU count the model's price config names (1 and 8 today)."""
    config = IncentiveConfig()
    return [
        get_hourly_rate(
            gpu_model, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour
        )
        for gpu_count in _configured_gpu_counts(gpu_model, config)
    ]


def _models_that_can_receive_idle_pay() -> list[str]:
    """Every model in BASE_GPU_MAP whose bucket has a positive cap and whose rate resolves above 0."""
    return sorted(
        gpu_model
        for gpu_model, base_model in BASE_GPU_MAP.items()
        if any(cap > 0 for cap in MAX_UNRENTED_GPUS_BY_TYPE[base_model].values())
        and all(rate > 0 for rate in _idle_rates_per_configured_gpu_count(gpu_model))
    )


@pytest.mark.parametrize("gpu_model", sorted(PAID_MEDIAN_30D))
def test_idle_rate_is_at_most_the_paid_median(gpu_model: str) -> None:
    """Fails when a configured idle rate (in either count bucket) is above the 30-day paid median.
    On main before DAH-3623 it fails for two models: H100 HBM3 (1.494 > 1.39) and RTX 5090 (0.65 > 0.40);
    the other thirteen fixture models are on or under their weighted median (A100 PCIe 0.36 < 0.45 and
    A100 SXM 0.6923 < 0.70 were above the per-rental median, not this one)."""
    cap = _cap_usd_per_gpu_hour(gpu_model)

    for rate in _idle_rates_per_configured_gpu_count(gpu_model):
        assert rate <= cap + 1e-9, (
            f"{gpu_model}: idle rate {rate} USD/GPU-h is above the cap {cap:.4f} "
            f"= {CAP_SHARE_OF_PAID_MEDIAN} x paid median {PAID_MEDIAN_30D[gpu_model].median_usd_per_gpu_hour} "
            f"(as of {PAID_MEDIAN_AS_OF})"
        )


@pytest.mark.parametrize("gpu_model", PINNED_AT_CAP)
def test_dah_3623_pins_sit_exactly_on_the_paid_median_rounded_down_to_the_cent(gpu_model: str) -> None:
    """Fails when a pin is mistyped (1.3 for 1.30 is fine, 1.03 is not) or drifts from the fixture it
    was derived from: the pinned rate must equal the paid median, rounded down to the cent, in both
    buckets. With main's config.py ten of the twelve fail this case (RTX 3090 0.16 and RTX 4090 0.30
    already are their weighted medians); the eight under their median fail this case and only this case
    (under the median is legal, off the pin is not): A100 PCIe 0.36 for 0.45, A100 SXM 0.6923 for 0.70,
    RTX 6000 Ada 0.69 for 0.75, H200 2.85 for 3.65, L40S 0.35 for 0.38, B200 4.25 for 5.60,
    RTX A6000 0.32 for 0.42, H100 PCIe 1.1988 for 1.30."""
    expected = math.floor(_cap_usd_per_gpu_hour(gpu_model) * 100 + 1e-9) / 100
    rates = _idle_rates_per_configured_gpu_count(gpu_model)

    assert rates == [expected] * len(rates), (gpu_model, rates, expected)


def test_every_fixture_model_is_pinned_or_named_unpinned() -> None:
    """Fails when a fixture row is added without a decision: every model with a median is either in
    PINNED_AT_CAP (rate = median) or in NOT_PINNED_UNDER_THE_MEDIAN (rate < median, reason in the
    module docstring), never both, never neither."""
    pinned = set(PINNED_AT_CAP)
    unpinned = set(NOT_PINNED_UNDER_THE_MEDIAN)

    assert not (pinned & unpinned), sorted(pinned & unpinned)
    assert pinned | unpinned == set(PAID_MEDIAN_30D), sorted((pinned | unpinned) ^ set(PAID_MEDIAN_30D))


def test_fixture_names_only_models_that_can_receive_idle_pay() -> None:
    """Fails when a fixture key is misspelt or names a model with no idle-pay bucket: such a row
    would resolve to rate 0 and the cap check above would pass without checking anything."""
    for gpu_model in PAID_MEDIAN_30D:
        base_model = BASE_GPU_MAP[gpu_model]
        assert any(cap > 0 for cap in MAX_UNRENTED_GPUS_BY_TYPE[base_model].values()), gpu_model
        assert all(rate > 0 for rate in _idle_rates_per_configured_gpu_count(gpu_model)), gpu_model


def test_every_model_that_can_receive_idle_pay_is_in_the_fixture_or_named_outside_it() -> None:
    """The inverse of test_fixture_names_only_models_that_can_receive_idle_pay (Rustam's review, 18 Sep
    2026 08:00Z): a model that takes idle pay and is missing from the fixture is never cap-checked.
    Fails when such a model appears (a new entry in BASE_GPU_MAP with a positive bucket cap), when
    either exclusion names a model that cannot take idle pay, when a model is named in both, or when
    an excluded model gains a fixture row and the exclusion is not trimmed. With both exclusions
    emptied, the four models they name fail here."""
    can_receive = set(_models_that_can_receive_idle_pay())
    without_row = set(IDLE_ELIGIBLE_MODELS_WITHOUT_A_MEDIAN_ROW)
    above = set(ABOVE_THE_MEDIAN_NOT_YET_PINNED)
    outside = without_row | above

    assert not (without_row & above), sorted(without_row & above)
    assert outside <= can_receive, sorted(outside - can_receive)
    assert not (outside & set(PAID_MEDIAN_30D)), sorted(outside & set(PAID_MEDIAN_30D))
    assert can_receive - set(PAID_MEDIAN_30D) == outside, sorted(
        (can_receive - set(PAID_MEDIAN_30D)) ^ outside
    )


@pytest.mark.parametrize("gpu_model", sorted(ABOVE_THE_MEDIAN_NOT_YET_PINNED))
def test_recorded_exception_still_sits_above_its_median(gpu_model: str) -> None:
    """The recorded exception is checked, not waived (Rustam's review, 18 Sep 2026 11:24Z): L40 is paid
    idle at 0.36 against a 0.33 median on 151 rentals and is not one of DAH-3623's twelve. Fails the
    day the rate is pinned (then the row belongs in PAID_MEDIAN_30D and PINNED_AT_CAP) or the median
    read moves above the rate (then the exception is gone and the row leaves this dict)."""
    row = ABOVE_THE_MEDIAN_NOT_YET_PINNED[gpu_model]
    assert row.rentals >= MIN_RENTALS_FOR_A_MEDIAN, gpu_model

    for rate in _idle_rates_per_configured_gpu_count(gpu_model):
        assert rate > row.median_usd_per_gpu_hour + 1e-9, (
            f"{gpu_model}: idle rate {rate} is no longer above its median {row.median_usd_per_gpu_hour}; "
            "pin it or drop the exception"
        )
