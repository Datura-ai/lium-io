"""DAH-3623: no idle rate above the median price renters paid for the model, and every pinned model
sits on that median, rounded down to the cent.

Owner rule, 17 Sep 2026: "idle pay should never be higher than rental rates"; 21:41Z: "pin means raise
too"; 18 Sep 2026 09:14Z: "pin the idle to median now" (the pin was 0.8 x the median until then).
Window: 30 days (Rustam, 18 Sep 2026 06:20Z on #1401: "fixed rates from a 7-day window are too
unstable"). The fixture below is the 30-day paid median per GPU model read from prod `rental_history`
(per-rental `price_per_gpu`, all gpu_count tiers, rentals active 19 Aug to 18 Sep 2026, i.e.
`coalesce(rental_end_time, now()) > now() - 30 days and rental_start_time < now()`, read
2026-09-18 09:16Z; the same 15 medians as the 06:24Z read) for every model that received idle pay in
the 24 h before the read and had at least 5 rentals in the window (fewer than 5 rentals is not a
median to pin on).

Three fixture models are not pinned and sit UNDER their median by an earlier decision: B300 (6.40
against an 8.00 median, DAH-3542) and the two RTX PRO 6000 editions (1.00 against 1.25 Workstation /
1.20 Server, held at parity by DAH-3230, test_rental_price_anchor_parity.py). They pass the cap and are
named in NOT_PINNED_UNDER_THE_MEDIAN so a new fixture row has to be classified one way or the other.

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
PAID_MEDIAN_AS_OF: str = "2026-09-18T09:16Z"


class PaidMedian(NamedTuple):
    """One fixture row: the two quantities are named so a swapped pair (399 USD, 0.30 rentals) cannot
    pass the sample-size check and set a 399 USD cap."""

    median_usd_per_gpu_hour: float
    rentals: int


# gpu_model -> the median USD per GPU-hour renters paid over 30 days and the rentals active in the window
PAID_MEDIAN_30D: dict[str, PaidMedian] = {
    "NVIDIA A100 80GB PCIe": PaidMedian(median_usd_per_gpu_hour=0.30, rentals=399),
    "NVIDIA H100 80GB HBM3": PaidMedian(median_usd_per_gpu_hour=1.30, rentals=699),
    "NVIDIA GeForce RTX 5090": PaidMedian(median_usd_per_gpu_hour=0.60, rentals=1950),
    "NVIDIA RTX 6000 Ada Generation": PaidMedian(median_usd_per_gpu_hour=0.69, rentals=401),
    "NVIDIA GeForce RTX 3090": PaidMedian(median_usd_per_gpu_hour=0.18, rentals=1235),
    "NVIDIA A100-SXM4-80GB": PaidMedian(median_usd_per_gpu_hour=0.68, rentals=127),
    "NVIDIA H200": PaidMedian(median_usd_per_gpu_hour=3.25, rentals=912),
    "NVIDIA L40S": PaidMedian(median_usd_per_gpu_hour=0.38, rentals=676),
    "NVIDIA GeForce RTX 4090": PaidMedian(median_usd_per_gpu_hour=0.32, rentals=2679),
    "NVIDIA B300 SXM6 AC": PaidMedian(median_usd_per_gpu_hour=8.00, rentals=345),
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": PaidMedian(median_usd_per_gpu_hour=1.25, rentals=103),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": PaidMedian(median_usd_per_gpu_hour=1.20, rentals=1002),
    "NVIDIA B200": PaidMedian(median_usd_per_gpu_hour=5.60, rentals=634),
    "NVIDIA RTX A6000": PaidMedian(median_usd_per_gpu_hour=0.42, rentals=369),
    "NVIDIA H100 PCIe": PaidMedian(median_usd_per_gpu_hour=1.50, rentals=333),
}

# Models that can receive idle pay but have no fixture row. The 30-day read (2026-09-18 08:06Z, prod
# replica, read-only; query and output in the loop's private folder, on request) found: H100 NVL and RTX 4090 D with zero rentals
# and zero idle pay; H200 NVL with 20 rentals (median 3.90, rate 2.90, under the median) and L40 with
# 169 rentals (median 0.33, rate 0.36, 14 idle-pay ledger rows in 14 days) left for Rustam's call on
# #1401, outside DAH-3623's twelve. A model added here without a reason in this comment is a mistake.
OUTSIDE_THE_FIXTURE: tuple[str, ...] = (
    "NVIDIA H200 NVL",
    "NVIDIA H100 NVL",
    "NVIDIA GeForce RTX 4090 D",
    "NVIDIA L40",
)

# Fixture models that are NOT pinned to their median and sit under it by an earlier decision (module
# docstring): B300 6.40 (DAH-3542) and the two RTX PRO 6000 editions at Workstation parity 1.00 (DAH-3230).
NOT_PINNED_UNDER_THE_MEDIAN: tuple[str, ...] = (
    "NVIDIA B300 SXM6 AC",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
)

# Pinned at the median by DAH-3623: commit 1 the three models whose lium-core rate was above the rental
# price itself, commit 2 (owner, 17 Sep 2026 16:56Z) the six that sat between 88 % and 102 % of it
# on the 30-day medians (A100 SXM 102 %), commit 3 (owner, 21:41Z: "pin means raise too") the three
# that sat under 80 % of it; commit 6 (owner, 18 Sep 09:14Z) moves all twelve from 0.8 x to 1.0 x.
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


def _idle_rates_for_1_and_8_gpus(gpu_model: str) -> list[float]:
    config = IncentiveConfig()
    return [
        get_hourly_rate(
            gpu_model, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour
        )
        for gpu_count in (1, 8)
    ]


def _models_that_can_receive_idle_pay() -> list[str]:
    """Every model in BASE_GPU_MAP whose bucket has a positive cap and whose rate resolves above 0."""
    return sorted(
        gpu_model
        for gpu_model, base_model in BASE_GPU_MAP.items()
        if any(cap > 0 for cap in MAX_UNRENTED_GPUS_BY_TYPE[base_model].values())
        and all(rate > 0 for rate in _idle_rates_for_1_and_8_gpus(gpu_model))
    )


@pytest.mark.parametrize("gpu_model", sorted(PAID_MEDIAN_30D))
def test_idle_rate_is_at_most_the_paid_median(gpu_model: str) -> None:
    """Fails when a configured idle rate (in either count bucket) is above the 30-day paid median.
    On main before DAH-3623 it failed for four models: A100 PCIe (0.36 > 0.30), H100 HBM3 (1.494 > 1.30),
    RTX 5090 (0.65 > 0.60) and A100 SXM (0.6923 > 0.68); the other eleven fixture models were on or
    under their median."""
    cap = _cap_usd_per_gpu_hour(gpu_model)

    for rate in _idle_rates_for_1_and_8_gpus(gpu_model):
        assert rate <= cap + 1e-9, (
            f"{gpu_model}: idle rate {rate} USD/GPU-h is above the cap {cap:.4f} "
            f"= {CAP_SHARE_OF_PAID_MEDIAN} x paid median {PAID_MEDIAN_30D[gpu_model].median_usd_per_gpu_hour} "
            f"(as of {PAID_MEDIAN_AS_OF})"
        )


@pytest.mark.parametrize("gpu_model", PINNED_AT_CAP)
def test_dah_3623_pins_sit_exactly_on_the_paid_median_rounded_down_to_the_cent(gpu_model: str) -> None:
    """Fails when a pin is mistyped (1.3 for 1.30 is fine, 1.03 is not) or drifts from the fixture it
    was derived from: the pinned rate must equal the paid median, rounded down to the cent, in both
    buckets. With main's config.py eleven of the twelve fail this case (RTX 6000 Ada's 0.69 already is
    its median); the seven under their median fail this case and only this case (under the median is
    legal, off the pin is not): H200 2.85 for 3.25, RTX 3090 0.16 for 0.18, RTX 4090 0.30 for 0.32,
    L40S 0.35 for 0.38, B200 4.25 for 5.60, RTX A6000 0.32 for 0.42, H100 PCIe 1.1988 for 1.50."""
    expected = math.floor(_cap_usd_per_gpu_hour(gpu_model) * 100 + 1e-9) / 100

    assert _idle_rates_for_1_and_8_gpus(gpu_model) == [expected, expected]


@pytest.mark.parametrize("gpu_model", NOT_PINNED_UNDER_THE_MEDIAN)
def test_unpinned_fixture_models_sit_under_their_median_by_an_earlier_decision(gpu_model: str) -> None:
    """Fails the day one of the three unpinned fixture models reaches or passes its median (the market
    fell, or someone raised the rate): at that point it is either pinned (moved to PINNED_AT_CAP with
    its rate on the median) or the earlier decision is re-made, and this tuple loses the row."""
    assert gpu_model not in PINNED_AT_CAP, f"{gpu_model} is pinned; drop it from this tuple"
    cap = _cap_usd_per_gpu_hour(gpu_model)

    for rate in _idle_rates_for_1_and_8_gpus(gpu_model):
        assert rate < cap - 1e-9, (
            f"{gpu_model}: idle rate {rate} USD/GPU-h is at or above its median {cap:.4f}; pin it or re-decide"
        )


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
        assert all(rate > 0 for rate in _idle_rates_for_1_and_8_gpus(gpu_model)), gpu_model


def test_every_model_that_can_receive_idle_pay_is_in_the_fixture_or_named_outside_it() -> None:
    """The inverse of test_fixture_names_only_models_that_can_receive_idle_pay (Rustam's review, 18 Sep
    2026 08:00Z): a model that takes idle pay and is
    missing from the fixture is never cap-checked. Fails when such a model appears (a new entry in
    BASE_GPU_MAP with a positive bucket cap), when the exclusion tuple names a model that cannot take
    idle pay, or when an excluded model gains a fixture row and the tuple is not trimmed. With the
    tuple emptied, the four models it names fail here."""
    can_receive = set(_models_that_can_receive_idle_pay())
    outside = set(OUTSIDE_THE_FIXTURE)

    assert outside <= can_receive, sorted(outside - can_receive)
    assert not (outside & set(PAID_MEDIAN_30D)), sorted(outside & set(PAID_MEDIAN_30D))
    assert can_receive - set(PAID_MEDIAN_30D) == outside, sorted(
        (can_receive - set(PAID_MEDIAN_30D)) ^ outside
    )
