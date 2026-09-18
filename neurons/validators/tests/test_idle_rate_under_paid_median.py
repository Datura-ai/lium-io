"""DAH-3623: no idle rate above 0.8 x the median price renters paid for the model, no idle rate above
the model's base price, and every pinned model sits on 0.8 x the lower of the two, rounded down to
the cent.

Owner rule, 17 Sep 2026: "idle pay should never be higher than rental rates"; 21:41Z: "pin means raise
too". Window: 30 days (Rustam, 18 Sep 2026 06:20Z on #1401: "fixed rates from a 7-day window are too
unstable"). The fixture below is the 30-day paid median per GPU model read from prod `rental_history`
(per-rental `price_per_gpu`, all gpu_count tiers, rentals active 19 Aug to 18 Sep 2026, i.e.
`coalesce(rental_end_time, now()) > now() - 30 days and rental_start_time < now()`, read
2026-09-18 06:24Z) for every model that received idle pay in the 24 h before the read and had at least
5 rentals in the window (fewer than 5 rentals is not a median to pin on).

Base price (Rustam, 18 Sep 2026 08:00Z on #1401): lium-core's `DEFAULT_SHARED_CONFIG.machine_prices`,
the table `RENTAL_PRICES_PER_HOUR` spreads before its pins. It is what a provider who lists at the
default earns per rented GPU-hour (the backend's `machine.price` is the same table; a listing may sit
between 0.5 x and 2.5 x of it), so a pin above 0.8 x base pays that provider more idle than rented
once the 20 % margin is counted, and a rate above base pays more idle than rented outright.

Family (Rustam, same review): within one cap bucket (`BASE_GPU_MAP`) a lower-class card is never paid
idle above the bucket's flagship, the member with the highest base price. H100 NVL is pinned to H100
HBM3's rate for that reason alone: it had zero rentals in the 30-day window and zero idle pay in the
14 days before the read, so it has no median of its own.

RTX PRO 6000 Server Edition is the one model held off the mark: its 30-day median is 1.20 (998
rentals), so the mark is 0.96, while its rate is pinned at 1.00 for parity with the Workstation Edition
(test_rental_price_anchor_parity.py; median 1.25, mark 1.00). Whether parity or the pin wins is
Rustam's call on #1401; until then its cap case is a strict xfail so the gap is visible, not hidden.

A rate raised above either cap fails here; a pin that drifts off the mark fails here. When the market
moves, the commit that changes a rate also updates the fixture row (median, rentals,
`PAID_MEDIAN_AS_OF`) and says where the new number came from.
"""

import math
from typing import NamedTuple

import pytest
from incentive.config import (
    BASE_GPU_MAP,
    MAX_UNRENTED_GPUS_BY_TYPE,
    RENTAL_PRICES_PER_HOUR,
    IncentiveConfig,
)
from incentive.utils import get_hourly_rate
from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG

CAP_SHARE_OF_PAID_MEDIAN: float = 0.8
CAP_SHARE_OF_BASE_PRICE: float = 0.8
MIN_RENTALS_FOR_A_MEDIAN: int = 5
PAID_MEDIAN_AS_OF: str = "2026-09-18T06:24Z"


class PaidMedian(NamedTuple):
    """One fixture row: the two quantities are named so a swapped pair (388 USD, 0.30 rentals) cannot
    pass the sample-size check and set a 310 USD cap."""

    median_usd_per_gpu_hour: float
    rentals: int


PAID_MEDIAN_30D: dict[str, PaidMedian] = {
    "NVIDIA A100 80GB PCIe": PaidMedian(median_usd_per_gpu_hour=0.30, rentals=388),
    "NVIDIA H100 80GB HBM3": PaidMedian(median_usd_per_gpu_hour=1.30, rentals=691),
    "NVIDIA GeForce RTX 5090": PaidMedian(median_usd_per_gpu_hour=0.60, rentals=1951),
    "NVIDIA RTX 6000 Ada Generation": PaidMedian(median_usd_per_gpu_hour=0.69, rentals=399),
    "NVIDIA GeForce RTX 3090": PaidMedian(median_usd_per_gpu_hour=0.18, rentals=1231),
    "NVIDIA A100-SXM4-80GB": PaidMedian(median_usd_per_gpu_hour=0.68, rentals=126),
    "NVIDIA H200": PaidMedian(median_usd_per_gpu_hour=3.25, rentals=900),
    "NVIDIA L40S": PaidMedian(median_usd_per_gpu_hour=0.38, rentals=676),
    "NVIDIA GeForce RTX 4090": PaidMedian(median_usd_per_gpu_hour=0.32, rentals=2673),
    "NVIDIA B300 SXM6 AC": PaidMedian(median_usd_per_gpu_hour=8.00, rentals=342),
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": PaidMedian(median_usd_per_gpu_hour=1.25, rentals=103),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": PaidMedian(median_usd_per_gpu_hour=1.20, rentals=998),
    "NVIDIA B200": PaidMedian(median_usd_per_gpu_hour=5.60, rentals=634),
    "NVIDIA RTX A6000": PaidMedian(median_usd_per_gpu_hour=0.42, rentals=369),
    "NVIDIA H100 PCIe": PaidMedian(median_usd_per_gpu_hour=1.50, rentals=333),
}

# Models that can receive idle pay but have no fixture row. The 30-day read (2026-09-18 08:06Z,
# RECOVERY/workers/responder-r315/sql/h100nvl_30d.txt) found: H100 NVL and RTX 4090 D with zero rentals
# and zero idle pay; H200 NVL with 20 rentals (median 3.90, rate 2.90, under both caps) and L40 with
# 169 rentals (median 0.33, rate 0.36, 14 idle-pay ledger rows in 14 days) left for Rustam's call on
# #1401, outside DAH-3623's twelve. A model added here without a reason in this comment is a mistake.
OUTSIDE_THE_FIXTURE: tuple[str, ...] = (
    "NVIDIA H200 NVL",
    "NVIDIA H100 NVL",
    "NVIDIA GeForce RTX 4090 D",
    "NVIDIA L40",
)

# Held at Workstation Edition parity (1.00) while its own 30-day mark is 0.96: see the module docstring.
HELD_AT_PARITY_OVER_THE_MARK: tuple[str, ...] = ("NVIDIA RTX PRO 6000 Blackwell Server Edition",)

# Pinned at the cap by DAH-3623: commit 1 the three models whose lium-core rate was above the rental
# price itself, commit 2 (owner, 17 Sep 2026 16:56Z) the six that sat between 88 % and 102 % of it
# on the 30-day medians (A100 SXM 102 %), commit 3 (owner, 21:41Z: "pin means raise too") the three
# that sat under 80 % of it, commit 5 (Rustam, 18 Sep 08:00Z) the base price as the second cap.
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

# Pinned to the flagship of their cap bucket, not to a median of their own (module docstring, family).
PINNED_TO_THE_FAMILY_FLAGSHIP: dict[str, str] = {
    "NVIDIA H100 NVL": "NVIDIA H100 80GB HBM3",
}

# Owner decisions that run AHEAD of the lium-core wheel pdm.lock pins (0.1.8): B300 6.40 against the
# wheel's 5.10 (DAH-3542) and the Server Edition at Workstation parity 1.00 against the wheel's 0.86
# (DAH-3230). Both equal the base price in packages/lium-core's source table, so they sit on base, not
# above it; the wheel is the wrong reference for them until it carries that table.
RATE_IS_THE_NEXT_BASE_PRICE: tuple[str, ...] = (
    "NVIDIA B300 SXM6 AC",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
)


def _base_price(gpu_model: str) -> float:
    return DEFAULT_SHARED_CONFIG.machine_prices[gpu_model]


def _median_cap_usd_per_gpu_hour(gpu_model: str) -> float:
    row = PAID_MEDIAN_30D[gpu_model]
    assert row.rentals >= MIN_RENTALS_FOR_A_MEDIAN, (
        f"{gpu_model}: a median on {row.rentals} rentals is not a cap; use a wider window or drop the row"
    )
    return CAP_SHARE_OF_PAID_MEDIAN * row.median_usd_per_gpu_hour


def _base_cap_usd_per_gpu_hour(gpu_model: str) -> float:
    return CAP_SHARE_OF_BASE_PRICE * _base_price(gpu_model)


def _floor_cent(value: float) -> float:
    return math.floor(value * 100 + 1e-9) / 100


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


def _cap_rows() -> list:
    """Every fixture model; the parity-held one is a strict xfail (it must fail while its rate 1.00 sits
    over its mark 0.96, and the xfail must go the moment the rate or the fixture moves)."""
    return [
        pytest.param(
            gpu_model,
            marks=pytest.mark.xfail(
                strict=True,
                reason="held at Workstation Edition parity 1.00 over its 30-day mark 0.96 (#1401)",
            ),
        )
        if gpu_model in HELD_AT_PARITY_OVER_THE_MARK
        else gpu_model
        for gpu_model in sorted(PAID_MEDIAN_30D)
    ]


@pytest.mark.parametrize("gpu_model", _cap_rows())
def test_idle_rate_is_at_most_80_percent_of_the_paid_median(gpu_model: str) -> None:
    """Fails when a configured idle rate (in either count bucket) is above 0.8 x the 30-day paid median.
    On main before DAH-3623 it failed for nine models: A100 PCIe (0.36 > 0.24), H100 HBM3 (1.494 > 1.04),
    RTX 5090 (0.65 > 0.48), H200 (2.85 > 2.60), RTX 3090 (0.16 > 0.144), RTX 6000 Ada (0.69 > 0.552),
    A100 SXM (0.6923 > 0.544), RTX 4090 (0.30 > 0.256) and L40S (0.35 > 0.304). H100 PCIe (1.1988
    against a 1.20 cap) passed here on main and fails only the pin and 0.8 x base cases below."""
    cap = _median_cap_usd_per_gpu_hour(gpu_model)

    for rate in _idle_rates_for_1_and_8_gpus(gpu_model):
        assert rate <= cap + 1e-9, (
            f"{gpu_model}: idle rate {rate} USD/GPU-h is above the cap {cap:.4f} "
            f"= {CAP_SHARE_OF_PAID_MEDIAN} x paid median {PAID_MEDIAN_30D[gpu_model].median_usd_per_gpu_hour} "
            f"(as of {PAID_MEDIAN_AS_OF})"
        )


@pytest.mark.parametrize(
    "gpu_model",
    [m for m in _models_that_can_receive_idle_pay() if m not in RATE_IS_THE_NEXT_BASE_PRICE],
)
def test_idle_rate_is_never_above_the_base_price(gpu_model: str) -> None:
    """The owner's rule read literally, for every model that can receive idle pay: a provider who lists
    at the base price must never earn more idle than rented. At #1401 head afa8f5d this failed for
    B200 (4.48 > 4.25), RTX A6000 (0.33 > 0.32) and H100 PCIe (1.20 > 1.1988), the three Rustam named.
    B300 and the Server Edition are left out by name (RATE_IS_THE_NEXT_BASE_PRICE): their rates are
    owner decisions that equal the next base price, not DAH-3623 pins."""
    base = _base_price(gpu_model)

    for rate in _idle_rates_for_1_and_8_gpus(gpu_model):
        assert rate <= base + 1e-9, (
            f"{gpu_model}: idle rate {rate} USD/GPU-h is above the base price {base} "
            "(lium-core DEFAULT_SHARED_CONFIG.machine_prices)"
        )


@pytest.mark.parametrize("gpu_model", RATE_IS_THE_NEXT_BASE_PRICE)
def test_the_wheel_is_still_behind_the_two_owner_overrides(gpu_model: str) -> None:
    """Fails the day the pinned lium-core wheel carries B300 6.40 / Server Edition 1.00: the model then
    belongs back in the base-price check above and this tuple loses its row."""
    assert all(rate > _base_price(gpu_model) for rate in _idle_rates_for_1_and_8_gpus(gpu_model)), (
        f"{gpu_model}: the wheel has caught up; drop it from RATE_IS_THE_NEXT_BASE_PRICE"
    )


@pytest.mark.parametrize("gpu_model", PINNED_AT_CAP)
def test_dah_3623_pins_are_at_most_80_percent_of_the_base_price(gpu_model: str) -> None:
    """Fails when a DAH-3623 pin is above 0.8 x the model's base price, in either count bucket. At
    #1401 head afa8f5d seven of the twelve failed: B200 (4.48 > 3.40), RTX A6000 (0.33 > 0.256),
    H100 PCIe (1.20 > 0.959), H200 (2.60 > 2.28), RTX 3090 (0.14 > 0.128), RTX 4090 (0.25 > 0.24) and
    L40S (0.30 > 0.28)."""
    cap = _base_cap_usd_per_gpu_hour(gpu_model)

    for rate in _idle_rates_for_1_and_8_gpus(gpu_model):
        assert rate <= cap + 1e-9, (
            f"{gpu_model}: pin {rate} USD/GPU-h is above {CAP_SHARE_OF_BASE_PRICE} x base price "
            f"{_base_price(gpu_model)} = {cap:.4f}"
        )


@pytest.mark.parametrize("gpu_model", PINNED_AT_CAP)
def test_dah_3623_pins_sit_exactly_on_the_lower_cap_rounded_down_to_the_cent(gpu_model: str) -> None:
    """Fails when a pin is mistyped (1.4 for 1.04) or drifts from the fixture it was derived from:
    the pinned rate must equal 0.8 x the lower of the paid median and the base price, rounded down to
    the cent, in both buckets. With main's config.py the three models commit 3 raised pass the median
    cap and fail this case and the 0.8 x base case (under the median cap is legal, off the pin is not):
    B200 4.25 for 3.40, RTX A6000 0.32 for 0.25, H100 PCIe 1.1988 for 0.95."""
    expected = _floor_cent(
        min(_median_cap_usd_per_gpu_hour(gpu_model), _base_cap_usd_per_gpu_hour(gpu_model))
    )

    assert _idle_rates_for_1_and_8_gpus(gpu_model) == [expected, expected]


def _buckets_with_idle_pay() -> list[str]:
    return sorted({BASE_GPU_MAP[gpu_model] for gpu_model in _models_that_can_receive_idle_pay()})


@pytest.mark.parametrize("base_model", _buckets_with_idle_pay())
def test_no_card_is_paid_idle_above_the_flagship_of_its_cap_bucket(base_model: str) -> None:
    """Fails when a member of a cap bucket is paid idle above the bucket's flagship, the member with the
    highest base price. At #1401 head afa8f5d the H100 bucket ran backwards: PCIe 1.20 and NVL 1.11
    above HBM3 1.04 (Rustam's comment on config.py:56). In the RTX PRO 6000 bucket the Workstation
    Edition is the flagship (base 1.0 in the pinned wheel, Server 0.86) and the Server Edition sits at
    its rate through the parity override; the source table in packages/lium-core has both at 1.0."""
    members = [
        gpu_model
        for gpu_model in _models_that_can_receive_idle_pay()
        if BASE_GPU_MAP[gpu_model] == base_model
    ]
    flagship = max(members, key=_base_price)
    flagship_rates = _idle_rates_for_1_and_8_gpus(flagship)

    for gpu_model in members:
        for flagship_rate, rate in zip(flagship_rates, _idle_rates_for_1_and_8_gpus(gpu_model)):
            assert rate <= flagship_rate + 1e-9, (
                f"{gpu_model}: idle rate {rate} USD/GPU-h is above the {base_model} flagship "
                f"{flagship} at {flagship_rate}"
            )


@pytest.mark.parametrize("gpu_model,flagship", sorted(PINNED_TO_THE_FAMILY_FLAGSHIP.items()))
def test_family_pins_equal_the_flagship_rate(gpu_model: str, flagship: str) -> None:
    """Fails when a family pin drifts from the flagship it follows (H100 NVL must move with H100 HBM3)
    or when the pinned card is given a median row of its own, at which point it belongs in
    PINNED_AT_CAP instead. With main's config.py H100 NVL is 1.11 for HBM3's 1.04."""
    assert gpu_model not in PAID_MEDIAN_30D, f"{gpu_model} has a median now; pin it in PINNED_AT_CAP"
    assert BASE_GPU_MAP[gpu_model] == BASE_GPU_MAP[flagship]
    assert _base_price(gpu_model) < _base_price(flagship), f"{gpu_model} is the flagship, not {flagship}"
    assert RENTAL_PRICES_PER_HOUR[gpu_model] == RENTAL_PRICES_PER_HOUR[flagship]
    assert _idle_rates_for_1_and_8_gpus(gpu_model) == _idle_rates_for_1_and_8_gpus(flagship)


def test_fixture_names_only_models_that_can_receive_idle_pay() -> None:
    """Fails when a fixture key is misspelt or names a model with no idle-pay bucket: such a row
    would resolve to rate 0 and the cap check above would pass without checking anything."""
    for gpu_model in PAID_MEDIAN_30D:
        base_model = BASE_GPU_MAP[gpu_model]
        assert any(cap > 0 for cap in MAX_UNRENTED_GPUS_BY_TYPE[base_model].values()), gpu_model
        assert all(rate > 0 for rate in _idle_rates_for_1_and_8_gpus(gpu_model)), gpu_model


def test_every_model_that_can_receive_idle_pay_is_in_the_fixture_or_named_outside_it() -> None:
    """The inverse of the test above (Rustam's comment on test :138): a model that takes idle pay and is
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
