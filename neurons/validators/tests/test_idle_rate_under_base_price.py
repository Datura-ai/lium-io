"""DAH-3623: the base price is the one of lium-platform#558 (DAH-3648); the idle rate is a share of it
set by how much the model is rented: scarce cards keep more idle pay, surplus cards less.

The validator's base table comes from the lium-core wheel pinned in pdm.lock, which still carries the
old prices, so RENTAL_PRICES_PER_HOUR overrides the models #558 moved.
"""

import pytest
from incentive.config import (
    GPU_COUNT_CUSTOM_PRICES,
    RENTAL_PRICES_PER_HOUR,
    DefaultPrice,
    IncentiveConfig,
)
from incentive.utils import get_hourly_rate


def _idle_paying_counts(gpu_model: str) -> tuple[int, ...]:
    """The GPU counts GPU_COUNT_CUSTOM_PRICES pays idle for on this model (its own row, else "*")."""
    row = GPU_COUNT_CUSTOM_PRICES.get(gpu_model, GPU_COUNT_CUSTOM_PRICES["*"])
    return tuple(
        sorted(int(count) for count, price in row.items() if count != "*" and isinstance(price, DefaultPrice))
    )


IDLE_PAYING_MODEL_COUNTS: tuple[tuple[str, int], ...] = tuple(
    (gpu_model, gpu_count)
    for gpu_model in sorted(RENTAL_PRICES_PER_HOUR)
    for gpu_count in _idle_paying_counts(gpu_model)
)

# lium-platform#558 MACHINE_PRICES that differ from the lium-core wheel, USD per GPU-hour
BASE_PRICE_FROM_PR_558: dict[str, float] = {
    "NVIDIA B300 SXM6 AC": 8.0,
    "NVIDIA B200": 5.6,
    "NVIDIA H200": 3.65,
    "NVIDIA H100 80GB HBM3": 1.39,
    "NVIDIA H100 PCIe": 1.3,
    "NVIDIA GeForce RTX 5090": 0.4,
    "NVIDIA RTX 6000 Ada Generation": 0.75,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 1.19,
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 1.19,
    "NVIDIA L40S": 0.38,
    "NVIDIA L40": 0.33,
    "NVIDIA A100 80GB PCIe": 0.45,
    "NVIDIA A100-SXM4-80GB": 0.7,
    "NVIDIA RTX A6000": 0.42,
}


@pytest.mark.parametrize("gpu_model", sorted(BASE_PRICE_FROM_PR_558))
def test_base_price_is_the_one_of_pr_558(gpu_model: str) -> None:
    assert RENTAL_PRICES_PER_HOUR[gpu_model] == BASE_PRICE_FROM_PR_558[gpu_model]


# gpu_price_stat.lium_rental_rate_30d on prod, 22 Sep 2026: rented / offered GPU-hours
RENTAL_RATE_30D: dict[str, float] = {
    "NVIDIA A100 80GB PCIe": 0.99,
    "NVIDIA B300 SXM6 AC": 0.90,
    "NVIDIA B300 SXM6 PC": 0.90,  # alias of the AC card
    "NVIDIA L40S": 0.87,
    "NVIDIA L40": 0.87,
    "NVIDIA A100-SXM4-80GB": 0.86,
    "NVIDIA RTX A6000": 0.85,
    "NVIDIA B200": 0.79,
    "NVIDIA H200": 0.78,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 0.77,
    # parity with the Server Edition (DAH-3230); its own 0.48 is on 10 GPUs
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 0.77,
    "NVIDIA GeForce RTX 4090": 0.72,
    "NVIDIA GeForce RTX 5090": 0.64,
    "NVIDIA RTX 6000 Ada Generation": 0.56,
    "NVIDIA H100 80GB HBM3": 0.54,
    "NVIDIA H100 PCIe": 0.53,
    "NVIDIA GeForce RTX 3090": 0.45,
}


def _idle_share_of_base_price(gpu_model: str) -> float:
    rental_rate = RENTAL_RATE_30D.get(gpu_model)
    if rental_rate is None:
        return 0.8
    if rental_rate >= 0.8:
        return 0.9
    if rental_rate >= 0.6:
        return 0.8
    return 0.7


def test_every_priced_model_pays_idle_for_at_least_one_gpu_count() -> None:
    assert {gpu_model for gpu_model, _ in IDLE_PAYING_MODEL_COUNTS} == set(RENTAL_PRICES_PER_HOUR)


@pytest.mark.parametrize(("gpu_model", "gpu_count"), IDLE_PAYING_MODEL_COUNTS)
def test_idle_rate_share_follows_the_rental_rate(gpu_model: str, gpu_count: int) -> None:
    config = IncentiveConfig()

    rate = get_hourly_rate(
        gpu_model, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour
    )

    assert rate == pytest.approx(_idle_share_of_base_price(gpu_model) * RENTAL_PRICES_PER_HOUR[gpu_model])
