"""DAH-3623: the idle rate is 0.8 x the base listing price, rounded down to the cent.

The base prices are the backend's MACHINE_PRICES from lium-platform#558 (DAH-3648, the 30-day paid
median). The idle rate is set in GPU_COUNT_CUSTOM_PRICES; RENTAL_PRICES_PER_HOUR does not change.
"""

import math

import pytest
from incentive.config import IncentiveConfig
from incentive.utils import get_hourly_rate

IDLE_SHARE_OF_BASE_PRICE: float = 0.8
GPU_COUNTS: tuple[int, ...] = (1, 8)

# lium-platform#558 MACHINE_PRICES, USD per GPU-hour
BASE_PRICE_USD_PER_GPU_HOUR: dict[str, float] = {
    "NVIDIA B300 SXM6 AC": 8.0,
    "NVIDIA B200": 5.6,
    "NVIDIA H200": 3.65,
    "NVIDIA H100 80GB HBM3": 1.39,
    "NVIDIA H100 PCIe": 1.3,
    "NVIDIA GeForce RTX 5090": 0.4,
    "NVIDIA GeForce RTX 4090": 0.30,
    "NVIDIA GeForce RTX 3090": 0.16,
    "NVIDIA RTX 6000 Ada Generation": 0.75,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 1.19,
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 1.19,
    "NVIDIA L40S": 0.38,
    "NVIDIA L40": 0.33,
    "NVIDIA A100 80GB PCIe": 0.45,
    "NVIDIA A100-SXM4-80GB": 0.7,
    "NVIDIA RTX A6000": 0.42,
}


def _idle_rate(gpu_model: str, gpu_count: int) -> float:
    config = IncentiveConfig()
    return get_hourly_rate(
        gpu_model, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour
    )


@pytest.mark.parametrize("gpu_count", GPU_COUNTS)
@pytest.mark.parametrize("gpu_model", sorted(BASE_PRICE_USD_PER_GPU_HOUR))
def test_idle_rate_is_0_8_of_the_base_price(gpu_model: str, gpu_count: int) -> None:
    base_price = BASE_PRICE_USD_PER_GPU_HOUR[gpu_model]
    expected = math.floor(IDLE_SHARE_OF_BASE_PRICE * base_price * 100 + 1e-9) / 100

    assert _idle_rate(gpu_model, gpu_count) == pytest.approx(expected)


@pytest.mark.parametrize("gpu_count", GPU_COUNTS)
@pytest.mark.parametrize("gpu_model", sorted(BASE_PRICE_USD_PER_GPU_HOUR))
def test_idle_rate_is_below_the_base_price(gpu_model: str, gpu_count: int) -> None:
    assert 0 < _idle_rate(gpu_model, gpu_count) < BASE_PRICE_USD_PER_GPU_HOUR[gpu_model]
