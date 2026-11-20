"""Tests for get_hourly_rate resolution logic."""
import pytest

from incentive.config import DEFAULT_PRICE
from incentive.utils import get_hourly_rate

D = DEFAULT_PRICE

DEFAULT_PRICES = {
    "H100": 1.26,
    "H200": 1.90,
    "RTX 4090": 0.14,
}


@pytest.mark.parametrize(
    "gpu_model,gpu_count,custom_prices,expected",
    [
        ("H100", 1, {"H100": {"*": 0, "1": 2.50, "8": 3.00}}, 2.50),
        ("H100", 1, {"H100": {"*": 0, "1": D, "8": D}}, 1.26),
        ("H100", 8, {"*": {"*": 0, "1": D}, "H100": {"*": 0, "8": 5.00}}, 5.00),
        ("H100", 8, {"H200": {"8": 1.90}}, 0.0),
    ],
    ids=[
        "exact-gpu-count-1",
        "default-sentinel-h100",
        "specific-overrides-wildcard",
        "no-match-no-wildcard",
    ],
)
def test_get_hourly_rate(
    gpu_model: str, gpu_count: int, custom_prices: dict, expected: float,
):
    assert get_hourly_rate(gpu_model, gpu_count, custom_prices, DEFAULT_PRICES) == expected
