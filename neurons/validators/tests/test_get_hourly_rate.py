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
        # exact match: specific GPU + specific count
        ("H100", 1, {"H100": {"*": 0, "1": 2.50, "8": 3.00}}, 2.50),
        ("H100", 8, {"H100": {"*": 0, "1": 2.50, "8": 3.00}}, 3.00),
        # exact match: count fallback to "*"
        ("H100", 4, {"H100": {"*": 0, "1": 2.50, "8": 3.00}}, 0),
        # DEFAULT_PRICE sentinel resolves to default_prices
        ("H100", 1, {"H100": {"*": 0, "1": D, "8": D}}, 1.26),
        ("H200", 8, {"H200": {"*": 0, "1": D, "8": D}}, 1.90),
        # DEFAULT_PRICE for unknown GPU falls back to 0
        ("UNKNOWN", 1, {"UNKNOWN": {"1": D}}, 0.0),
        # wildcard GPU: listed count
        ("H100", 1, {"*": {"*": 0, "1": D, "8": D}}, 1.26),
        ("H100", 4, {"*": {"*": 0, "1": D, "8": D}}, 0),
        # specific GPU takes priority over wildcard
        ("H100", 8, {"*": {"*": 0, "1": D}, "H100": {"*": 0, "8": 5.00}}, 5.00),
        ("H100", 1, {"*": {"*": 0, "1": D}, "H100": {"*": 0, "8": 5.00}}, 0),
        # no matching GPU, no wildcard
        ("H100", 8, {"H200": {"8": 1.90}}, 0.0),
        # empty config
        ("H100", 8, {}, 0.0),
        # wildcard GPU with all DEFAULT_PRICE (equivalent to old multiplier=1)
        ("RTX 4090", 1, {"*": {"*": D}}, 0.14),
    ],
    ids=[
        "exact-gpu-count-1",
        "exact-gpu-count-8",
        "exact-gpu-count-fallback-star",
        "default-sentinel-h100",
        "default-sentinel-h200",
        "default-sentinel-unknown-gpu",
        "wildcard-gpu-listed-count",
        "wildcard-gpu-unlisted-count",
        "specific-overrides-wildcard",
        "specific-star-fallback",
        "no-match-no-wildcard",
        "empty-config",
        "wildcard-all-default",
    ],
)
def test_get_hourly_rate(
    gpu_model: str, gpu_count: int, custom_prices: dict, expected: float,
):
    assert get_hourly_rate(gpu_model, gpu_count, custom_prices, DEFAULT_PRICES) == expected
