"""Tests for GPU count multiplier resolution logic."""
import pytest

from incentive.utils import get_gpu_count_multiplier


@pytest.mark.parametrize(
    "gpu_model,gpu_count,config,expected",
    [
        # wildcard GPU: listed count → match, unlisted → "*" fallback
        ("H100", 1, {"*": {"*": 0, "1": 1, "8": 1}}, 1),
        ("H100", 4, {"*": {"*": 0, "1": 1, "8": 1}}, 0),
        # specific GPU takes priority over wildcard
        ("B200", 8, {"*": {"*": 0, "1": 1}, "B200": {"*": 0, "8": 1}}, 1),
        ("B200", 1, {"*": {"*": 0, "1": 1}, "B200": {"*": 0, "8": 1}}, 0),
        # fractional multipliers + "*" count fallback
        ("H200", 4, {"H200": {"1": 1, "4": 0.5, "*": 0.3}}, 0.5),
        ("H200", 2, {"H200": {"1": 1, "4": 0.5, "*": 0.3}}, 0.3),
        # empty GPU entry — no fallback to wildcard
        ("H100", 8, {"H100": {}, "*": {"*": 1}}, 0.0),
        # no matching GPU, no wildcard → 0
        ("H100", 8, {"B200": {"8": 1}}, 0.0),
        # empty config → 0
        ("H100", 8, {}, 0.0),
    ],
    ids=[
        "wildcard-listed", "wildcard-fallback",
        "specific-overrides-wildcard", "specific-star-fallback",
        "fractional", "fractional-star",
        "empty-entry-no-fallback",
        "no-match-no-wildcard",
        "empty-config",
    ],
)
def test_gpu_count_multiplier(
    gpu_model: str, gpu_count: int, config: dict, expected: float,
):
    assert get_gpu_count_multiplier(gpu_model, gpu_count, config) == expected


@pytest.mark.parametrize(
    "gpu_count,expected",
    [(1, 1), (8, 1), (4, 0)],
    ids=["allowed-1", "allowed-8", "blocked-4"],
)
def test_default_config_when_none_passed(gpu_count: int, expected: float):
    """Falls back to GPU_COUNT_MULTIPLIERS when no config provided."""
    assert get_gpu_count_multiplier("H100", gpu_count) == expected
