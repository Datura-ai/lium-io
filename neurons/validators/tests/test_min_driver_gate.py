"""Tests for the minimum NVIDIA driver gate."""

from incentive.default import get_min_driver_multiplier


def test_compliant_driver_is_unaffected():
    # Default floor is 580.65.06 (r580, ships CUDA 13.0).
    assert get_min_driver_multiplier("580.95.05") == 1.0
    assert get_min_driver_multiplier("595.45.04") == 1.0
    # Exactly at the floor is compliant.
    assert get_min_driver_multiplier("580.65.06") == 1.0


def test_unknown_driver_fails_open():
    # A real GPU always reports a driver string; missing/garbage means "not reported".
    assert get_min_driver_multiplier("") == 1.0
    assert get_min_driver_multiplier("unknown") == 1.0


def test_old_driver_is_fully_gated():
    # 570.211.01 (CUDA 12.8) is below the 580.65.06 floor — no incentive.
    assert get_min_driver_multiplier("570.211.01") == 0.0


def test_version_compared_numerically_not_lexically():
    # "580.40" < "580.65.06" numerically even though it could mislead a string compare,
    # and 580.211 (hypothetical) must beat the floor on the second component.
    assert get_min_driver_multiplier("580.40.00") == 0.0
    assert get_min_driver_multiplier("580.211.01") == 1.0


def test_rented_executor_is_exempt():
    # An active customer must not be penalised because the miner has not upgraded yet.
    assert get_min_driver_multiplier("570.211.01", is_rented=True) == 1.0
    # Unrented stays gated for parity with the unrented behaviour.
    assert get_min_driver_multiplier("570.211.01", is_rented=False) == 0.0
