"""Tests for the minimum NVIDIA driver gate."""

from datetime import UTC, timedelta

from core.config import settings
from incentive.default import get_min_driver_multiplier

BEFORE_CUTOFF = settings.MIN_DRIVER_CUTOFF - timedelta(days=1)
AFTER_CUTOFF = settings.MIN_DRIVER_CUTOFF + timedelta(days=1)


def test_compliant_driver_is_unaffected():
    # Default floor is 580.65.06 (r580, ships CUDA 13.0).
    assert get_min_driver_multiplier("580.95.05", reference_time=AFTER_CUTOFF) == 1.0
    assert get_min_driver_multiplier("595.45.04", reference_time=AFTER_CUTOFF) == 1.0
    # Exactly at the floor is compliant.
    assert get_min_driver_multiplier("580.65.06", reference_time=AFTER_CUTOFF) == 1.0


def test_unknown_driver_fails_open():
    # A real GPU always reports a driver string; missing/garbage means "not reported".
    assert get_min_driver_multiplier("", reference_time=AFTER_CUTOFF) == 1.0
    assert get_min_driver_multiplier("unknown", reference_time=AFTER_CUTOFF) == 1.0


def test_old_driver_in_grace_period_is_unaffected():
    # Before the cutoff providers still have time to upgrade — no penalty yet.
    assert get_min_driver_multiplier("570.211.01", reference_time=BEFORE_CUTOFF) == 1.0


def test_old_driver_is_fully_gated_after_cutoff():
    # On/after the cutoff a non-compliant unrented executor earns nothing.
    assert get_min_driver_multiplier("570.211.01", reference_time=AFTER_CUTOFF) == 0.0


def test_version_compared_numerically_not_lexically():
    # "580.40" < "580.65.06" numerically even though it could mislead a string compare,
    # and 580.211 (hypothetical) must beat the floor on the second component.
    assert get_min_driver_multiplier("580.40.00", reference_time=AFTER_CUTOFF) == 0.0
    assert get_min_driver_multiplier("580.211.01", reference_time=AFTER_CUTOFF) == 1.0


def test_rented_executor_is_exempt():
    # An active customer must not be penalised, even after the cutoff.
    assert get_min_driver_multiplier("570.211.01", is_rented=True, reference_time=AFTER_CUTOFF) == 1.0
    # Unrented stays gated after the cutoff.
    assert get_min_driver_multiplier("570.211.01", is_rented=False, reference_time=AFTER_CUTOFF) == 0.0


def test_tz_aware_reference_time_is_normalised():
    aware_after = AFTER_CUTOFF.replace(tzinfo=UTC)
    assert get_min_driver_multiplier("570.211.01", reference_time=aware_after) == 0.0
