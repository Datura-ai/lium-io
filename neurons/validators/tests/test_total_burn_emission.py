"""Tests for the shared-config-sourced burn emission accessor (DAH-2274).

SharedConfigClient already falls back to the lium-core default when the backend is
unreachable, so get_total_burn_emission() only guards against a structurally valid
but out-of-range served value (e.g. an operator typo) reaching on-chain weights:
it returns the served value when in [0, 1], otherwise the lium-core default.
"""

import pytest
from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG

from core.config import get_total_burn_emission, shared_client

FALLBACK = DEFAULT_SHARED_CONFIG.total_burn_emission


def _config_with_burn(value):
    # model_copy bypasses validation, letting us inject out-of-range values.
    return DEFAULT_SHARED_CONFIG.model_copy(update={"total_burn_emission": value})


@pytest.mark.parametrize("value", [0.87, 0.91, 0.0, 1.0, 0.13])
def test_valid_value_is_returned(monkeypatch, value):
    monkeypatch.setattr(shared_client, "_config", _config_with_burn(value))
    assert get_total_burn_emission() == pytest.approx(value)


@pytest.mark.parametrize("value", [-0.01, 1.01, 9.1, -5.0])
def test_out_of_range_value_falls_back_to_default(monkeypatch, value):
    monkeypatch.setattr(shared_client, "_config", _config_with_burn(value))
    assert get_total_burn_emission() == pytest.approx(FALLBACK)


def test_default_offline_config_is_in_range(monkeypatch):
    # The conftest patches _fetch -> None, so the live config is the offline default.
    monkeypatch.setattr(shared_client, "_config", DEFAULT_SHARED_CONFIG)
    assert get_total_burn_emission() == pytest.approx(FALLBACK)
    assert 0.0 <= get_total_burn_emission() <= 1.0
