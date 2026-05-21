"""Integration tests for ValidationService.encrypt_challenge() pre-check wiring."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import services.matrix_validation_service as mvs


@pytest.fixture
def service_with_mock_wrapper(monkeypatch):
    """ValidationService whose DMCompVerifyWrapper is fully mocked."""
    mock_wrapper = MagicMock(name="DMCompVerifyWrapper")
    mock_wrapper.DMCompVerify_new.return_value = "fake_verifier_ptr"
    mock_wrapper.getCipherText.return_value = "deadbeef"
    monkeypatch.setattr(mvs, "DMCompVerifyWrapper", lambda *_a, **_kw: mock_wrapper)
    return mvs.ValidationService(), mock_wrapper


def _machine_info(**override) -> str:
    info = {
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "gpu_capacity_mb": 81920,
        "gpu_count": 1,
        "uuids": "GPU-abc",
    }
    info.update(override)
    return json.dumps(info, sort_keys=True)


# --- Bad spec short-circuits before any native call -------------------------
def test_bad_spec_short_circuits(service_with_mock_wrapper):
    svc, wrapper = service_with_mock_wrapper
    bad_mi = _machine_info(gpu_capacity_mb=24064)  # out of range for H100
    cipher = svc.encrypt_challenge(1000, 2000, 42, bad_mi, "uuid-xyz")
    assert cipher == ""
    wrapper.setDimension.assert_not_called()
    wrapper.generateChallenge.assert_not_called()


def test_unknown_model_short_circuits(service_with_mock_wrapper):
    svc, wrapper = service_with_mock_wrapper
    bad_mi = _machine_info(gpu_model="Totally Unknown GPU", gpu_capacity_mb=24064)
    cipher = svc.encrypt_challenge(1000, 2000, 42, bad_mi, "uuid-xyz")
    assert cipher == ""
    wrapper.setDimension.assert_not_called()


def test_missing_field_short_circuits(service_with_mock_wrapper):
    svc, wrapper = service_with_mock_wrapper
    bad_mi = _machine_info(gpu_model="", gpu_capacity_mb=0)
    cipher = svc.encrypt_challenge(1000, 2000, 42, bad_mi, "uuid-xyz")
    assert cipher == ""
    wrapper.setDimension.assert_not_called()


# --- Happy path proceeds ----------------------------------------------------
def test_happy_path_proceeds(service_with_mock_wrapper):
    svc, wrapper = service_with_mock_wrapper
    cipher = svc.encrypt_challenge(1000, 2000, 42, _machine_info(), "uuid-xyz")
    assert cipher == "deadbeef"
    wrapper.setDimension.assert_called_once_with("fake_verifier_ptr", 1000, 2000)
    wrapper.generateChallenge.assert_called_once()


# --- Malformed JSON returns safely (treated as missing_field) ---------------
def test_malformed_json_safe_return(service_with_mock_wrapper):
    svc, wrapper = service_with_mock_wrapper
    cipher = svc.encrypt_challenge(1000, 2000, 42, "{not valid json", "uuid-xyz")
    assert cipher == ""
    wrapper.setDimension.assert_not_called()
