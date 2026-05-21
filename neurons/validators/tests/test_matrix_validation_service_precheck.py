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


# --- validate_gpu_model_and_process_job short-circuits on empty cipher ------
@pytest.mark.asyncio
async def test_validate_short_circuits_on_empty_cipher(service_with_mock_wrapper):
    """When encrypt_challenge returns "" (precheck rejection), no SSH happens
    and we return a clean ValidationResult(success=False) without sending an
    empty cipher to the executor.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    svc, _wrapper = service_with_mock_wrapper

    # Force encrypt_challenge to return "" — simulates precheck rejection.
    svc.encrypt_challenge = lambda *a, **kw: ""

    ssh_client = SimpleNamespace(run=AsyncMock())
    executor_info = SimpleNamespace(
        root_dir="/root/app",
        python_path="/root/app/.venv/bin/python",
    )
    machine_spec = {
        "gpu": {
            "count": 1,
            "details": [
                {
                    "name": "NVIDIA RTX A4000",
                    "uuid": "GPU-abc",
                    "capacity": 15352,
                },
            ],
        },
    }

    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=ssh_client,
        executor_info=executor_info,
        default_extra={},
        machine_spec=machine_spec,
    )

    assert result.success is False
    assert "precheck rejection" in result.error_message.lower() or "native" in result.error_message.lower()
    # Critically: no SSH round-trip.
    ssh_client.run.assert_not_called()
