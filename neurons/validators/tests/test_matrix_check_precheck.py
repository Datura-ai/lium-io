"""Integration tests for MatrixValidationCheck._generate_cipher_text pre-check wiring."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from preflight.checks.matrix_check import CipherGenError, MatrixValidationCheck


@pytest.fixture
def mock_wrapper():
    w = MagicMock(name="DMCompVerifyWrapper")
    w.create_verifier.return_value = "fake_verifier_ptr"
    w.get_cipher_text.return_value = "deadbeef"
    return w


def _gpu_info(**override) -> dict:
    info = {
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "gpu_memory_mb": 81920,
        "gpu_count": 1,
        "gpu_uuids": "GPU-abc",
    }
    info.update(override)
    return info


def _call_generate(check, wrapper, gpu_info):
    return check._generate_cipher_text(
        wrapper=wrapper,
        gpu_info=gpu_info,
        dim_n=1900,
        dim_k=2000,
        seed=42,
        challenge_uuid="challenge-uuid-xyz",
    )


# --- Bad spec short-circuits before native allocation -----------------------
def test_bad_spec_raises_before_native_alloc(mock_wrapper):
    check = MatrixValidationCheck()
    bad = _gpu_info(gpu_memory_mb=24064)
    with pytest.raises(CipherGenError) as exc:
        _call_generate(check, mock_wrapper, bad)
    assert "precheck rejected" in str(exc.value)
    mock_wrapper.create_verifier.assert_not_called()
    mock_wrapper.set_dimension.assert_not_called()
    mock_wrapper.generate_challenge.assert_not_called()


def test_unknown_model_raises_before_native_alloc(mock_wrapper):
    check = MatrixValidationCheck()
    bad = _gpu_info(gpu_model="Totally Unknown GPU")
    with pytest.raises(CipherGenError):
        _call_generate(check, mock_wrapper, bad)
    mock_wrapper.create_verifier.assert_not_called()


# --- Happy path proceeds ----------------------------------------------------
def test_happy_path_proceeds(mock_wrapper):
    check = MatrixValidationCheck()
    cipher = _call_generate(check, mock_wrapper, _gpu_info())
    assert cipher == "deadbeef"
    mock_wrapper.create_verifier.assert_called_once_with(10, 10)
    mock_wrapper.set_dimension.assert_called_once()
    mock_wrapper.generate_challenge.assert_called_once()
    mock_wrapper.free.assert_called_once()


def test_machine_info_does_not_contain_gpu_capacity_mb(mock_wrapper):
    """Critical invariant: machine_info passed to libdmcompverify must NOT
    contain gpu_capacity_mb. The .so's decrypt side rebuilds machine_info
    via getGPUInfo() locally ({gpu_count, gpu_model, uuids} only) and any
    extra field breaks the AES-key derivation.
    """
    check = MatrixValidationCheck()
    _call_generate(check, mock_wrapper, _gpu_info())
    args, _kwargs = mock_wrapper.generate_challenge.call_args
    machine_info_str = args[2]  # (verifier_ptr, seed, machine_info, uuid)
    assert "gpu_capacity_mb" not in machine_info_str, (
        f"machine_info contains gpu_capacity_mb, breaks libdmcompverify "
        f"cryptographic binding: {machine_info_str!r}"
    )
