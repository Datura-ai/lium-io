"""Integration tests for the validator-side GPU pre-check.

Pre-check now lives at the top of `validate_gpu_model_and_process_job` (the
caller), NOT inside `encrypt_challenge`. This is required because anything
embedded in `machine_info` JSON passed to libdmcompverify must exactly
match what the executor's .so reconstructs locally via getGPUInfo() —
adding fields breaks the cryptographic binding.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


def _machine_spec(*, gpu_model="NVIDIA H100 80GB HBM3", capacity=81920, count=1):
    return {
        "gpu": {
            "count": count,
            "details": [
                {"name": gpu_model, "uuid": "GPU-abc", "capacity": capacity},
            ],
        },
    }


def _ssh_client():
    """Mock SSH client whose .run is async."""
    return SimpleNamespace(run=AsyncMock())


def _executor_info():
    return SimpleNamespace(
        root_dir="/root/app",
        python_path="/root/app/.venv/bin/python",
    )


# --- Bad spec short-circuits before any native call AND no SSH --------------
@pytest.mark.asyncio
async def test_bad_vram_short_circuits(service_with_mock_wrapper):
    """Out-of-range VRAM → ValidationResult(success=False), no .so call, no SSH."""
    svc, wrapper = service_with_mock_wrapper
    ssh = _ssh_client()
    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=ssh,
        executor_info=_executor_info(),
        default_extra={},
        machine_spec=_machine_spec(gpu_model="NVIDIA H100 80GB HBM3", capacity=24064),
    )
    assert result.success is False
    assert "precheck" in result.error_message.lower()
    wrapper.setDimension.assert_not_called()
    wrapper.generateChallenge.assert_not_called()
    ssh.run.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_model_short_circuits(service_with_mock_wrapper):
    svc, wrapper = service_with_mock_wrapper
    ssh = _ssh_client()
    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=ssh,
        executor_info=_executor_info(),
        default_extra={},
        machine_spec=_machine_spec(gpu_model="Totally Unknown GPU", capacity=24064),
    )
    assert result.success is False
    assert "precheck" in result.error_message.lower()
    wrapper.setDimension.assert_not_called()
    ssh.run.assert_not_called()


@pytest.mark.asyncio
async def test_missing_capacity_short_circuits(service_with_mock_wrapper):
    svc, wrapper = service_with_mock_wrapper
    ssh = _ssh_client()
    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=ssh,
        executor_info=_executor_info(),
        default_extra={},
        machine_spec=_machine_spec(capacity=0),
    )
    assert result.success is False
    wrapper.setDimension.assert_not_called()
    ssh.run.assert_not_called()


# --- Happy path: pre-check passes, machine_info has no gpu_capacity_mb ------
@pytest.mark.asyncio
async def test_machine_info_does_not_contain_gpu_capacity_mb(
    service_with_mock_wrapper,
):
    """Critical invariant: machine_info passed to libdmcompverify must NOT
    contain gpu_capacity_mb — the executor's .so reconstructs machine_info
    via getGPUInfo() locally, which produces {gpu_count, gpu_model, uuids}
    only. Including gpu_capacity_mb breaks the cryptographic AES-key binding
    and the executor's decrypt fails with returned_uuid=None.
    """
    svc, wrapper = service_with_mock_wrapper
    # SSH returns a stdout that the validator will parse — we don't care
    # about the UUID match outcome; we just want to inspect the call args.
    ssh = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(stdout="UUID: foo", stderr=""))
    )
    await svc.validate_gpu_model_and_process_job(
        ssh_client=ssh,
        executor_info=_executor_info(),
        default_extra={},
        machine_spec=_machine_spec(),
    )
    # Inspect the machine_info string that was handed to the native call.
    assert wrapper.generateChallenge.called, "generateChallenge should have been called"
    args, _kwargs = wrapper.generateChallenge.call_args
    machine_info_str = args[2]  # signature: (verifier_ptr, seed, machine_info, uuid)
    assert "gpu_capacity_mb" not in machine_info_str, (
        f"machine_info contains gpu_capacity_mb, which breaks libdmcompverify "
        f"cryptographic binding: {machine_info_str!r}"
    )
    # Sanity: required fields still present
    assert "gpu_model" in machine_info_str
    assert "gpu_count" in machine_info_str
    assert "uuids" in machine_info_str


# --- Empty cipher_text from encrypt_challenge → short-circuit before SSH ----
@pytest.mark.asyncio
async def test_empty_cipher_text_short_circuits_ssh(service_with_mock_wrapper):
    """If encrypt_challenge returns "" (e.g. unexpected .so failure), no SSH."""
    svc, _wrapper = service_with_mock_wrapper
    svc.encrypt_challenge = lambda *a, **kw: ""
    ssh = _ssh_client()
    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=ssh,
        executor_info=_executor_info(),
        default_extra={},
        machine_spec=_machine_spec(),
    )
    assert result.success is False
    assert "cipher" in result.error_message.lower()
    ssh.run.assert_not_called()
