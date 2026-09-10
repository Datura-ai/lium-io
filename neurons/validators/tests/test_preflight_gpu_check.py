"""Preflight GPUCheck: the NVML name is canonicalised before the supported-model lookup (DAH-3339)."""

from __future__ import annotations

import pytest

from preflight.base import CheckStatus
from preflight.checks import gpu_check as gpu_check_module
from preflight.checks.gpu_check import GPUCheck
from preflight.main import create_default_context


def _gpu_info(gpu_model: str, gpu_count: int = 1) -> dict:
    """What preflight.utils.get_gpu_info returns for an idle host of `gpu_count` cards."""
    return {
        "gpu_count": gpu_count,
        "gpu_model": gpu_model,
        "gpu_uuids": ",".join(f"GPU-{i}" for i in range(gpu_count)),
        "uuids": ",".join(f"GPU-{i}" for i in range(gpu_count)),
        "gpu_details": [
            {
                "index": i,
                "name": gpu_model,
                "uuid": f"GPU-{i}",
                "utilization": 0,
                "memory_utilization": 0,
                "memory_total_mb": 23028,
            }
            for i in range(gpu_count)
        ],
    }


async def _run(monkeypatch, gpu_model: str):
    monkeypatch.setattr(
        gpu_check_module, "get_gpu_info", lambda include_utilization=False: _gpu_info(gpu_model)
    )
    return await GPUCheck().run(create_default_context())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "nvml_name,canonical",
    [
        ("NVIDIA A10G", "NVIDIA A10 Tensor Core GPU"),  # AWS g5
        ("NVIDIA A10", "NVIDIA A10 Tensor Core GPU"),
        ("Tesla T4", "NVIDIA T4 Tensor Core GPU"),
    ],
)
async def test_gpu_check_accepts_nvml_spelling_of_a_supported_model(
    monkeypatch, nvml_name, canonical
):
    result = await _run(monkeypatch, nvml_name)

    assert result.status == CheckStatus.PASSED
    assert result.message == f"GPU validation passed: 1x {canonical}"


@pytest.mark.asyncio
async def test_gpu_check_passes_a_canonical_name_unchanged(monkeypatch):
    result = await _run(monkeypatch, "NVIDIA L4")

    assert result.status == CheckStatus.PASSED
    assert result.message == "GPU validation passed: 1x NVIDIA L4"


@pytest.mark.asyncio
async def test_gpu_check_still_refuses_an_unknown_model(monkeypatch):
    result = await _run(monkeypatch, "NVIDIA A10Z")

    assert result.status == CheckStatus.FAILED
    assert result.message.startswith("GPU model 'NVIDIA A10Z' is not supported.")
