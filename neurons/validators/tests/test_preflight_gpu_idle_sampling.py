"""Preflight GPU idle check: a start-up utilization spike must not refuse an idle GPU (ticket-0996)."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import patch

from preflight import utils
from preflight.base import CheckStatus
from preflight.checks.gpu_check import GPUCheck


def _fake_nvml(reads_by_gpu):
    """pynvml stand-in: each GPU returns its next (gpu, memory) utilization read per call."""
    iters = [iter(r) for r in reads_by_gpu]

    def rates(handle):
        gpu, mem = next(iters[handle])
        return SimpleNamespace(gpu=gpu, memory=mem)

    return SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: len(reads_by_gpu),
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetName=lambda h: "NVIDIA GeForce RTX 3090",
        nvmlDeviceGetUUID=lambda h: f"GPU-{h}",
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(total=24576 * 1024 * 1024),
        nvmlDeviceGetUtilizationRates=rates,
    )


def _run(reads_by_gpu):
    with patch.dict(sys.modules, {"pynvml": _fake_nvml(reads_by_gpu)}), patch.object(utils.time, "sleep"):
        return asyncio.run(GPUCheck().run({"gpu": {
            "max_gpu_count": 8, "gpu_utilization_limit": 5, "gpu_memory_utilization_limit": 5,
        }}))


def test_first_sample_spike_on_idle_gpu_passes():
    result = _run([[(100, 40), (0, 0), (0, 0), (0, 0), (0, 0)]])
    assert result.status == CheckStatus.PASSED, result.message


def test_busy_gpu_fails_with_measured_values():
    result = _run([[(0, 0)] * 5, [(97, 30), (95, 31), (99, 29), (96, 30), (98, 30)]])
    assert result.status == CheckStatus.FAILED
    assert "GPU 1: GPU 95%, memory 29%" in result.message
    assert "GPU 0" not in result.message


def test_unreadable_utilization_counts_as_idle():
    def not_supported(handle):
        raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")

    nvml = _fake_nvml([[]])
    nvml.nvmlDeviceGetUtilizationRates = not_supported
    with patch.dict(sys.modules, {"pynvml": nvml}), patch.object(utils.time, "sleep"):
        info = utils.get_gpu_info(include_utilization=True)
    assert info["gpu_details"][0]["utilization"] == 0
