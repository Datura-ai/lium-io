"""DAH-5863: NVIDIA GB10 (DGX Spark) is a known model, and its GPU uses the host RAM.

NVML answers Not Supported for a GB10's memory (and its power cap), so the scrape reports the host RAM as its
capacity; every VRAM reading downstream (precheck window, disk-to-VRAM rule, matmul size) then reads that.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from incentive.config import BASE_GPU_MAP, MAX_UNRENTED_GPUS_BY_TYPE, RENTAL_PRICES_PER_HOUR
from neurons.validators.tests.helpers import build_scrape_namespace
from preflight import utils as preflight_utils
from services.const import GPU_MODEL_RATES
from services.gpu_reference_table import GPU_REFERENCE_TABLE
from services.gpu_precheck import GpuPrecheckError, precheck_gpu_spec
from services.gpu_spec_table import matmul_memory_mb, reference_gpu_model

GB10 = "NVIDIA GB10"
SCRAPE = Path(__file__).resolve().parents[1] / "src" / "miner_jobs" / "machine_scrape.py"
NOT_SUPPORTED = 3
HOST_RAM_BYTES = 121_856 * 1024**2  # a DGX Spark's MemTotal, ~119 GiB of its 128 GB


class NVMLError(Exception):
    def __init__(self, value):
        super().__init__(value)
        self.value = value


def _raise(code):
    raise NVMLError(code)


def _scrape(memory_info, power_limit=lambda handle: 300_000):
    stubs = {
        name: (lambda handle, *_: 0)
        for name in (
            "nvmlDeviceGetPowerManagementDefaultLimit",
            "nvmlDeviceGetClockInfo",
            "nvmlDeviceGetMaxClockInfo",
            "nvmlDeviceGetCurrPcieLinkWidth",
            "nvmlDeviceGetPcieSpeed",
            "nvmlDeviceGetTemperature",
            "nvmlDeviceGetClocksEventReasons",
        )
    }
    stubs |= {
        "NVMLError": NVMLError,
        "NVML_ERROR_NOT_SUPPORTED": NOT_SUPPORTED,
        "NVML_CLOCK_GRAPHICS": 0,
        "NVML_CLOCK_MEM": 1,
        "NVML_TEMPERATURE_GPU": 0,
        "psutil": SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=HOST_RAM_BYTES)),
        "nvmlDeviceGetName": lambda handle: GB10 if handle == 0 else "NVIDIA H100 80GB HBM3",
        "nvmlDeviceGetHandleByIndex": lambda i: i,
        "nvmlDeviceGetCudaComputeCapability": lambda handle: (12, 1),
        "nvmlDeviceGetUtilizationRates": lambda handle: SimpleNamespace(
            c_nvmlUtilization_t_gpu=0, c_nvmlUtilization_t_memory=0
        ),
        "nvmlDeviceGetMemoryInfo": memory_info,
        "nvmlDeviceGetPowerManagementLimit": power_limit,
        "nvmlDeviceGetPowerManagementLimitConstraints": lambda handle: (0, 0),
        "nvmlDeviceGetUUID": lambda handle: f"GPU-{handle}",
        "nvmlDeviceGetComputeRunningProcesses_v2": lambda handle: [],
    }
    return build_scrape_namespace(
        SCRAPE, {"read_gpu_details", "read_gpu_memory_mb", "UNIFIED_MEMORY_GPU_NAMES", "safeNvmlValue"}, stubs
    )


def test_gb10_is_its_own_model_and_idle_tier():
    assert GPU_MODEL_RATES[GB10] == 0.0
    assert BASE_GPU_MAP[GB10] == "GB10"
    assert MAX_UNRENTED_GPUS_BY_TYPE["GB10"] == {1: 10, 8: 64}
    # the RTX 5090's listing price (the backend's, in the reference table), as the other added types
    assert RENTAL_PRICES_PER_HOUR[GB10] == 0.25
    assert GB10 not in GPU_REFERENCE_TABLE
    assert reference_gpu_model(GB10, HOST_RAM_BYTES / 1024**2) == GB10


def test_the_host_ram_reading_passes_the_gb10_precheck_and_half_of_it_does_not():
    precheck_gpu_spec(GB10, HOST_RAM_BYTES // 1024**2)
    with pytest.raises(GpuPrecheckError):
        precheck_gpu_spec(GB10, HOST_RAM_BYTES // 1024**2 // 2)


def test_a_gb10_reads_the_host_ram_and_no_used_memory_and_tolerates_no_power_cap():
    scrape = _scrape(lambda handle: _raise(NOT_SUPPORTED), power_limit=lambda handle: _raise(NOT_SUPPORTED))

    details, error = scrape["read_gpu_details"](1, set())

    assert error is None
    assert details[0]["gpu.capacity"] == 121_856
    assert details[0]["gpu.memory_used_mb"] is None
    assert details[0]["gpu.power_limit"] is None


@pytest.mark.parametrize(
    ("handle", "code"),
    [(1, NOT_SUPPORTED), (0, 999)],
    ids=["another card with no memory reading", "a gb10 with another nvml error"],
)
def test_only_a_gb10_not_supported_falls_back_to_the_host_ram(handle, code):
    scrape = _scrape(lambda handle: _raise(code))

    with pytest.raises(NVMLError):
        scrape["read_gpu_memory_mb"](handle, scrape["nvmlDeviceGetName"](handle))


def test_a_card_with_its_own_memory_reads_nvml():
    scrape = _scrape(
        lambda handle: SimpleNamespace(c_nvmlMemory_t_total=80 * 1024**3, c_nvmlMemory_t_used=1024**3)
    )

    assert scrape["read_gpu_memory_mb"](1, "NVIDIA H100 80GB HBM3") == (81920, 1024)


def test_a_gb10_matmul_leaves_the_host_its_share_of_the_memory():
    assert matmul_memory_mb(GB10, 120_000) == 90_000
    assert matmul_memory_mb("NVIDIA H100 80GB HBM3", 81_559) == 81_559
    assert matmul_memory_mb(GB10, None) is None


def test_preflight_reads_the_host_ram_for_a_gb10(monkeypatch):
    def no_memory(handle):
        raise RuntimeError("Not Supported")

    pynvml = SimpleNamespace(nvmlDeviceGetMemoryInfo=no_memory)
    monkeypatch.setattr(
        preflight_utils.psutil, "virtual_memory", lambda: SimpleNamespace(total=HOST_RAM_BYTES)
    )

    assert preflight_utils._memory_total_mb(pynvml, None, GB10) == 121_856
    assert preflight_utils._memory_total_mb(pynvml, None, "NVIDIA H100 80GB HBM3") == 0
