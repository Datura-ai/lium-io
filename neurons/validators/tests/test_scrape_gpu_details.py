"""One card NVML errors on (a hung card answers 999) must not hide the cards after it.

machine_scrape.py is a script, so read_gpu_details is extracted by ast with NVML stubbed in its namespace.
"""

from pathlib import Path
from types import SimpleNamespace

from neurons.validators.tests.helpers import build_scrape_namespace

SCRAPE = Path(__file__).resolve().parents[1] / "src" / "miner_jobs" / "machine_scrape.py"


class NVMLError(Exception):
    def __repr__(self) -> str:
        return f"NVMLError({self.args[0]})"


def _hung(handle):
    if handle == 1:
        raise NVMLError(999)
    return handle


def test_a_hung_card_is_skipped_and_the_rest_are_read():
    stubs = {
        name: (lambda handle, *_: _hung(handle))
        for name in (
            "nvmlDeviceGetName",
            "nvmlDeviceGetPowerManagementLimit",
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
        "NVML_CLOCK_GRAPHICS": 0,
        "NVML_CLOCK_MEM": 1,
        "NVML_TEMPERATURE_GPU": 0,
        "nvmlDeviceGetHandleByIndex": lambda i: i,
        "nvmlDeviceGetCudaComputeCapability": lambda handle: (9, 0),
        "nvmlDeviceGetUtilizationRates": lambda handle: SimpleNamespace(
            c_nvmlUtilization_t_gpu=0, c_nvmlUtilization_t_memory=0
        ),
        "nvmlDeviceGetMemoryInfo": lambda handle: SimpleNamespace(
            c_nvmlMemory_t_total=0, c_nvmlMemory_t_used=0
        ),
        "nvmlDeviceGetPowerManagementLimitConstraints": lambda handle: (0, 0),
        "nvmlDeviceGetUUID": lambda handle: f"GPU-{_hung(handle)}",
        "nvmlDeviceGetComputeRunningProcesses_v2": lambda handle: [
            SimpleNamespace(c_nvmlProcessInfo_v2_t_pid=100 + handle)
        ],
    }
    scrape = build_scrape_namespace(SCRAPE, {"read_gpu_details", "safeNvmlValue"}, stubs)
    pids: set[int] = set()

    details, error = scrape["read_gpu_details"](4, pids)

    assert [row["gpu.uuid"] for row in details] == ["GPU-0", "GPU-2", "GPU-3"]
    assert error == "NVMLError(999)"
    assert pids == {100, 102, 103}
