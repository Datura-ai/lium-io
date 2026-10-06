"""DAH-2427: shared ghost-GPU signature and cure.

A hard-killed GPU process (SIGKILL via container force-remove, or a crash) can latch a
Blackwell GSP "engine busy" counter: the card reads 100% utilization with no memory and no
process while still being offered as available. The latch is telemetry-only and persists
because persistence mode keeps the device initialized. Two callers act on it — the periodic
GpuUsageCheck and the post-teardown sweep in DockerService — so the signature and the cure
live here once instead of drifting apart.
"""

import asyncio
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

from services.const import GPU_WEDGE_SWEEP_MEMORY_MAX_MIB, GPU_WEDGE_UTILIZATION_MIN

if TYPE_CHECKING:
    # Type-only: importing services.task at runtime pulls its package __init__, which reaches
    # back into the checks that import this module.
    from services.task.runner import SSHCommandRunner

GPU_QUERY_COMMAND = (
    "nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader,nounits"
)
COMPUTE_APPS_QUERY_COMMAND = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
NVIDIA_SMI_QUERY_TIMEOUT_SECONDS = 30

# A clean CUDA context create/destroy on the latched card clears the counter. Chosen over
# `nvidia-smi -r` (refused on ~45% of the Blackwell fleet: never available on 580.126.09,
# and blocked by ANY other NVML client elsewhere) and over killing NVML holders (only
# worked when a leaked fd existed, and stops the image pre-pull daemon). The context cycle
# needs no root, tolerates concurrent clients, and cured 11/11 latched cards in live tests
# on both driver branches. python3 + libcuda.so.1 ship in the executor container
# (NVIDIA_DRIVER_CAPABILITIES=all). Takes 2-5s per GPU.
_CUDA_CONTEXT_CYCLE_CODE = (
    "import ctypes; "
    'lib = ctypes.CDLL("libcuda.so.1"); '
    "assert lib.cuInit(0) == 0; "
    "dev = ctypes.c_int(); "
    "assert lib.cuDeviceGet(ctypes.byref(dev), 0) == 0; "
    "ctx = ctypes.c_void_p(); "
    "assert lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0; "
    "assert lib.cuCtxDestroy_v2(ctx) == 0; "
    'print("ctx open/close OK")'
)
CURE_SUCCESS_MARKER = "ctx open/close OK"
_CURE_TIMEOUT_SECONDS = 60
_CURE_OUTPUT_TAIL_CHARS = 300


@dataclass(frozen=True)
class GpuCureOutcome:
    gpu_uuid: str
    exit_code: int  # SSHCommandRunner reports -1 when the command could not run at all
    output: str
    error: str | None = None

    @property
    def cured(self) -> bool:
        return self.exit_code == 0 and CURE_SUCCESS_MARKER in self.output


def matches_wedge_utilization(gpu_utilization: float | None) -> bool:
    return (gpu_utilization or 0) >= GPU_WEDGE_UTILIZATION_MIN


def build_cuda_context_cycle_command(gpu_uuid: str) -> str:
    return (
        f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_uuid)} "
        f"python3 -c {shlex.quote(_CUDA_CONTEXT_CYCLE_CODE)}"
    )


def parse_wedged_gpu_uuids(gpu_query_csv: str, compute_apps_csv: str) -> list[str]:
    """Pick wedged cards out of GPU_QUERY_COMMAND output (uuid, utilization, memory.used MiB).

    A host with any live compute app is skipped entirely: a running CUDA process can
    legitimately pin a card, and only orphaned load is the wedge signature.
    """
    if compute_apps_csv.strip():
        return []

    wedged_uuids: list[str] = []
    for line in gpu_query_csv.strip().splitlines():
        parts: list[str] = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        gpu_uuid, utilization_raw, memory_raw = parts
        try:
            utilization = float(utilization_raw)
            memory_mib = float(memory_raw)
        except ValueError:
            continue
        is_wedged: bool = (
            gpu_uuid.startswith("GPU-")
            and matches_wedge_utilization(utilization)
            and memory_mib <= GPU_WEDGE_SWEEP_MEMORY_MAX_MIB
        )
        if is_wedged:
            wedged_uuids.append(gpu_uuid)
    return wedged_uuids


async def query_wedged_gpu_uuids(runner: "SSHCommandRunner") -> list[str]:
    """Sample the live nvidia-smi state and return the currently wedged card uuids."""
    gpu_query, compute_apps = await asyncio.gather(
        runner.run(GPU_QUERY_COMMAND, timeout=NVIDIA_SMI_QUERY_TIMEOUT_SECONDS, retryable=False),
        runner.run(
            COMPUTE_APPS_QUERY_COMMAND, timeout=NVIDIA_SMI_QUERY_TIMEOUT_SECONDS, retryable=False
        ),
    )
    return parse_wedged_gpu_uuids(gpu_query.stdout, compute_apps.stdout)


async def cure_wedged_gpus(
    runner: "SSHCommandRunner", gpu_uuids: list[str]
) -> list[GpuCureOutcome]:
    """Clear each latched card with a CUDA context create/destroy cycle, sequentially.

    SSHCommandRunner never raises — a failure comes back as exit_code -1 plus
    error_message — so this cannot break its caller.
    """
    outcomes: list[GpuCureOutcome] = []
    for gpu_uuid in gpu_uuids:
        result = await runner.run(
            build_cuda_context_cycle_command(gpu_uuid),
            timeout=_CURE_TIMEOUT_SECONDS,
            retryable=False,
        )
        output: str = result.stdout or result.stderr or ""
        outcomes.append(
            GpuCureOutcome(
                gpu_uuid=gpu_uuid,
                exit_code=result.exit_code,
                output=output[-_CURE_OUTPUT_TAIL_CHARS:],
                error=result.error_message,
            )
        )
    return outcomes
