"""DAH-5301: a Pod Reboot (or edit) also resets a pod GPU that hung.

The create path parks the pod's old container (stopped) before it runs the new one. In that window
one SSH query looks for a hung pod GPU; a healthy reboot costs only that query. A hung GPU is reset
under the same refusal rules as the renter's Reset GPU (gpu_reset.py): NVLink peers must be inside
the pod, no MIG, no process on the GPUs (the pod's own container is stopped, so any is foreign), and
`nvidia-smi -r` only with `-i` on the pod's GPUs; when the driver refuses it, the DAH-2427 context
cycle is the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from payload_models.payloads import ContainerWarningCode, GpuResetErrorCode
from services.gpu_reset import (
    _RESET_TIMEOUT_SECONDS,
    HOST_GPUS_COMMAND,
    GpuResetRefused,
    HostGpu,
    _own_cgroup,
    _processes_outside,
    _run,
    _tail,
    _widen_targets_to_nvlink_peers,
    build_reset_command,
    parse_host_gpus,
)
from services.gpu_wedge import (
    COMPUTE_APPS_QUERY_COMMAND,
    GPU_QUERY_COMMAND,
    cure_wedged_gpus,
    parse_wedged_gpu_uuids,
)

if TYPE_CHECKING:
    from services.task.runner import SSHCommandRunner

_NVIDIA_SMI_TIMEOUT_SECONDS = 20
_APPS_MARKER = "--compute-apps--"
# one SSH call: the GPU query (a hung card times out, drops out or reads as an error) and the compute
# apps the DAH-2427 latch signature needs
HUNG_GPU_QUERY_COMMAND = (
    f"timeout {_NVIDIA_SMI_TIMEOUT_SECONDS} {GPU_QUERY_COMMAND}; echo {_APPS_MARKER}; "
    f"timeout {_NVIDIA_SMI_TIMEOUT_SECONDS} {COMPUTE_APPS_QUERY_COMMAND}"
)
_ERROR_MARKERS = ("Unknown Error", "GPU is lost", "ERR!")
_NOT_RESET_WARNING = {
    GpuResetErrorCode.MigEnabled: ContainerWarningCode.HungGpuNotResetMig,
    GpuResetErrorCode.NvlinkPeersOutsidePod: ContainerWarningCode.HungGpuNotResetNvlinkShared,
    GpuResetErrorCode.GpuInUseOutsidePod: ContainerWarningCode.HungGpuNotResetInUse,
}


@dataclass
class HungGpuResult:
    hung_uuids: list[str] = field(default_factory=list)
    # None: no pod GPU hung, nothing changed
    warning: ContainerWarningCode | None = None
    detail: str | None = None


def parse_hung_pod_gpus(output: str, pod_gpu_uuids: list[str]) -> list[str]:
    """The pod GPUs missing from the query, read as an error, or latched (DAH-2427)."""
    gpu_csv, _, apps_csv = output.partition(_APPS_MARKER)
    healthy = {
        line.split(",")[0].strip()
        for line in gpu_csv.splitlines()
        if not any(marker in line for marker in _ERROR_MARKERS)
    }
    latched = set(parse_wedged_gpu_uuids(gpu_csv, apps_csv))
    return [uuid for uuid in pod_gpu_uuids if uuid not in healthy or uuid in latched]


async def reset_hung_pod_gpus(runner: SSHCommandRunner, pod_gpu_uuids: list[str]) -> HungGpuResult:
    """Call with the pod's container stopped. Changes nothing unless a pod GPU is hung."""
    query = await _run(runner, HUNG_GPU_QUERY_COMMAND, timeout=2 * _NVIDIA_SMI_TIMEOUT_SECONDS + 10)
    hung = parse_hung_pod_gpus(query.stdout or "", pod_gpu_uuids)
    if not hung:
        return HungGpuResult()
    try:
        targets = await _checked_targets(runner, hung, pod_gpu_uuids)
    except GpuResetRefused as refused:
        warning = _NOT_RESET_WARNING.get(refused.code, ContainerWarningCode.HungGpuNotResetUnsupported)
        return HungGpuResult(hung, warning, refused.detail)

    reset = await _run(
        runner,
        build_reset_command(targets, container_uuids=pod_gpu_uuids, pod_gpu_uuids=pod_gpu_uuids),
        timeout=_RESET_TIMEOUT_SECONDS,
    )
    if reset.exit_code == 0:
        return HungGpuResult(hung, ContainerWarningCode.HungGpuReset, f"reset {targets}")
    # as in reset_pod_gpus: the context cycle is the fix when it cures every target
    cures = await cure_wedged_gpus(runner, targets)
    detail = f"nvidia-smi -r refused: {_tail(reset.stdout or reset.stderr)}"
    if cures and all(cure.cured for cure in cures):
        return HungGpuResult(hung, ContainerWarningCode.HungGpuReset, f"{detail}; latch cleared on {targets}")
    return HungGpuResult(hung, ContainerWarningCode.HungGpuNotResetUnsupported, detail)


async def _checked_targets(runner: SSHCommandRunner, hung: list[str], pod_gpu_uuids: list[str]) -> list[str]:
    """The hung GPUs widened to their NVLink peers, after Reset GPU's refusal rules."""
    host_gpus = parse_host_gpus((await _run(runner, HOST_GPUS_COMMAND)).stdout or "")
    if not host_gpus:
        raise GpuResetRefused(GpuResetErrorCode.ResetNotSupported, "nvidia-smi listed no GPU on the host")
    listed = {gpu.uuid for gpu in host_gpus}
    # a lost pod GPU can drop out of the listing: it is still the pod's, so "the pod owns every host GPU" holds
    host_gpus += [HostGpu(-1, uuid, "", False) for uuid in pod_gpu_uuids if uuid not in listed]
    targets = await _widen_targets_to_nvlink_peers(runner, hung, pod_gpu_uuids, host_gpus)
    if any(gpu.mig_enabled for gpu in host_gpus if gpu.uuid in targets):
        raise GpuResetRefused(GpuResetErrorCode.MigEnabled, "MIG is enabled on a GPU to reset")
    outside = await _processes_outside(runner, targets, None, ignore_cgroup=await _own_cgroup(runner))
    if outside:
        raise GpuResetRefused(GpuResetErrorCode.GpuInUseOutsidePod, f"processes use the GPUs: {outside}")
    return targets
