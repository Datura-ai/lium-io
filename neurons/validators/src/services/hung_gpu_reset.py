"""DAH-5301: a Pod Reboot (or edit) also resets a pod GPU that hung.

The create path parks the pod's old container (stopped) before it runs the new one. In that window
one SSH query looks for a hung pod GPU; a healthy reboot costs only that query. A hung GPU is reset
under the same refusal rules as the renter's Reset GPU (gpu_reset.py): NVLink peers must be inside
the pod, no MIG, no process on the GPUs (the pod's own container is stopped, so any is foreign), and
`nvidia-smi -r` only with `-i` on the pod's GPUs, with Fabric Manager stopped around it on an NVSwitch
pod that owns every GPU (as reset_pod_gpus does); when the driver refuses it, the DAH-2427 context
cycle is the fallback.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from payload_models.payloads import ContainerWarningCode, GpuResetErrorCode
from services.gpu_reset import (
    _RESET_TIMEOUT_SECONDS,
    FABRIC_SERVICES,
    HOST_GPUS_COMMAND,
    HOST_SYSTEMCTL,
    NVSWITCH_COMMAND,
    TOPOLOGY_COMMAND,
    GpuResetRefused,
    HostGpu,
    _own_cgroup,
    _processes_outside,
    _run,
    _service_active,
    _start_fabric_services,
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
_KILL_AFTER_SECONDS = 5
_APPS_MARKER = "--compute-apps--"
# one SSH call: the GPU query (a hung card times out, drops out or reads as an error) and the compute
# apps the DAH-2427 latch signature needs
_TIMEOUT = f"timeout -k {_KILL_AFTER_SECONDS} {_NVIDIA_SMI_TIMEOUT_SECONDS}"  # SIGKILL a hung nvidia-smi ignoring TERM
HUNG_GPU_QUERY_COMMAND = (
    f"{_TIMEOUT} {GPU_QUERY_COMMAND}; echo {_APPS_MARKER}; {_TIMEOUT} {COMPUTE_APPS_QUERY_COMMAND}"
)
_QUERY_SSH_TIMEOUT_SECONDS = 2 * (_NVIDIA_SMI_TIMEOUT_SECONDS + _KILL_AFTER_SECONDS) + 10
_NVLINK_CELL = re.compile(r"\bNV\d+\b")
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


@dataclass(frozen=True)
class _ResetPlan:
    targets: list[str]
    # stopped around `nvidia-smi -r` and started again (an NVSwitch pod owning every GPU)
    fabric_services: list[str]


def parse_hung_pod_gpus(gpu_csv: str, latched: set[str], pod_gpu_uuids: list[str]) -> list[str]:
    """The pod GPUs missing from the GPU query, read as an error, or latched (DAH-2427)."""
    healthy = {
        line.split(",")[0].strip()
        for line in gpu_csv.splitlines()
        if not any(marker in line for marker in _ERROR_MARKERS)
    }
    return [uuid for uuid in pod_gpu_uuids if uuid not in healthy or uuid in latched]


async def reset_hung_pod_gpus(
    runner: SSHCommandRunner,
    pod_gpu_uuids: list[str],
    *,
    resetting: Callable[[], AbstractAsyncContextManager[list[HungGpuResult] | None]] = contextlib.nullcontext,
) -> HungGpuResult:
    """Call with the pod's container stopped. Changes nothing unless a pod GPU is hung.

    `resetting` is entered once a GPU is found hung and held until the reset is over (the caller's
    gpu_reset pending-pod mark, so the rented checks skip the node meanwhile). A list it yields gets the
    result before it exits."""
    query = await _run(runner, HUNG_GPU_QUERY_COMMAND, timeout=_QUERY_SSH_TIMEOUT_SECONDS)
    if query.exit_code == -1 and not (query.stdout or "").strip():
        # the SSH command itself failed: nothing is known about the GPUs (a hung nvidia-smi is the shell's 124)
        return HungGpuResult()
    gpu_csv, _, apps_csv = (query.stdout or "").partition(_APPS_MARKER)
    latched = set(parse_wedged_gpu_uuids(gpu_csv, apps_csv))
    hung = parse_hung_pod_gpus(gpu_csv, latched, pod_gpu_uuids)
    if not hung:
        return HungGpuResult()
    async with resetting() as reset_result_holder:
        result = await _reset(runner, hung, latched, pod_gpu_uuids)
        if reset_result_holder is not None:
            reset_result_holder.append(result)  # the caller acts on it while still holding its mark
    return result


async def _reset(runner: SSHCommandRunner, hung: list[str], latched: set[str], pod_gpu_uuids: list[str]) -> HungGpuResult:
    if set(hung) <= latched:
        # latch only: the seconds-long context cycle first, not a reset (a healthy GPU stopped a moment
        # ago can still read busy); -r below only when the cycle does not clear it
        cures = await cure_wedged_gpus(runner, hung)
        if cures and all(cure.cured for cure in cures):
            return HungGpuResult(hung, ContainerWarningCode.HungGpuReset, f"latch cleared on {hung}")
    try:
        plan = await _reset_targets_or_refuse(runner, hung, pod_gpu_uuids)
    except GpuResetRefused as refused:
        warning = _NOT_RESET_WARNING.get(refused.code, ContainerWarningCode.HungGpuNotResetUnsupported)
        return HungGpuResult(hung, warning, refused.detail)

    command = build_reset_command(plan.targets, container_uuids=pod_gpu_uuids, pod_gpu_uuids=pod_gpu_uuids)
    reset = None
    stop_problem: str | None = None
    stopped_services: list[str] = []
    try:
        # as in reset_pod_gpus: the driver refuses -r while Fabric Manager holds the GPUs (the pod owns them all)
        for unit in plan.fabric_services:
            stopped_services.append(unit)  # a failed stop may still have stopped it: start it regardless
            unit_stop = await _run(runner, f"{HOST_SYSTEMCTL} stop {unit}", timeout=60)
            if unit_stop.exit_code != 0:
                stop_problem = f"could not stop {unit}: {_tail(unit_stop.stderr)}"
                break
        else:
            reset = await _run(runner, command, timeout=_RESET_TIMEOUT_SECONDS)
    finally:
        fabric_manager_problem = await _start_fabric_services(runner, stopped_services) if stopped_services else None
    if reset is None or fabric_manager_problem:
        detail = "; ".join(problem for problem in (stop_problem, fabric_manager_problem) if problem)
        return HungGpuResult(hung, ContainerWarningCode.HungGpuNotResetUnsupported, detail)
    if reset.exit_code == 0:
        return HungGpuResult(hung, ContainerWarningCode.HungGpuReset, f"reset {plan.targets}")
    # as in reset_pod_gpus: the context cycle is the fix when it cures every target
    cures = await cure_wedged_gpus(runner, plan.targets)
    detail = f"nvidia-smi -r refused: {_tail(reset.stdout or reset.stderr)}"
    if cures and all(cure.cured for cure in cures):
        return HungGpuResult(hung, ContainerWarningCode.HungGpuReset, f"{detail}; latch cleared on {plan.targets}")
    return HungGpuResult(hung, ContainerWarningCode.HungGpuNotResetUnsupported, detail)


async def _reset_targets_or_refuse(
    runner: SSHCommandRunner, hung: list[str], pod_gpu_uuids: list[str]
) -> _ResetPlan:
    """The hung GPUs widened to their NVLink peers, and the fabric services to stop around the reset,
    after Reset GPU's refusal rules."""
    host_gpus = parse_host_gpus((await _run(runner, HOST_GPUS_COMMAND)).stdout or "")
    if not host_gpus:
        raise GpuResetRefused(GpuResetErrorCode.ResetNotSupported, "nvidia-smi listed no GPU on the host")
    listed = {gpu.uuid for gpu in host_gpus}
    unlisted = [uuid for uuid in pod_gpu_uuids if uuid not in listed]
    if unlisted and not listed <= set(pod_gpu_uuids) and await _has_nvlink(runner):
        # a GPU gone from the listing has no `topo -m` row: its NVLink peers cannot be checked
        raise GpuResetRefused(
            GpuResetErrorCode.NvlinkPeersOutsidePod, f"NVLink peers of {unlisted} unknown on a shared host"
        )
    # it is still the pod's, so "the pod owns every host GPU" holds
    host_gpus += [HostGpu(-1, uuid, "", False) for uuid in unlisted]
    targets, fabric_services = await _widen_targets_to_nvlink_peers(runner, hung, pod_gpu_uuids, host_gpus)
    if any(gpu.mig_enabled for gpu in host_gpus if gpu.uuid in targets):
        raise GpuResetRefused(GpuResetErrorCode.MigEnabled, "MIG is enabled on a GPU to reset")
    outside = await _processes_outside(runner, targets, None, ignore_cgroup=await _own_cgroup(runner))
    if outside:
        raise GpuResetRefused(GpuResetErrorCode.GpuInUseOutsidePod, f"processes use the GPUs: {outside}")
    return _ResetPlan(targets, fabric_services)


async def _has_nvlink(runner: SSHCommandRunner) -> bool:
    """An NVSwitch fabric (its devices or a running fabric service) or any NVLink cell in `topo -m`; an
    unreadable topology counts as NVLink."""
    if ((await _run(runner, NVSWITCH_COMMAND)).stdout or "").strip():
        return True
    if any([await _service_active(runner, unit) for unit in FABRIC_SERVICES]):
        return True
    topology = await _run(runner, TOPOLOGY_COMMAND)
    return topology.exit_code != 0 or bool(_NVLINK_CELL.search(topology.stdout or ""))
