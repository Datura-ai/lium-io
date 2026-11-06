"""DAH-5301: a Pod Reboot (or edit) also resets a pod GPU that hung.

The create path parks the pod's old container (stopped) before it runs the new one. In that window
one SSH query looks for a hung pod GPU; a healthy reboot costs only that query. A hung GPU is reset
under the same refusal rules as the renter's Reset GPU (gpu_reset.py): NVLink peers must be inside
the pod, no MIG, no process on the GPUs (the pod's own container is stopped, so any is foreign), and
`nvidia-smi -r` only with `-i` on the pod's GPUs, with every host GPU holder (Fabric Manager, DCGM, a
dcgm-exporter container, IMEX, nvidia-persistenced) stopped around it on a pod that owns every GPU, all in one
host systemd unit (the same plan as reset_pod_gpus); when the driver refuses it, the DAH-2427 context cycle is the
fallback. A latch clear is never reported as a reset (HungGpuNotResetUnsupported, as reset_pod_gpus reports it not ok).
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
    _FABRIC_START_FAILED_EXIT,
    _FABRIC_STOP_FAILED_EXIT,
    _RESET_TIMEOUT_SECONDS,
    FABRIC_SERVICES,
    HOST_GPUS_COMMAND,
    NVSWITCH_COMMAND,
    TOPOLOGY_COMMAND,
    GpuResetRefused,
    HostGpu,
    _own_cgroup,
    _processes_outside,
    _reset_unit_gone,
    _reset_unit_pending,
    _restore_fabric_services,
    _run,
    _service_active,
    _tail,
    _widen_targets_to_nvlink_peers,
    build_fabric_reset_command,
    build_reset_command,
    parse_host_gpus,
    plan_host_holders,
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
    # None: nothing to tell the renter (no pod GPU hung, or the latch clear alone fixed it)
    warning: ContainerWarningCode | None = None
    detail: str | None = None


@dataclass(frozen=True)
class _ResetPlan:
    targets: list[str]
    # as gpu_reset._Plan: stopped around `nvidia-smi -r` and started again (a pod owning every host GPU)
    host_services: list[str] = field(default_factory=list)
    holder_containers: list[str] = field(default_factory=list)
    owns_host: bool = False


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
    settle_seconds: float = 2.0,
    shadow: bool = False,
) -> HungGpuResult:
    """Call with the pod's container stopped. Changes nothing unless a pod GPU is hung.

    `shadow`: only read the host and say what would happen in `detail` (no latch cure, no reset, `resetting` is
    never entered, no warning).

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
    if shadow:
        return HungGpuResult(hung, None, await _shadow_detail(runner, hung, latched, pod_gpu_uuids))
    async with resetting() as reset_result_holder:
        result = await _reset(runner, hung, latched, pod_gpu_uuids, settle_seconds)
        if reset_result_holder is not None:
            reset_result_holder.append(result)  # the caller acts on it while still holding its mark
    return result


async def _shadow_detail(
    runner: SSHCommandRunner, hung: list[str], latched: set[str], pod_gpu_uuids: list[str]
) -> str:
    """What _reset would do, from the same read-only checks."""
    reason = "latched (busy, no memory, no process)" if set(hung) <= latched else "not healthy in the GPU query"
    if await _reset_unit_pending(runner):
        return f"{reason}; would refuse: the GPU reset is still running on the host"
    first = "would try the latch clear first, then " if set(hung) <= latched else "would "
    try:
        plan = await _reset_targets_or_refuse(runner, hung, pod_gpu_uuids)
    except GpuResetRefused as refused:
        return f"{reason}; {first}refuse the reset: {refused.code.value}: {refused.detail}"
    where = (
        f" in the host unit, stopping {plan.host_services + plan.holder_containers}" if plan.owns_host else ""
    )
    return f"{reason}; {first}reset {plan.targets}{where}"


async def _reset(
    runner: SSHCommandRunner, hung: list[str], latched: set[str], pod_gpu_uuids: list[str], settle_seconds: float
) -> HungGpuResult:
    if await _reset_unit_pending(runner):
        # as in reset_pod_gpus: an earlier reset that outlived its SSH timeout still holds the GPUs and the fabric
        detail = "the GPU reset is still running on the host"
        return HungGpuResult(hung, ContainerWarningCode.HungGpuNotResetUnsupported, detail)
    if set(hung) <= latched:
        # latch only: the seconds-long context cycle first, not a reset (a healthy GPU stopped a moment
        # ago can still read busy); -r below only when the cycle does not clear it
        cures = await cure_wedged_gpus(runner, hung)
        if cures and all(cure.cured for cure in cures):
            # the GPUs work again: nothing to tell the renter, and no reset to report either
            return HungGpuResult(hung, None, f"only the latch clear ran on {hung}")
    try:
        plan = await _reset_targets_or_refuse(runner, hung, pod_gpu_uuids)
    except GpuResetRefused as refused:
        warning = _NOT_RESET_WARNING.get(refused.code, ContainerWarningCode.HungGpuNotResetUnsupported)
        return HungGpuResult(hung, warning, refused.detail)

    command = build_reset_command(plan.targets, container_uuids=pod_gpu_uuids, pod_gpu_uuids=pod_gpu_uuids)
    if not plan.owns_host:
        reset = await _run(runner, command, timeout=_RESET_TIMEOUT_SECONDS)
    else:
        # as in reset_pod_gpus: the driver refuses -r while Fabric Manager or another host client holds the GPUs
        # (the pod owns them all); stop, reset and start run as one host unit that outlives the SSH session
        reset = None
        try:
            reset = await _run(
                runner,
                build_fabric_reset_command(plan.host_services, command, plan.holder_containers),
                timeout=_RESET_TIMEOUT_SECONDS + 120,
            )
        finally:
            if (reset is None or reset.exit_code < 0) and not await _reset_unit_gone(runner):
                fabric_manager_problem = "the GPU reset is still running on the host"
            else:
                fabric_manager_problem = await _restore_fabric_services(
                    runner, plan.host_services, plan.holder_containers, settle_seconds
                )
        if fabric_manager_problem:
            return HungGpuResult(hung, ContainerWarningCode.HungGpuNotResetUnsupported, fabric_manager_problem)
        if reset.exit_code in (_FABRIC_STOP_FAILED_EXIT, _FABRIC_START_FAILED_EXIT):
            return HungGpuResult(
                hung,
                ContainerWarningCode.HungGpuNotResetUnsupported,
                _tail(reset.stderr) or "a host GPU holder did not stop or start",
            )
    if reset.exit_code == 0:
        return HungGpuResult(hung, ContainerWarningCode.HungGpuReset, f"reset {plan.targets}")
    # as in reset_pod_gpus: the context cycle at least clears the latch, but it is no reset
    cures = await cure_wedged_gpus(runner, plan.targets)
    detail = f"nvidia-smi -r refused, the GPUs were not reset: {_tail(reset.stdout or reset.stderr)}"
    if cures and all(cure.cured for cure in cures):
        detail += f"; only the latch clear ran on {plan.targets}"
    return HungGpuResult(hung, ContainerWarningCode.HungGpuNotResetUnsupported, detail)


async def _reset_targets_or_refuse(
    runner: SSHCommandRunner, hung: list[str], pod_gpu_uuids: list[str]
) -> _ResetPlan:
    """The hung GPUs widened to their NVLink peers, and the fabric services to stop around the reset,
    after Reset GPU's refusal rules."""
    host_gpus_result = await _run(runner, HOST_GPUS_COMMAND)
    host_gpus = parse_host_gpus(host_gpus_result.stdout or "")
    # as in reset_pod_gpus: a failed listing may be partial, and a partial one can make a shared host look like the pod's
    if host_gpus_result.exit_code != 0 or not host_gpus:
        raise GpuResetRefused(GpuResetErrorCode.ResetNotSupported, "nvidia-smi could not list the host GPUs")
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
    if {gpu.uuid for gpu in host_gpus} != set(pod_gpu_uuids):
        return _ResetPlan(targets)
    # the pod's own container is parked and stopped, so no running container is the pod's
    holders = await plan_host_holders(runner, fabric_services, None)
    return _ResetPlan(targets, holders.host_services, holders.holder_containers, holders.owns_host)


async def _has_nvlink(runner: SSHCommandRunner) -> bool:
    """An NVSwitch fabric (its devices or a running fabric service) or any NVLink cell in `topo -m`; an
    unreadable topology counts as NVLink."""
    if ((await _run(runner, NVSWITCH_COMMAND)).stdout or "").strip():
        return True
    if any([await _service_active(runner, unit) for unit in FABRIC_SERVICES]):
        return True
    topology = await _run(runner, TOPOLOGY_COMMAND)
    return topology.exit_code != 0 or bool(_NVLINK_CELL.search(topology.stdout or ""))
