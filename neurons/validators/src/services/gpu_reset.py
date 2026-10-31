"""DAH-4658: a renter resets the GPUs of their own running pod.

The validator does all of it over its SSH session into the executor, which runs privileged with
`pid: host`, every GPU and docker.sock, so `docker` and `nvidia-smi` here are the HOST's view.
Nothing is ever read from inside the pod (no `docker exec`): the renter controls that output.

    prechecks (change nothing) -> docker stop -t 10 -> no process left on the targets
        -> nvidia-smi -r -i <uuids> (or the DAH-2427 latch cure when the driver refuses) -> start

Safety invariants, each pinned by a test:
- the reset set is a subset of BOTH the container's own DeviceRequests GPUs (from the docker
  daemon) and the GPUs the backend holds for the pod;
- `nvidia-smi -r` never runs without `-i`;
- the container is always started again once it was stopped.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from payload_models.payloads import GpuResetErrorCode, GpuResetOutcome
from protocol.vc_protocol.validator_requests import ResetVerifiedJobEvidence
from services.gpu_wedge import cure_wedged_gpus
from services.redis_service import RecentGpuReset

if TYPE_CHECKING:
    # Type-only, as in gpu_wedge: services.task at runtime reaches back into the checks.
    from services.task.runner import SSHCommandRunner

DOCKER = "/usr/bin/docker"
HOST_GPUS_COMMAND = (
    "nvidia-smi --query-gpu=index,uuid,pci.bus_id,mig.mode.current --format=csv,noheader"
)
COMPUTE_APPS_COMMAND = "nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader"
TOPOLOGY_COMMAND = "nvidia-smi topo -m"
NVSWITCH_COMMAND = (
    'for p in /dev/nvidia-nvswitch[0-9]*; do [ -e "$p" ] && printf "%s\\n" "$p"; done; true'
)
STOP_GRACE_SECONDS = 10
_QUERY_TIMEOUT_SECONDS = 30
_RESET_TIMEOUT_SECONDS = 120
_DETAIL_TAIL_CHARS = 300
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_GPU_LABEL = re.compile(r"GPU\d+")


class GpuResetRefused(Exception):
    def __init__(self, code: GpuResetErrorCode, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass
class GpuResetReport:
    ok: bool
    error_code: GpuResetErrorCode | None = None
    container_restarted: bool = False
    gpus: list[GpuResetOutcome] = field(default_factory=list)
    detail: str | None = None
    # docker stop ran: the pod's GPUs may drop out until the driver brings them back
    pod_stopped: bool = False


@dataclass(frozen=True)
class HostGpu:
    index: int
    uuid: str
    bus_id: str
    mig_enabled: bool


def parse_host_gpus(csv: str) -> list[HostGpu]:
    gpus: list[HostGpu] = []
    for line in csv.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or not parts[1].startswith("GPU-"):
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        gpus.append(HostGpu(index, parts[1], parts[2].lower(), parts[3].lower() == "enabled"))
    return gpus


def container_gpu_uuids(device_requests: list[dict] | None, host_gpus: list[HostGpu]) -> list[str]:
    """The GPUs the docker daemon gave the container: its DeviceRequests ids, or every host GPU for a
    `--gpus all` request (Count -1)."""
    uuids: list[str] = []
    for request in device_requests or []:
        capabilities = [cap for group in request.get("Capabilities") or [] for cap in group]
        if "gpu" not in capabilities:
            continue
        if request.get("DeviceIDs"):
            uuids += [str(device_id) for device_id in request["DeviceIDs"]]
        elif request.get("Count") == -1:
            uuids += [gpu.uuid for gpu in host_gpus]
    return list(dict.fromkeys(uuids))


@dataclass
class PodGpus:
    ordered: list[str]  # the container's GPUs in pod order
    requested: list[str]  # the ones gpu_indices picks (None: the backend's)
    index_by_uuid: dict[str, int]  # uuid -> pod index


def pod_gpus(
    device_requests: list[dict] | None,
    host_gpus: list[HostGpu],
    pod_gpu_uuids: list[str],
    gpu_indices: list[int] | None,
) -> PodGpus:
    """The one numbering the reset and the clock lock share. Pod index = the container's GPUs by host PCI bus
    id, the 0..n-1 order nvidia-smi shows inside the pod. Refuses a backend GPU the container was not given,
    or an index it lacks."""
    host_by_uuid = {gpu.uuid: gpu for gpu in host_gpus}
    container_uuids = [uuid for uuid in container_gpu_uuids(device_requests, host_gpus) if uuid in host_by_uuid]
    missing = [uuid for uuid in pod_gpu_uuids if uuid not in container_uuids]
    if not container_uuids or not pod_gpu_uuids or missing:
        raise GpuResetRefused(
            GpuResetErrorCode.GpuNotInContainer, f"GPUs not given to the container: {missing or pod_gpu_uuids}"
        )
    ordered = sorted(container_uuids, key=lambda uuid: host_by_uuid[uuid].bus_id)
    index_by_uuid = {uuid: index for index, uuid in enumerate(ordered)}
    if gpu_indices is None:
        # all of the pod's GPUs: the backend's, also in a `--gpus all` container that sees more of the host
        return PodGpus(ordered, [uuid for uuid in ordered if uuid in pod_gpu_uuids], index_by_uuid)
    if any(index < 0 or index >= len(ordered) for index in gpu_indices):
        raise GpuResetRefused(GpuResetErrorCode.GpuNotInContainer, f"no pod GPU at index {gpu_indices}")
    return PodGpus(ordered, [ordered[index] for index in sorted(set(gpu_indices))], index_by_uuid)


def parse_nvlink_peers(topology: str, host_gpus: list[HostGpu]) -> dict[str, set[str]]:
    """uuid -> the uuids it reaches over NVLink (`NV<n>` cells of `nvidia-smi topo -m`)."""
    uuid_by_index = {gpu.index: gpu.uuid for gpu in host_gpus}
    lines = [_ANSI.sub("", line) for line in topology.splitlines()]
    columns: list[int] = []
    peers: dict[str, set[str]] = {}
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        # the header is the first line led by a GPU label: the runner strips its leading tab
        if not columns and _GPU_LABEL.fullmatch(tokens[0]):
            columns = [int(token[3:]) for token in tokens if _GPU_LABEL.fullmatch(token)]
            continue
        if columns and _GPU_LABEL.fullmatch(tokens[0]):
            row = int(tokens[0][3:])
            for column, cell in zip(columns, tokens[1 : 1 + len(columns)], strict=False):
                if cell.startswith("NV") and row in uuid_by_index and column in uuid_by_index:
                    peers.setdefault(uuid_by_index[row], set()).add(uuid_by_index[column])
    return peers


def parse_compute_apps(csv: str) -> list[tuple[int, str]]:
    apps: list[tuple[int, str]] = []
    for line in csv.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            apps.append((int(parts[0]), parts[1]))
        except ValueError:
            continue
    return apps


def build_reset_command(uuids: list[str], *, container_uuids: list[str], pod_gpu_uuids: list[str]) -> str:
    """The one place `nvidia-smi -r` is spelled: always with `-i`, only on the pod's own GPUs."""
    if not uuids:
        raise ValueError("refusing nvidia-smi -r without -i: no GPU to reset")
    if not set(uuids) <= set(container_uuids) or not set(uuids) <= set(pod_gpu_uuids):
        raise ValueError("refusing to reset a GPU outside the pod")
    return f"nvidia-smi -r -i {shlex.quote(','.join(uuids))}"


def _tail(text: str | None) -> str:
    return (text or "").strip()[-_DETAIL_TAIL_CHARS:]


async def _run(runner: SSHCommandRunner, cmd: str, timeout: int = _QUERY_TIMEOUT_SECONDS):
    return await runner.run(cmd, timeout=timeout, retryable=False)


async def _cgroups(runner: SSHCommandRunner, pids: list[int]) -> dict[int, str]:
    if not pids:
        return {}
    pid_list = " ".join(str(pid) for pid in pids)  # ints: nothing to quote
    result = await _run(
        runner,
        f"for p in {pid_list}; do printf '%s ' \"$p\"; tr '\\n' ' ' < /proc/$p/cgroup 2>/dev/null; echo; done",
    )
    cgroups: dict[int, str] = {}
    for line in (result.stdout or "").splitlines():
        pid, _, cgroup = line.strip().partition(" ")
        if pid.isdigit() and cgroup.strip():
            cgroups[int(pid)] = cgroup
    return cgroups


@dataclass
class _Plan:
    container_id: str
    container_uuids: list[str]
    targets: list[str]
    index_by_uuid: dict[str, int]
    widened: bool


async def _precheck(
    runner: SSHCommandRunner,
    *,
    container_name: str,
    pod_gpu_uuids: list[str],
    gpu_indices: list[int] | None,
    in_tee: bool,
    encrypted_volume_mount: str | None,
    local_volume_path: str | None,
) -> _Plan:
    """Everything that can refuse, before anything on the host changes."""
    if in_tee:
        raise GpuResetRefused(GpuResetErrorCode.ResetNotSupported, "GPU reset is not supported on TEE nodes")

    inspected = await _run(runner, f"{DOCKER} inspect --format '{{{{json .}}}}' {shlex.quote(container_name)}")
    if inspected.exit_code != 0:
        raise GpuResetRefused(GpuResetErrorCode.ContainerNotFound, f"container {container_name} not found")
    try:
        container = json.loads(inspected.stdout)
        container_id: str = container["Id"]
        container_pid = int((container.get("State") or {}).get("Pid") or 0)
        device_requests = (container.get("HostConfig") or {}).get("DeviceRequests")
        mounts = container.get("Mounts") or []
    except (ValueError, KeyError, TypeError) as exc:
        raise GpuResetRefused(GpuResetErrorCode.ContainerNotFound, f"unreadable docker inspect: {exc}") from exc
    if container_pid <= 0:
        raise GpuResetRefused(GpuResetErrorCode.ResetNotSupported, "the container is not running")

    # no host PID view = no way to tell whose a GPU process is: the container's own main process must
    # show up in our /proc with the container's cgroup
    own = await _cgroups(runner, [container_pid])
    if container_id not in own.get(container_pid, ""):
        raise GpuResetRefused(GpuResetErrorCode.ResetNotSupported, "the executor has no host PID view")

    if encrypted_volume_mount and not local_volume_path:
        if any(mount.get("Destination") == encrypted_volume_mount for mount in mounts):
            # the restart has to remount the encrypted volume and cannot guess its plaintext path
            raise GpuResetRefused(
                GpuResetErrorCode.ResetNotSupported, "encrypted volume without its plaintext path"
            )

    host_gpus_result = await _run(runner, HOST_GPUS_COMMAND)
    host_gpus = parse_host_gpus(host_gpus_result.stdout or "")
    if host_gpus_result.exit_code != 0 or not host_gpus:
        raise GpuResetRefused(GpuResetErrorCode.ResetNotSupported, "nvidia-smi listed no GPU on the host")
    host_by_uuid = {gpu.uuid: gpu for gpu in host_gpus}
    gpus = pod_gpus(device_requests, host_gpus, pod_gpu_uuids, gpu_indices)
    ordered, requested, index_by_uuid = gpus.ordered, gpus.requested, gpus.index_by_uuid

    targets = await _widen_targets_to_nvlink_peers(runner, requested, ordered, host_gpus)
    # on the final set: an NVLink peer reset with the requested GPUs must not be MIG either
    if any(host_by_uuid[uuid].mig_enabled for uuid in targets):
        raise GpuResetRefused(GpuResetErrorCode.MigEnabled, "MIG is enabled on a GPU to reset")

    if not set(targets) <= set(pod_gpu_uuids):
        raise GpuResetRefused(GpuResetErrorCode.GpuNotInContainer, "an NVLink peer is not one of the pod's GPUs")

    outside = await _processes_outside(runner, targets, container_id, ignore_cgroup=await _own_cgroup(runner))
    if outside:
        raise GpuResetRefused(
            GpuResetErrorCode.GpuInUseOutsidePod, f"processes outside the pod use the GPUs: {outside}"
        )
    return _Plan(container_id, ordered, targets, index_by_uuid, widened=set(targets) != set(requested))


async def _widen_targets_to_nvlink_peers(
    runner: SSHCommandRunner, requested: list[str], ordered: list[str], host_gpus: list[HostGpu]
) -> list[str]:
    """The requested GPUs plus every NVLink peer they reach (in pod order); refuses a peer outside the pod."""
    nvswitch = await _run(runner, NVSWITCH_COMMAND)
    if (nvswitch.stdout or "").strip():
        # NVSwitch fabrics reset as one: only a pod that owns every host GPU, and then all of them
        if set(ordered) != {gpu.uuid for gpu in host_gpus}:
            raise GpuResetRefused(
                GpuResetErrorCode.NvlinkPeersOutsidePod, "NVSwitch node shared with GPUs outside the pod"
            )
        return list(ordered)
    topology = await _run(runner, TOPOLOGY_COMMAND)
    if topology.exit_code != 0:
        raise GpuResetRefused(GpuResetErrorCode.ResetNotSupported, "could not read the GPU topology")
    peers = parse_nvlink_peers(topology.stdout or "", host_gpus)
    targets = set(requested)
    frontier = list(requested)
    while frontier:
        for peer in peers.get(frontier.pop(), set()):
            if peer not in ordered:
                raise GpuResetRefused(
                    GpuResetErrorCode.NvlinkPeersOutsidePod, "a GPU to reset has NVLink peers outside the pod"
                )
            if peer not in targets:
                targets.add(peer)
                frontier.append(peer)
    return [uuid for uuid in ordered if uuid in targets]


async def _own_cgroup(runner: SSHCommandRunner) -> str | None:
    """The cgroup of the validator's own SSH session (its GPU probes run there); None when it is the root."""
    result = await _run(runner, "tr '\\n' ' ' < /proc/self/cgroup")
    cgroup = (result.stdout or "").strip() if result.exit_code == 0 else ""
    # the root cgroup also holds every unconfined host process, so it proves nothing
    return cgroup if cgroup and cgroup != "0::/" else None


async def _processes_outside(
    runner: SSHCommandRunner, targets: list[str], container_id: str | None, *, ignore_cgroup: str | None = None
) -> list[int]:
    """PIDs computing on the targets that are not the container's (every one when container_id is None).

    PIDs in ignore_cgroup (the validator's own session) do not count."""
    apps = await _run(runner, COMPUTE_APPS_COMMAND)
    if apps.exit_code != 0:
        raise GpuResetRefused(GpuResetErrorCode.ResetNotSupported, "could not list the GPU processes")
    pids = sorted({pid for pid, uuid in parse_compute_apps(apps.stdout or "") if uuid in targets})
    if container_id is None and ignore_cgroup is None:
        return pids
    cgroups = await _cgroups(runner, pids)
    # a PID whose cgroup cannot be read is not proven to be the pod's
    return [
        pid
        for pid in pids
        if (container_id is None or container_id not in cgroups.get(pid, ""))
        and (ignore_cgroup is None or cgroups.get(pid, "").strip() != ignore_cgroup)
    ]


async def reset_pod_gpus(
    runner: SSHCommandRunner,
    *,
    container_name: str,
    pod_gpu_uuids: list[str],
    gpu_indices: list[int] | None,
    start_container: Callable[[], Awaitable[None]],
    in_tee: bool = False,
    encrypted_volume_mount: str | None = None,
    local_volume_path: str | None = None,
    settle_seconds: float = 2.0,
) -> GpuResetReport:
    try:
        plan = await _precheck(
            runner,
            container_name=container_name,
            pod_gpu_uuids=pod_gpu_uuids,
            gpu_indices=gpu_indices,
            in_tee=in_tee,
            encrypted_volume_mount=encrypted_volume_mount,
            local_volume_path=local_volume_path,
        )
    except GpuResetRefused as refused:
        return GpuResetReport(ok=False, error_code=refused.code, detail=refused.detail)

    report = GpuResetReport(ok=False)
    notes: list[str] = []
    if plan.widened:
        notes.append("NVLink peers inside the pod were reset together")
    try:
        stopped = await _run(
            runner, f"{DOCKER} stop -t {STOP_GRACE_SECONDS} {shlex.quote(container_name)}", timeout=60
        )
        if stopped.exit_code != 0:
            raise GpuResetRefused(GpuResetErrorCode.ResetNotSupported, f"docker stop failed: {_tail(stopped.stderr)}")
        report.pod_stopped = True
        # docker stop killed every process of the pod: anything still on the targets is someone else's,
        # except a probe of the validator's own session (as in the precheck)
        own_cgroup = await _own_cgroup(runner)
        pids_still_on_targets: list[int] = []
        for attempt in range(3):
            pids_still_on_targets = await _processes_outside(runner, plan.targets, None, ignore_cgroup=own_cgroup)
            if not pids_still_on_targets:
                break
            if attempt < 2:
                await asyncio.sleep(settle_seconds)
        if pids_still_on_targets:
            raise GpuResetRefused(
                GpuResetErrorCode.GpuInUseOutsidePod, f"processes left on the GPUs: {pids_still_on_targets}"
            )

        reset = await _run(
            runner,
            build_reset_command(plan.targets, container_uuids=plan.container_uuids, pod_gpu_uuids=pod_gpu_uuids),
            timeout=_RESET_TIMEOUT_SECONDS,
        )
        if reset.exit_code == 0:
            report.ok = True
            report.gpus = [GpuResetOutcome(index=plan.index_by_uuid[u], uuid=u, result="reset") for u in plan.targets]
        else:
            # ~45% of the Blackwell fleet refuses `nvidia-smi -r` (gpu_wedge.py); the context cycle at
            # least clears the DAH-2427 latch, on the pod's GPUs only
            report.error_code = GpuResetErrorCode.ResetNotSupported
            notes.append(f"nvidia-smi -r refused: {_tail(reset.stdout or reset.stderr)}")
            for cure in await cure_wedged_gpus(runner, plan.targets):
                report.gpus.append(
                    GpuResetOutcome(
                        index=plan.index_by_uuid[cure.gpu_uuid],
                        uuid=cure.gpu_uuid,
                        result="latch_cleared" if cure.cured else "failed",
                        detail=None if cure.cured else _tail(cure.error or cure.output),
                    )
                )
            # the fallback is the fix when it cures every target; the refused -r stays in the notes
            if report.gpus and all(gpu.result == "latch_cleared" for gpu in report.gpus):
                report.ok, report.error_code = True, None
    except GpuResetRefused as refused:
        report.error_code = refused.code
        notes.append(refused.detail)
    except Exception as exc:  # the start below must still run
        report.error_code = GpuResetErrorCode.ExecutorUnreachable
        notes.append(f"reset failed: {exc}")
    finally:
        try:
            await start_container()
            report.container_restarted = True
        except Exception as exc:
            report.ok = False
            notes.append(f"the container did not start again: {exc}")
    report.detail = "; ".join(notes) or None
    return report


# a scrape this soon after a renter's reset can still miss the cards the driver is bringing back
RECENT_RENTER_GPU_RESET_SECONDS = 5 * 60

# the checks a renter's own reset can trip (the pod is stopped, its GPUs drop out for a moment); a reset of
# the verified job by any other check (a GPU swap, a duplicate node) is still sent during the reset
RESET_CAUSED_CHECK_IDS = frozenset({
    "executor.validate.rented_state",
    "executor.validate.rental_probe",
    "executor.validate.rental_verification",
    "gpu.scrape.machine_spec",
    "gpu.validate.rented_gpu_drop",
})
# these two also clear the job for a card swap or a mixed node: only their missing-card reasons are the reset's
RESET_CAUSED_REASON_CODES = {
    "gpu.validate.fingerprint": frozenset({"GPU_MISSING"}),
    "gpu.validate.model": frozenset({"GPU_DETAILS_MISMATCH"}),
}


def reset_caused(check_id: str | None, reason_code: str | None, evidence: ResetVerifiedJobEvidence) -> bool:
    """True when a verified-job reset by this check and reason is one a renter's GPU reset can cause."""
    if check_id in RESET_CAUSED_CHECK_IDS:
        return True
    if reason_code in RESET_CAUSED_REASON_CODES.get(check_id, ()):
        return True
    # warn mode reports a missing card as GPU_UUID_CHANGED, with the hard verdict in the evidence
    return check_id == "gpu.validate.fingerprint" and evidence.anchor_hard_would_be == "GPU_MISSING"


async def renter_gpu_resets(redis, miner_hotkey: str, executor_id: str) -> list[dict]:
    """The node's readable gpu_reset marks; a Redis error (or anything but a list) is none, so the checks run."""
    try:
        marks = await redis.gpu_reset_marks(miner_hotkey, executor_id)
    except Exception:
        return []
    return marks if isinstance(marks, list) else []


async def renter_gpu_reset_in_progress(redis, miner_hotkey: str, executor_id: str, pod_id: str | None = None) -> bool:
    """True on a readable gpu_reset mark on the node, or on that pod when pod_id is given."""
    marks = await renter_gpu_resets(redis, miner_hotkey, executor_id)
    return any(pod_id is None or mark.get("pod_id") == pod_id for mark in marks)


async def renter_gpu_reset_recent_mark(redis, miner_hotkey: str, executor_id: str) -> RecentGpuReset | None:
    """The node's mark of a renter reset that ended within RECENT_RENTER_GPU_RESET_SECONDS; a Redis error (or
    anything but a mark) is none, so the checks run."""
    try:
        mark = await redis.gpu_reset_recent(miner_hotkey, executor_id)
    except Exception:
        return None
    return mark if isinstance(mark, RecentGpuReset) else None


async def renter_gpu_reset_recent(redis, miner_hotkey: str, executor_id: str, pod_id: str | None = None) -> bool:
    """True on a readable recent-reset mark on the node, or on that pod when pod_id is given."""
    mark = await renter_gpu_reset_recent_mark(redis, miner_hotkey, executor_id)
    return mark is not None and pod_id in (None, mark.pod_id)
