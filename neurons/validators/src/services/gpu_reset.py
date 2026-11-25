"""DAH-4658: a renter resets the GPUs of their own running pod.

The validator does all of it over its SSH session into the executor, which runs privileged with
`pid: host`, every GPU and docker.sock, so `docker` and `nvidia-smi` here are the HOST's view.
Nothing is ever read from inside the pod (no `docker exec`): the renter controls that output.

    prechecks (change nothing) -> docker stop -t 10 -> no process left on the targets
        -> nvidia-smi -r -i <uuids> (or the DAH-2427 latch cure when the driver refuses) -> start

On an NVSwitch node whose Fabric Manager runs (many HGX nodes show no /dev/nvidia-nvswitch*), the driver
refuses `-r` while Fabric Manager holds the GPUs: a pod that owns every host GPU gets Fabric Manager (and
the NVLink subnet manager on HGX Blackwell, when it runs) stopped around the reset and always started
again. A pod that owns every host GPU also gets the host's other GPU holders stopped the same way (DCGM, a
dcgm-exporter container, IMEX, nvidia-persistenced, and any other host service still holding /dev/nvidia*): the
driver refuses `-r` "In use" while any of them holds a GPU. All of it runs in one host systemd unit that
outlives the SSH session (a reset that hangs past its timeout is left
to that unit, which starts the fabric when the reset ends; the latch cure and the container start wait until the
unit is proven gone);
any other pod is refused (the provider resets). A pod that owns only some host GPUs is refused before its stop while
a process outside it holds one of them open (DAH-6741): nothing on the host is stopped for it, so the driver would refuse.

Safety invariants, each pinned by a test:
- the reset set is a subset of BOTH the container's own DeviceRequests GPUs (from the docker
  daemon) and the GPUs the backend holds for the pod;
- `nvidia-smi -r` never runs without `-i`, except on a pod that owns every host GPU of an NVSwitch fabric whose
  Fabric Manager the unit stops (the whole baseboard resets, NVSwitches included);
- a refused `-r` never reports ok: the latch cure alone reads `latch_cleared`, never `reset`;
- every host GPU holder stopped for the reset is started again, also when the reset fails;
- the container is always started again once it was stopped (or a stop of it failed), and a reset
  of a pod whose container is not running starts it first.
"""

from __future__ import annotations

import asyncio
import json
import logging
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

logger = logging.getLogger(__name__)

DOCKER = "/usr/bin/docker"
HOST_GPUS_COMMAND = (
    "nvidia-smi --query-gpu=index,uuid,pci.bus_id,mig.mode.current --format=csv,noheader"
)
COMPUTE_APPS_COMMAND = "nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader"
TOPOLOGY_COMMAND = "nvidia-smi topo -m"
NVSWITCH_COMMAND = (
    'for p in /dev/nvidia-nvswitch[0-9]*; do [ -e "$p" ] && printf "%s\\n" "$p"; done; true'
)
# the executor runs privileged with pid: host; PID 1's mount namespace reaches the host's systemd
HOST_SYSTEMCTL = "nsenter -t 1 -m -- systemctl"
# the NVSwitch fabric's host services, in stop order (started again in reverse): Fabric Manager, then the
# NVLink subnet manager it uses on HGX Blackwell
FABRIC_SERVICES = ("nvidia-fabricmanager", "nvidia-nvlsm")
# every host service the unit may stop, in stop order: GPU clients first, the fabric, then nvidia-persistenced
# (it keeps the driver loaded, so it goes last and comes back first)
HOST_GPU_SERVICES = (
    "nvidia-dcgm-exporter",
    "dcgm-exporter",
    "nvidia-dcgm",
    "dcgm",
    "nvidia-imex",
    *FABRIC_SERVICES,
    "nvidia-persistenced",
)
# id, AutoRemove, image of every running container: a --rm container is deleted by its stop, so it is never stopped
DOCKER_PS_COMMAND = (
    f"{DOCKER} inspect --format '{{{{.Id}}}} {{{{.HostConfig.AutoRemove}}}} {{{{.Config.Image}}}}' "
    f"$({DOCKER} ps -q --no-trunc)"
)
# the host PIDs holding any NVIDIA device open, matched by device number (every major the nvidia drivers registered
# in /proc/devices), not by path, since any node with those numbers opens the device; exits 3 when the scan is not
# whole (an open file whose device could not be read). Its stat can block on a stalled filesystem (a descriptor on a
# hung FUSE mount), so it runs before anything is stopped, never inside the host unit.
# (a subshell: put a space after a "$(" that wraps it, or the shell reads "$((" as arithmetic)
GPU_HOLDER_PIDS_SCRIPT = (
    "( m=$(awk '$2 ~ /^nvidia/ { printf \" %x \", $1 }' /proc/devices) && [ -n \"$m\" ] || exit 3; "
    "d=$(mktemp -d) || exit 3; trap 'rm -rf \"$d\"' EXIT; "
    "LC_ALL=C find /proc/[0-9]*/fd -type l -exec stat -L -c '%t %n' {} + >\"$d/o\" 2>\"$d/e\"; r=$?; "
    "awk -v m=\"$m\" 'index(m, \" \" $1 \" \") { sub(\"/fd/.*\", \"\", $2); sub(\".*/\", \"\", $2); print $2 }' "
    "\"$d/o\" | sort -u; "
    "[ \"$r\" = 0 ] || { [ -s \"$d/e\" ] && ! grep -qv 'No such file or directory' \"$d/e\"; } || exit 3 )"
)
# the process names still holding a /dev/nvidia* node (host PID view): what blocked a refused `nvidia-smi -r`. Only
# readlink on /proc, which never touches the file's filesystem, so it cannot block the unit before its restore; text
# for the report only, nothing is decided on it. No ${...} in a unit script: systemd expands it in the command line
# before /bin/sh runs (to "" if unknown)
GPU_HOLDERS_SCRIPT = (
    "for p in $(find /proc/[0-9]*/fd -lname '/dev/nvidia*' 2>/dev/null | sed 's#/fd/.*##; s#.*/##' | sort -u); "
    "do cat /proc/$p/comm; done 2>/dev/null | sort -u | tr '\\n' ' '"
)


def device_holders_command(bus_ids: list[str]) -> str:
    """A "gpu <bus>" line for each GPU whose holders were read, then "<pid> <name>" of every process holding one of
    these GPUs' devices open (host PID view), but the scanning shell and its ancestors (the SSH session).

    The driver refuses `nvidia-smi -r` while any process has the GPU open: a graphics (Xorg) or monitoring one
    too, which the compute-app listing never shows. An open device is matched by its device number (the NVIDIA
    major 195 and the minor from the driver's own per-bus file), not by its path, since any device node with those
    numbers opens the GPU. A GPU without a "gpu" line was not read whole (no minor, or an open file whose device
    could not be read), so its holders are unknown. The one error that leaves the reading whole is a process or
    file that went away mid-scan (No such file or directory)."""
    buses = " ".join(shlex.quote(_proc_bus(bus)) for bus in bus_ids)
    return (
        "for c in find stat awk grep; do command -v $c >/dev/null || exit 3; done; "
        "d=$(mktemp -d) || exit 3; trap 'rm -rf \"$d\"' EXIT; "
        's=" "; p=$$; while [ -n "$p" ] && [ "$p" -gt 1 ]; do '
        's="$s$p "; p=$(sed -n \'s/^PPid:[[:space:]]*//p\' /proc/$p/status 2>/dev/null); done; '
        # the device number (hex major, minor) of what every open file of every process is
        'LC_ALL=C find /proc/[0-9]*/fd -type l -exec stat -L -c "%t %T %n" {} + >"$d/o" 2>"$d/e"; r=$?; '
        'w=0; if [ "$r" = 0 ] || { [ -s "$d/e" ] && ! grep -qv "No such file or directory" "$d/e"; }; then w=1; fi; '
        f"for b in {buses}; do "
        "m=$(sed -n 's/^Device Minor:[[:space:]]*//p' /proc/driver/nvidia/gpus/$b/information 2>/dev/null); "
        'case "$m" in ""|*[!0-9]*) continue;; esac; '
        "awk -v m=\"$(printf %x \"$m\")\" "
        "'$1 == \"c3\" && $2 == m { sub(\"/fd/.*\", \"\", $3); sub(\".*/\", \"\", $3); print $3 }' \"$d/o\"; "
        'if [ "$w" = 1 ]; then echo "gpu $b"; fi; '
        "done | sort -u | while read k v; do "
        'if [ "$k" = gpu ]; then echo "gpu $v"; else case "$s" in *" $k "*) ;; '
        "*) printf '%s %s\\n' \"$k\" \"$(cat /proc/$k/comm 2>/dev/null)\";; esac; fi; done"
    )


def _proc_bus(bus_id: str) -> str:
    """nvidia-smi spells the PCI domain with 8 digits, /proc/driver/nvidia/gpus/ with 4."""
    return bus_id[-12:]


# the host services of the processes holding a GPU device, but the reset unit itself (exits 3 when the holder scan is
# not whole, or a live holder's cgroup cannot be read). The enclosing service of a nested cgroup counts
# (system.slice/gpu.slice/x.service, system.slice/x.service/worker). Run in the host's mount and cgroup namespaces,
# so /proc/<pid>/cgroup reads as the host's
GPU_HOLDER_UNITS_SCRIPT = (
    f"hp=$( {GPU_HOLDER_PIDS_SCRIPT}) || exit 3; hu=$(for p in $hp; do "
    "c=$(cat /proc/$p/cgroup 2>/dev/null) || { [ -e /proc/$p ] && exit 3; continue; }; "
    "printf '%s\\n' \"$c\" | sed -n 's#.*:/system\\.slice/\\([^/]*\\.slice/\\)*\\([^/]*\\.service\\)\\(/.*\\)\\{0,1\\}$#\\2#p'; "
    "done) || exit 3; [ -z \"$hu\" ] || printf '%s\\n' \"$hu\" | grep -vx 'lium-gpu-reset-fabric.service' | sort -u"
)
# a fabric reset runs as one host systemd unit, so a dropped SSH session cannot leave Fabric Manager down
# named, so a reset that outlives its SSH timeout can be stopped (one per host: it owns every host GPU)
FABRIC_RESET_UNIT = "lium-gpu-reset-fabric"
HOST_GPU_HOLDER_UNITS_COMMAND = f"nsenter -t 1 -m -C -- /bin/sh -c {shlex.quote(GPU_HOLDER_UNITS_SCRIPT)}"
_HOLDER_SCAN_TIMEOUT_SECONDS = 60
# the unit's second look after its stops (GPU_HOLDER_UNITS_SCRIPT: device numbers, so a GPU opened through any path,
# a bind mount included, is found) runs as its own host unit, outside the reset unit's cgroup, into a fresh file
# under this prefix. The reset unit waits for that file at most LATE_SCAN_SECONDS and never on the scan's processes:
# a scan stuck on an open file (uninterruptible) cannot keep it from its starts or from ending. One scan unit per
# host (its fixed name), bounded by RuntimeMaxSec: while a stuck one is still there, a new one does not start and
# the reset is skipped
LATE_SCAN_UNIT = "lium-gpu-reset-holder-scan"
LATE_SCAN_FILE = f"/run/{LATE_SCAN_UNIT}"
LATE_SCAN_SECONDS = 30
# the launch is --no-block (only queued, never a wait on systemd's start job) and the call itself has this deadline: a
# busy manager cannot keep the stopped services down; a launch that does not return in time skips the reset
LATE_SCAN_LAUNCH_SECONDS = 10
# every host service a reset unit stopped, one per line, written before its stop: the validator's restore checks
# them with the ones it asked for, and removes the file only once all of them are up again. Never truncated by a
# unit, so a reset whose restore did not run (a dropped session) leaves them to the next restore
STOPPED_UNITS_FILE = f"/run/{FABRIC_RESET_UNIT}.stopped"
STOPPED_UNITS_READ_COMMAND = (
    f"nsenter -t 1 -m -- /bin/sh -c {shlex.quote(f'[ ! -e {STOPPED_UNITS_FILE} ] || cat {STOPPED_UNITS_FILE}')}"
)
# marks a service the unit found late (after the validator's scan) in STOPPED_UNITS_FILE: it may need a listed one
# (Fabric Manager, nvidia-persistenced) up first, so it starts after them, never by its place in the stop order
LATE_MARK = "+"
STOPPED_UNITS_CLEAR_COMMAND = f"nsenter -t 1 -m -- rm -f {STOPPED_UNITS_FILE}"
HOST_SYSTEMD_RUN = f"nsenter -t 1 -m -- systemd-run --quiet --wait --pipe --collect --unit={FABRIC_RESET_UNIT}"
# starts an exporter that outlived its kill once it exits, after the reset unit has ended. systemd owns the
# retries: any failed wait or start (a Docker daemon restart included) fails the unit and it runs again; only an
# observed exit plus a start that is still up ("true false") a few seconds later, or a removed container, ends it.
# One unit per container ID (the unit name ends in it), so two exporters each get their own
EXPORTER_RESTART_UNIT = "lium-gpu-reset-exporter-restart"
EXPORTER_RESTART_SCRIPT = (
    # every docker call has a deadline (a stalled dockerd): a timeout fails the unit, so systemd runs it again
    'timeout -k 5 300 docker wait "$0" >/dev/null || '
    '{ timeout -k 5 30 docker inspect "$0" 2>&1 >/dev/null | grep -q "No such" && exit 0; exit 1; }; '
    'timeout -k 5 30 docker start "$0" >/dev/null && sleep 5 && '
    "[ \"$(timeout -k 5 30 docker inspect -f '{{.State.Running}} {{.State.Restarting}}' \"$0\")\" = 'true false' ]"
)
EXPORTER_RESTART_RUN = (
    f'systemd-run --quiet --collect --unit={EXPORTER_RESTART_UNIT}-"$dying" -p Restart=on-failure -p RestartSec=10 '
    f'/bin/sh -c {shlex.quote(EXPORTER_RESTART_SCRIPT)} "$dying" '
    # a unit already there for this container (an earlier reset's) starts it already
    f'|| [ "$(systemctl show -p LoadState --value {EXPORTER_RESTART_UNIT}-"$dying")" = loaded ]'
)
# "no-systemd" only when the host proves it has no systemd (not booted with it, no systemd-run): any other answer,
# a failed probe included, keeps the reset in the host unit
HOST_SYSTEMD_PROBE = "nsenter -t 1 -m -- sh -c " + shlex.quote(
    "[ -d /run/systemd/system ] || command -v systemd-run >/dev/null 2>&1 || echo no-systemd"
)
# processes left in the reset unit's cgroup (v2, or v1's systemd hierarchy); 0 only for an unloaded unit or an
# existing, readable, empty cgroup.procs, or a host without systemctl (no unit can have run there). Neither file
# there, or a failed read, exits 1: unknown, never 0
FABRIC_RESET_UNIT_PROCS = "nsenter -t 1 -m -- sh -c " + shlex.quote(
    "command -v systemctl >/dev/null 2>&1 || { echo 0; exit 0; }; "
    f"cg=$(systemctl show -p ControlGroup --value {FABRIC_RESET_UNIT}) || exit 1; "
    '[ -z "$cg" ] && { echo 0; exit 0; }; '
    'for f in /sys/fs/cgroup"$cg"/cgroup.procs /sys/fs/cgroup/systemd"$cg"/cgroup.procs; do '
    '[ -f "$f" ] || continue; n=$(wc -l < "$f") || exit 1; echo "$n"; exit 0; done; exit 1'
)
_FABRIC_STOP_FAILED_EXIT = 97
STOP_GRACE_SECONDS = 10
# process deadline (plus a 2s kill) on each docker CLI call of the host unit beyond its own grace or wait: a
# stalled dockerd cannot hold the unit forever. Worst case of the unit's docker calls with one holder container:
# stop 22 + wait 92 + inspect 12 + kill 12 + wait 32 + start 12 = 182s; each further holder container adds 34s
_DOCKER_CALL_SECONDS = 10
# "<running> <restarting>" of a holder container: only "true false" is up
CONTAINER_STATE_COMMAND = f"{DOCKER} inspect -f '{{{{.State.Running}}}} {{{{.State.Restarting}}}}'"
# a `docker stop` that got no exit event leaves the container dying: wait this long for it to exit before the start
EXIT_WAIT_SECONDS = 90
# after the kill of a container that outlasted EXIT_WAIT_SECONDS
KILL_WAIT_SECONDS = 30
_QUERY_TIMEOUT_SECONDS = 30
_RESET_TIMEOUT_SECONDS = 120
_DETAIL_TAIL_CHARS = 300
_LOG_OUTPUT_CHARS = 4000
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_GPU_LABEL = re.compile(r"GPU\d+")


class GpuResetRefused(Exception):
    def __init__(self, code: GpuResetErrorCode, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        # True once the precheck saw the pod's container running: a refusal from then on left it running as before
        self.container_running: bool | None = None


class ContainerNotRunning(GpuResetRefused):
    def __init__(self):
        super().__init__(GpuResetErrorCode.ResetNotSupported, "the container is not running")


@dataclass
class GpuResetReport:
    ok: bool
    error_code: GpuResetErrorCode | None = None
    container_restarted: bool = False
    gpus: list[GpuResetOutcome] = field(default_factory=list)
    detail: str | None = None
    # docker stop ran: the pod's GPUs may drop out until the driver brings them back
    pod_stopped: bool = False
    # the wire's pod_stopped (DAH-6741). True: the container was down during this reset, by its docker stop (even
    # one that then failed) or found stopped by an earlier one. False: it was seen running and never touched.
    # None: not known (refused before its state was read, an earlier reset unit still running, a dropped session)
    stopped_during_reset: bool | None = None


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


def build_reset_command(
    uuids: list[str], *, container_uuids: list[str], pod_gpu_uuids: list[str], whole_fabric: bool = False
) -> str:
    """The one place `nvidia-smi -r` is spelled: with `-i`, only on the pod's own GPUs.

    whole_fabric (the pod owns every host GPU and the host unit stops Fabric Manager): no `-i`, so the NVSwitches
    reset with the GPUs, as NVIDIA's Fabric Manager guide resets an HGX H100-or-later baseboard."""
    if not uuids:
        raise ValueError("refusing nvidia-smi -r without -i: no GPU to reset")
    if not set(uuids) <= set(container_uuids) or not set(uuids) <= set(pod_gpu_uuids):
        raise ValueError("refusing to reset a GPU outside the pod")
    if whole_fabric:
        if not set(uuids) == set(container_uuids) == set(pod_gpu_uuids):
            raise ValueError("refusing nvidia-smi -r without -i: the pod does not reset every GPU")
        return "nvidia-smi -r"
    return f"nvidia-smi -r -i {shlex.quote(','.join(uuids))}"


def resets_whole_fabric(owns_host: bool, host_services: list[str]) -> bool:
    """Fabric Manager stopped around the reset loses the fabric state: every GPU and NVSwitch of it must reset."""
    return owns_host and any(unit in FABRIC_SERVICES for unit in host_services)


def _tail(text: str | None) -> str:
    return (text or "").strip()[-_DETAIL_TAIL_CHARS:]


def _head(text: str | None) -> str:
    """The start of a refusal: nvidia-smi names the GPU and the cause first, then a long generic advice."""
    return " ".join((text or "").split())[:_DETAIL_TAIL_CHARS]


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
    # host services and containers to stop around the reset (only for a pod that owns every host GPU)
    host_services: list[str] = field(default_factory=list)
    holder_containers: list[str] = field(default_factory=list)
    # the reset runs in the host unit, which also stops host GPU holders the validator does not list (a pod that owns
    # every host GPU, unless its host is proven without systemd with nothing to stop)
    owns_host: bool = False
    # the pod owns every host GPU (container and backend alike), in the host unit or not
    whole_host: bool = False
    # host PCI bus id of each host GPU
    bus_by_uuid: dict[str, str] = field(default_factory=dict)


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
    # before the not-running check: a stopped pod is started first, and its remount needs the path (review 4447076002)
    if encrypted_volume_mount and not local_volume_path:
        if any(mount.get("Destination") == encrypted_volume_mount for mount in mounts):
            # the restart has to remount the encrypted volume and cannot guess its plaintext path
            raise GpuResetRefused(
                GpuResetErrorCode.ResetNotSupported, "encrypted volume without its plaintext path"
            )

    if container_pid <= 0:
        raise ContainerNotRunning()
    try:
        return await _plan_running_pod(runner, container_id, container_pid, device_requests, pod_gpu_uuids, gpu_indices)
    except GpuResetRefused as refused:
        refused.container_running = True
        raise


async def _plan_running_pod(
    runner: SSHCommandRunner,
    container_id: str,
    container_pid: int,
    device_requests: list[dict] | None,
    pod_gpu_uuids: list[str],
    gpu_indices: list[int] | None,
) -> _Plan:
    """The precheck of a running container: everything that can still refuse, before anything changes."""
    # no host PID view = no way to tell whose a GPU process is: the container's own main process must
    # show up in our /proc with the container's cgroup
    own = await _cgroups(runner, [container_pid])
    if container_id not in own.get(container_pid, ""):
        raise GpuResetRefused(GpuResetErrorCode.ResetNotSupported, "the executor has no host PID view")

    host_gpus_result = await _run(runner, HOST_GPUS_COMMAND)
    host_gpus = parse_host_gpus(host_gpus_result.stdout or "")
    # a dropped channel says nothing about the host's GPUs; only a hung or failed nvidia-smi does
    if host_gpus_result.error_type not in (None, "timeout"):
        raise GpuResetRefused(GpuResetErrorCode.ExecutorUnreachable, "the GPU listing lost its SSH channel")
    if host_gpus_result.exit_code != 0 or not host_gpus:
        raise GpuResetRefused(
            GpuResetErrorCode.GpusLostOnHost,
            "The GPUs stopped responding on the host. The provider has to reboot it before they can be used again.",
        )
    host_by_uuid = {gpu.uuid: gpu for gpu in host_gpus}
    gpus = pod_gpus(device_requests, host_gpus, pod_gpu_uuids, gpu_indices)
    ordered, requested, index_by_uuid = gpus.ordered, gpus.requested, gpus.index_by_uuid

    targets, fabric_services = await _widen_targets_to_nvlink_peers(runner, requested, ordered, host_gpus)
    # on the final set: an NVLink peer reset with the requested GPUs must not be MIG either
    if any(host_by_uuid[uuid].mig_enabled for uuid in targets):
        raise GpuResetRefused(GpuResetErrorCode.MigEnabled, "MIG is enabled on a GPU to reset")

    if not set(targets) <= set(pod_gpu_uuids):
        raise GpuResetRefused(GpuResetErrorCode.GpuNotInContainer, "an NVLink peer is not one of the pod's GPUs")

    outside = await _processes_outside(runner, targets, container_id, ignore_cgroup=await _own_cgroup(runner))
    if outside:
        raise GpuResetRefused(
            GpuResetErrorCode.GpuInUseOutsidePod,
            # an unreadable cgroup counts here too, so the process may be the renter's own
            "processes the node could not place in the pod (not in the pod's container, or their cgroup could not "
            f"be read) use the GPUs (host PIDs): {outside}; stop them only if you can tell they are yours from "
            "inside the pod, then retry",
        )
    host_services: list[str] = list(fabric_services)
    holder_containers: list[str] = []
    # both ownership sources: a `--gpus all` container may see more GPUs than the backend gave the pod
    whole_host = owns_host = set(ordered) == set(host_by_uuid) == set(pod_gpu_uuids)
    if owns_host:
        holders = await plan_host_holders(runner, fabric_services, container_id)
        host_services, holder_containers, owns_host = holders.host_services, holders.holder_containers, holders.owns_host
    return _Plan(
        container_id,
        ordered,
        targets,
        index_by_uuid,
        widened=set(targets) != set(requested),
        host_services=host_services,
        holder_containers=holder_containers,
        owns_host=owns_host,
        whole_host=whole_host,
        bus_by_uuid={gpu.uuid: gpu.bus_id for gpu in host_gpus},
    )


@dataclass(frozen=True)
class HostHolders:
    host_services: list[str]
    holder_containers: list[str]
    owns_host: bool


async def plan_host_holders(
    runner: SSHCommandRunner, fabric_services: list[str], container_id: str | None
) -> HostHolders:
    """For a pod that owns every host GPU: the host services and holder containers to stop around its reset, and
    whether the reset runs in the host unit (False only on a host proven without systemd with nothing to stop)."""
    host_services = [
        unit
        for unit in HOST_GPU_SERVICES
        if unit in fabric_services or (unit not in FABRIC_SERVICES and await _service_active(runner, unit))
    ]
    # any other host service holding a GPU, read now, before anything is stopped: a scan that blocks (a stalled
    # filesystem) or is not whole refuses the reset with nothing changed, instead of holding the host unit between
    # its stops and its starts. They stop first and start last (they may need the listed ones)
    scan = await _run(runner, HOST_GPU_HOLDER_UNITS_COMMAND, timeout=_HOLDER_SCAN_TIMEOUT_SECONDS)
    if scan.exit_code == 3:
        raise GpuResetRefused(
            GpuResetErrorCode.ResetNotSupported,
            "could not read which host processes hold the GPUs; nothing was stopped",
        )
    if scan.exit_code != 0:
        raise GpuResetRefused(
            GpuResetErrorCode.ExecutorUnreachable,
            "the scan of which host services hold the GPUs got no answer; nothing was stopped",
        )
    listed = {f"{unit}.service" for unit in HOST_GPU_SERVICES}
    discovered = [unit for unit in (scan.stdout or "").split() if unit.endswith(".service") and unit not in listed]
    host_services = [*discovered, *host_services]
    holder_containers: list[str] = []
    running = await _run(runner, DOCKER_PS_COMMAND)
    for line in (running.stdout or "").splitlines() if running.exit_code == 0 else []:
        cid, auto_remove, image = (line.strip().split(" ", 2) + ["", ""])[:3]
        if (
            "dcgm-exporter" in image
            and auto_remove == "false"
            and cid != container_id
            and re.fullmatch(r"[0-9a-f]{12,64}", cid)
        ):
            holder_containers.append(cid)
    if not host_services and not holder_containers:
        # a host proven without systemd and with nothing to stop resets directly, as before the host unit
        probe = await _run(runner, HOST_SYSTEMD_PROBE)
        return HostHolders([], [], not (probe.exit_code == 0 and (probe.stdout or "").strip() == "no-systemd"))
    return HostHolders(host_services, holder_containers, True)


async def _service_active(runner: SSHCommandRunner, unit: str) -> bool:
    result = await _run(runner, f"{HOST_SYSTEMCTL} is-active {shlex.quote(unit)}")
    return (result.stdout or "").strip() == "active"


async def _widen_targets_to_nvlink_peers(
    runner: SSHCommandRunner, requested: list[str], ordered: list[str], host_gpus: list[HostGpu]
) -> tuple[list[str], list[str]]:
    """The requested GPUs plus every NVLink peer they reach (in pod order); refuses a peer outside the pod.

    Also returns the fabric services running on the host (they have to be stopped for the reset)."""
    nvswitch = await _run(runner, NVSWITCH_COMMAND)
    fabric_services = [unit for unit in FABRIC_SERVICES if await _service_active(runner, unit)]
    if (nvswitch.stdout or "").strip() or fabric_services:
        # NVSwitch fabrics reset as one: only a pod that owns every host GPU, and then all of them
        if set(ordered) != {gpu.uuid for gpu in host_gpus}:
            raise GpuResetRefused(
                GpuResetErrorCode.NvlinkPeersOutsidePod,
                "NVSwitch node shared with other pods: its GPUs reset only together, so the provider must "
                "reset them on the host",
            )
        return list(ordered), fabric_services
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
    return [uuid for uuid in ordered if uuid in targets], []


def build_fabric_reset_command(units: list[str], reset_command: str, containers: list[str] = ()) -> str:
    """Stop the GPU holders (containers, then the given services: the ones plan_host_holders found holding a GPU
    first), reset, start them all again (the given services last stopped first, then the containers) as one host
    systemd unit. After its stops it looks once more (GPU_HOLDER_UNITS_SCRIPT) for a host service that opened a
    GPU after the validator's scan: it stops it too and starts it again after the others. Every service it stops is
    listed in STOPPED_UNITS_FILE first, for the validator's restore check. A refused reset prints the processes still
    holding a GPU. The second look follows open files (device numbers) in its own host unit (LATE_SCAN_UNIT) with a
    deadline, so a stalled one can keep neither this unit from its starts nor from ending; a look that is not whole,
    not done in time or not started skips the reset.

    The unit outlives the SSH session, and SIGPIPE is ignored so a closed pipe cannot kill it before the start."""
    fail = f'|| {{ echo "could not stop $u" >&2; rc={_FABRIC_STOP_FAILED_EXIT}; break; }}'
    # listed before the stop, so the restore check knows it even if this unit never gets to its start
    stop_one = f'echo "$u" >>{STOPPED_UNITS_FILE} && systemctl stop "$u" {fail}'
    # a late one is listed with LATE_MARK: the restore starts it after the others, as this unit does
    stop_late = f'echo "{LATE_MARK}$u" >>{STOPPED_UNITS_FILE} && systemctl stop "$u" {fail}'
    script = "trap '' PIPE; rc=0; dying=''; late=''; "
    if containers:
        ids = " ".join(shlex.quote(c) for c in containers)
        script += (
            f'for u in {ids}; do timeout -k 2 {STOP_GRACE_SECONDS + _DOCKER_CALL_SECONDS} '
            f'docker stop -t {STOP_GRACE_SECONDS} "$u" >/dev/null || '
            f'{{ echo "could not stop $u" >&2; rc={_FABRIC_STOP_FAILED_EXIT}; dying="$u"; break; }}; done; '
        )
    if units:
        stop = " ".join(shlex.quote(u) for u in units)
        script += f'[ $rc -ne 0 ] || for u in {stop}; do {stop_one}; done; '
    # a host service that opened a GPU after the validator's scan: stopped too (and started again after the others),
    # or the driver refuses the reset with the pod already stopped
    planned = " ".join(u if u.endswith(".service") else f"{u}.service" for u in units)
    # a scan that is not whole, did not end in time, or whose launch failed or timed out: the holders are unknown, so
    # no reset (the starts still run)
    scan = f'( {GPU_HOLDER_UNITS_SCRIPT} ) >"$0"; echo $? >"$0.rc"'
    script += (
        f"[ $rc -ne 0 ] || {{ if f=$(mktemp {LATE_SCAN_FILE}.XXXXXXXX) && timeout -k 2 {LATE_SCAN_LAUNCH_SECONDS} "
        "systemd-run --quiet --collect --no-block "
        f"--unit={LATE_SCAN_UNIT} -p RuntimeMaxSec={LATE_SCAN_SECONDS} /bin/sh -c {shlex.quote(scan)} \"$f\" "
        ">/dev/null 2>&1; then "
        f'i=0; while [ ! -s "$f.rc" ] && [ $i -lt {LATE_SCAN_SECONDS} ]; do sleep 1; i=$((i+1)); done; fi; '
        'if [ -n "$f" ] && [ "$(cat "$f.rc" 2>/dev/null)" = 0 ]; then for u in $(cat "$f"); do '
        f'case " {planned} " in *" $u "*) ;; *) late="$late $u"; {stop_late};; esac; done; '
        'else echo "could not read which host services hold a GPU after the stops" >&2; '
        f'rc={_FABRIC_STOP_FAILED_EXIT}; fi; rm -f "$f" "$f.rc"; }}; '
    )
    script += (
        f"[ $rc -ne 0 ] || {{ {reset_command}; rc=$?; "
        f'[ $rc -eq 0 ] || echo "GPU holders left: $({GPU_HOLDERS_SCRIPT})"; }}; '
    )
    if units:
        # reversed: a found holder (first in the list) may need Fabric Manager or nvidia-persistenced to start; a
        # start that fails here is checked and retried by the validator's restore of every one of them
        start = " ".join(shlex.quote(u) for u in reversed(units))
        script += f'for u in {start}; do systemctl start "$u"; done; '
    script += 'for u in $late; do systemctl start "$u"; done; '
    if containers:
        # a failed stop may leave the container dying, and `docker start` is a no-op on a running one: wait for
        # the exit (a removed container ends the wait), then kill one still there and wait once more. Bounded, so
        # the unit always ends inside the validator's wait and the renter pod starts. One that outlives its kill
        # (stuck in the kernel) still reads running, so the restore check cannot see it: its start goes to a host
        # unit that outlives this one and starts it once it exits, and the report names it (restart pending)
        script += (
            f'[ -z "$dying" ] || until timeout -k 2 {EXIT_WAIT_SECONDS} docker wait "$dying" >/dev/null; do '
            f'! timeout -k 2 {_DOCKER_CALL_SECONDS} docker inspect "$dying" 2>&1 >/dev/null | grep -q "No such" || break; '
            'echo "$dying did not exit, killing it" >&2; '
            f'timeout -k 2 {_DOCKER_CALL_SECONDS} docker kill "$dying" >/dev/null 2>&1; '
            f'timeout -k 2 {KILL_WAIT_SECONDS} docker wait "$dying" >/dev/null 2>&1 || {{ '
            f"{{ {EXPORTER_RESTART_RUN}; }} "
            '>/dev/null 2>&1 && echo "$dying did not exit after a kill: restart pending, it starts once it exits" >&2 '
            '|| echo "could not start $dying: it did not exit after a kill" >&2; }; break; done; '
        )
        ids = " ".join(shlex.quote(c) for c in reversed(containers))
        # a start that times out reads as not running again in the validator's restore check
        script += f'for u in {ids}; do timeout -k 2 {_DOCKER_CALL_SECONDS} docker start "$u" >/dev/null; done; '
    script += "exit $rc"
    return f"{HOST_SYSTEMD_RUN} /bin/sh -c {shlex.quote(script)}"


async def _reset_unit_gone(runner: SSHCommandRunner) -> bool:
    """True once the host unit of a reset that did not return is proven gone.

    Never stopped: a stop kills the unit's shell, so nothing would start the fabric when the reset ends."""
    try:
        state = await _run(runner, f"{HOST_SYSTEMCTL} is-active {FABRIC_RESET_UNIT}")
        procs = await _run(runner, FABRIC_RESET_UNIT_PROCS)
    except Exception as exc:
        logger.error("could not read the hung GPU reset unit: %s", exc)
        return False
    # a collected unit reads "inactive"; anything else (running, stopping, no answer) may still be resetting.
    # "failed" alone is no proof: systemd also fails a unit whose processes survive its final SIGKILL (a reset in
    # D state on a wedged GPU), so the unit's cgroup must also be empty
    left = (procs.stdout or "").strip() if procs.exit_code == 0 else "unknown"
    if (state.stdout or "").strip() in ("inactive", "failed") and left == "0":
        return True
    logger.error("the hung GPU reset unit is still running: %r, %s processes", (state.stdout or "").strip(), left)
    return False


async def _reset_unit_pending(runner: SSHCommandRunner) -> bool:
    """False only once the earlier reset unit is proven gone (empty cgroup, or no unit can exist on the host).

    Fails closed: an unreadable count may hide a reset that still owns the GPUs. Only a missing nsenter (127) is
    proof, since the unit is started through nsenter too."""
    try:
        procs = await _run(runner, FABRIC_RESET_UNIT_PROCS)
    except Exception as exc:
        logger.error("could not read the GPU reset unit: %s", exc)
        return True
    if procs.exit_code == 127:
        return False
    left = (procs.stdout or "").strip() if procs.exit_code == 0 else ""
    if left == "0":
        return False
    logger.error("an earlier GPU reset unit may still run: exit %s, %r processes", procs.exit_code, left)
    return True


def _unit_name(unit: str) -> str:
    """nvidia-fabricmanager and nvidia-fabricmanager.service are one unit"""
    return unit if unit.endswith(".service") else f"{unit}.service"


async def _restore_fabric_services(
    runner: SSHCommandRunner, units: list[str], containers: list[str] = (), settle_seconds: float = 0
) -> str | None:
    """Starts every stopped host service and holder container that is not up after the reset; the problem left,
    else None. A container counts as up only when it still runs (not exited, not restarting) after settle_seconds."""
    problems: list[str] = []
    listed: list[str] = []
    late: list[str] = []
    try:
        # the ones a unit stopped beyond these: found late by this reset's unit, or left by a reset whose restore
        # did not run. No nsenter (127): no unit can have run on this host, as in _reset_unit_pending. The listed
        # ones start in the unit's order: the planned ones last stopped first, then the late ones (LATE_MARK), which
        # may need a planned one (Fabric Manager) up and were stopped after it
        stopped = await _run(runner, STOPPED_UNITS_READ_COMMAND)
        if stopped.exit_code not in (0, 127):
            problems.append(f"could not read which host services the reset unit stopped: {_tail(stopped.stderr)}")
        elif stopped.exit_code == 0:
            listed = (stopped.stdout or "").split()
            planned = [u for u in listed if not u.startswith(LATE_MARK)]
            units = [*dict.fromkeys(u for u in planned if u not in units), *units]
            known = {_unit_name(u) for u in units}
            late = [*dict.fromkeys(u[len(LATE_MARK):] for u in listed if u.startswith(LATE_MARK))]
            late = [u for u in late if _unit_name(u) not in known]
    except Exception as exc:
        problems.append(f"could not read which host services the reset unit stopped: {exc}")
    for cid in reversed(containers):
        try:
            started = await _run(runner, f"{DOCKER} start {shlex.quote(cid)}", timeout=_RESET_TIMEOUT_SECONDS)
            if started.exit_code != 0:
                problems.append(f"container {cid[:12]} is not running again after the reset: {_tail(started.stderr)}")
                continue
            await asyncio.sleep(settle_seconds)
            state = await _run(runner, f"{CONTAINER_STATE_COMMAND} {shlex.quote(cid)}")
            answer = (state.stdout or "").strip() if state.exit_code == 0 else ""
            if answer != "true false":
                problems.append(
                    f"container {cid[:12]} is not running again after the reset: {answer or 'state unreadable'}"
                )
        except Exception as exc:
            problems.append(f"container {cid[:12]} state unknown after the reset: {exc}")
    for unit in [*reversed(units), *late]:
        try:
            if await _service_active(runner, unit):
                continue
            start = f"{HOST_SYSTEMCTL} start {shlex.quote(unit)}"  # an escaped name (\x2d) keeps its backslash
            started = await _run(runner, start, timeout=_RESET_TIMEOUT_SECONDS)
            if started.exit_code != 0 or not await _service_active(runner, unit):
                problems.append(f"{unit} is not active again after the reset: {_tail(started.stderr)}")
        except Exception as exc:
            problems.append(f"{unit} state unknown after the reset: {exc}")
    if problems:
        logger.error("host GPU services down after a GPU reset: %s", "; ".join(problems))
    elif listed:
        try:
            # every one is up again; a list left behind only makes a later restore check them once more
            await _run(runner, STOPPED_UNITS_CLEAR_COMMAND)
        except Exception as exc:
            logger.warning("could not clear the reset unit's list of stopped services: %s", exc)
    return "; ".join(problems) or None


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


async def _holders_outside(runner: SSHCommandRunner, plan: _Plan, settle_seconds: float) -> list[str]:
    """"<name> (pid <pid>)" of the processes outside the pod holding a target GPU open, in both of two reads
    settle_seconds apart: a passing nvidia-smi holds it for a moment only. Raises when a target GPU's holders
    could not be read: an unread GPU is not a free one."""
    buses = {uuid: _proc_bus(plan.bus_by_uuid.get(uuid, "")) for uuid in plan.targets}
    held: dict[int, str] | None = None
    for attempt in range(2):
        if attempt:
            await asyncio.sleep(settle_seconds)
        listed = await _run(runner, device_holders_command([bus for bus in buses.values() if bus]))
        if listed.exit_code < 0:
            raise GpuResetRefused(
                GpuResetErrorCode.ExecutorUnreachable, "the scan of who holds the GPUs on the host got no answer"
            )
        read: set[str] = set()
        seen: dict[int, str] = {}
        for line in (listed.stdout or "").splitlines() if listed.exit_code == 0 else []:
            key, _, value = line.strip().partition(" ")
            if key == "gpu":
                read.add(value.strip())
            elif key.isdigit():
                seen[int(key)] = value.strip() or "unknown"
        unread = [plan.index_by_uuid[uuid] for uuid, bus in buses.items() if not bus or bus not in read]
        if unread:
            raise GpuResetRefused(
                GpuResetErrorCode.ResetNotSupported,
                f"could not read which host processes hold GPU {', '.join(map(str, sorted(unread)))}. "
                "The GPUs were not reset and the pod was not stopped",
            )
        cgroups = await _cgroups(runner, list(seen))
        # only the pod's own processes are excluded: an unreadable cgroup is not proven to be the pod's
        outside = {pid: name for pid, name in seen.items() if plan.container_id not in cgroups.get(pid, "")}
        held = outside if held is None else {pid: name for pid, name in outside.items() if pid in held}
        if not held:
            return []
    return [f"{name} (pid {pid})" for pid, name in sorted(held.items())]


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
    async def precheck() -> _Plan:
        return await _precheck(
            runner,
            container_name=container_name,
            pod_gpu_uuids=pod_gpu_uuids,
            gpu_indices=gpu_indices,
            in_tee=in_tee,
            encrypted_volume_mount=encrypted_volume_mount,
            local_volume_path=local_volume_path,
        )

    report = GpuResetReport(ok=False)
    notes: list[str] = []
    fabric_manager_problem: str | None = None
    reset_still_running = False
    if await _reset_unit_pending(runner):
        # an earlier reset that outlived its SSH timeout still holds the GPUs and the fabric: before the stopped
        # container is started and before the fabric check (Fabric Manager is down until that unit ends)
        return GpuResetReport(
            ok=False,
            error_code=GpuResetErrorCode.OperationInProgress,
            detail="the GPU reset is still running on the host",
        )
    # host services an earlier reset unit stopped and whose restore never ran (a dropped session): started again
    # before anything here can return early, and nothing is stopped while one of them stays down
    left_down = await _restore_fabric_services(runner, [])
    if left_down:
        return GpuResetReport(
            ok=False,
            error_code=GpuResetErrorCode.ResetNotSupported,
            detail=f"a host GPU service an earlier reset stopped is down, nothing was stopped: {left_down}",
        )
    try:
        plan = await precheck()
    except ContainerNotRunning as refused:
        # the backend sends a reset only for a RUNNING pod: a stopped container (an earlier reset whose stop
        # timed out) is started first, so the renter's next reset is the recovery, not a refusal
        try:
            await start_container()
        except Exception as exc:
            return GpuResetReport(
                ok=False,
                error_code=refused.code,
                detail=f"{refused.detail}; it did not start: {exc}",
                stopped_during_reset=True,
            )
        report.container_restarted = True
        report.stopped_during_reset = True
        notes.append("the container was not running: started it again")
        try:
            plan = await precheck()
        except GpuResetRefused as again:
            report.error_code = again.code
            report.detail = "; ".join([*notes, again.detail])
            return report
    except GpuResetRefused as refused:
        return GpuResetReport(
            ok=False,
            error_code=refused.code,
            detail=refused.detail,
            stopped_during_reset=False if refused.container_running else None,
        )
    if report.stopped_during_reset is None:
        report.stopped_during_reset = False  # the precheck saw it running

    if plan.widened:
        notes.append("NVLink peers inside the pod were reset together")
    if not plan.whole_host:
        # nothing on the host is stopped for a pod that owns only some GPUs, and the driver refuses the reset while
        # any process outside the pod holds a target GPU open: refuse before the pod is stopped
        try:
            held = await _holders_outside(runner, plan, settle_seconds)
        except GpuResetRefused as refused:
            report.error_code = refused.code
            report.detail = "; ".join([*notes, refused.detail])
            return report
        if held:
            report.error_code = GpuResetErrorCode.GpuInUseOutsidePod
            report.detail = "; ".join(
                [
                    *notes,
                    # an unreadable cgroup counts as outside, so the holder may be the renter's own process
                    "a process the node could not place in the pod (not in the pod's container, or its cgroup could "
                    f"not be read) holds the GPU open on the host (host PID): {', '.join(held)}. The GPUs were not "
                    "reset and the pod was not stopped; stop that process only if you can tell it is yours from "
                    "inside the pod, then retry, otherwise give the host PID to the provider or support",
                ]
            )
            return report
    stop_failed = False
    report.stopped_during_reset = True
    try:
        stopped = await _run(
            runner, f"{DOCKER} stop -t {STOP_GRACE_SECONDS} {shlex.quote(container_name)}", timeout=60
        )
        if stopped.exit_code != 0:
            stop_failed = True
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

        command = build_reset_command(
            plan.targets,
            container_uuids=plan.container_uuids,
            pod_gpu_uuids=pod_gpu_uuids,
            whole_fabric=resets_whole_fabric(plan.owns_host, plan.host_services),
        )
        if not plan.owns_host:
            reset = await _run(runner, command, timeout=_RESET_TIMEOUT_SECONDS)
        else:
            # the driver refuses -r while Fabric Manager or another host client holds the GPUs; the pod owns them all
            reset = None
            try:
                reset = await _run(
                    runner,
                    build_fabric_reset_command(plan.host_services, command, plan.holder_containers),
                    timeout=_RESET_TIMEOUT_SECONDS + 120,
                )
            finally:
                # a reset that did not return (timeout, dropped session) may still hold the GPUs and the fabric
                if (reset is None or reset.exit_code < 0) and not await _reset_unit_gone(runner):
                    reset_still_running = True
                    fabric_manager_problem = "the GPU reset is still running on the host"
                else:
                    # before the latch cure and the container start: CUDA on an NVSwitch node needs Fabric Manager
                    fabric_manager_problem = await _restore_fabric_services(
                        runner, plan.host_services, plan.holder_containers, settle_seconds
                    )
                if fabric_manager_problem:
                    notes.append(fabric_manager_problem)
        if reset is not None and reset.exit_code != 0:
            # the report keeps only a short tail; the whole host output is what explains a refusal, logged before a
            # holder that did not come back ends the reset early
            logger.warning(
                "nvidia-smi -r refused for %s: exit %s, host services %s, holder containers %s, stdout %r, stderr %r",
                container_name,
                reset.exit_code,
                plan.host_services,
                plan.holder_containers,
                (reset.stdout or "")[-_LOG_OUTPUT_CHARS:],
                (reset.stderr or "")[-_LOG_OUTPUT_CHARS:],
            )
        if plan.owns_host:
            if fabric_manager_problem:
                raise GpuResetRefused(
                    GpuResetErrorCode.ResetNotSupported, "no latch cure while a host GPU service is down"
                )
            if reset.exit_code == _FABRIC_STOP_FAILED_EXIT:
                raise GpuResetRefused(
                    GpuResetErrorCode.ResetNotSupported, _tail(reset.stderr) or "a host GPU holder did not stop"
                )
        if reset.exit_code == 0:
            report.ok = True
            report.gpus = [GpuResetOutcome(index=plan.index_by_uuid[u], uuid=u, result="reset") for u in plan.targets]
        else:
            # ~45% of the Blackwell fleet refuses `nvidia-smi -r` (gpu_wedge.py); the context cycle at
            # least clears the DAH-2427 latch, on the pod's GPUs only
            report.error_code = GpuResetErrorCode.ResetNotSupported
            notes.append(f"nvidia-smi -r refused, the GPUs were not reset: {_head(reset.stdout or reset.stderr)}")
            for cure in await cure_wedged_gpus(runner, plan.targets):
                report.gpus.append(
                    GpuResetOutcome(
                        index=plan.index_by_uuid[cure.gpu_uuid],
                        uuid=cure.gpu_uuid,
                        result="latch_cleared" if cure.cured else "failed",
                        detail=None if cure.cured else _tail(cure.error or cure.output),
                    )
                )
            # the latch cure is no reset: the report stays not ok (each GPU reads latch_cleared, never reset)
            if report.gpus and all(gpu.result == "latch_cleared" for gpu in report.gpus):
                notes.append("only the latch clear ran")
    except GpuResetRefused as refused:
        report.error_code = refused.code
        notes.append(refused.detail)
    except Exception as exc:  # the start below must still run
        report.error_code = GpuResetErrorCode.ExecutorUnreachable
        notes.append(f"reset failed: {exc}")
    finally:
        if reset_still_running:
            # never start the pod on GPUs a reset may still be resetting; the host unit starts the fabric itself
            notes.append("the container was not started again while the GPU reset still runs")
        else:
            # a stop without an exit event leaves the container dying: `docker start` is then a no-op,
            # the remount's exec fails on setns and the pod exits seconds later. Wait for the exit, then start; a
            # start that fails for any other reason gets one more try after the same wait.
            for attempt in range(2):
                try:
                    if stop_failed or attempt:
                        await _wait_exited(runner, container_name)
                    await start_container()
                    report.container_restarted = True
                    break
                except Exception as exc:
                    if attempt:
                        # the flag reports the last start: an earlier recovery start does not count (review 4447075986)
                        report.ok, report.container_restarted = False, False
                        # no error_code: the backend reads ok false with every GPU reset as the container not restarting
                        notes.append(f"the container did not start again: {exc}")
    if fabric_manager_problem:
        report.ok = False
        report.error_code = report.error_code or GpuResetErrorCode.ResetNotSupported
    report.detail = "; ".join(notes) or None
    return report


async def _wait_exited(runner: SSHCommandRunner, container_name: str) -> None:
    """Until the container has exited, at most EXIT_WAIT_SECONDS (`docker wait` answers at once for a stopped one)."""
    try:
        await _run(
            runner,
            f"timeout {EXIT_WAIT_SECONDS} {DOCKER} wait {shlex.quote(container_name)}",
            timeout=EXIT_WAIT_SECONDS + 30,
        )
    except Exception:  # best effort: the start runs over its own session either way
        pass


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
