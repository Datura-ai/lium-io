from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shlex
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from services.rental_docker_sdk import RENTAL_NETWORK_NAME

from core.config import settings
from core.utils import _m, get_extra_info

from ..host_disk import (
    DEVICE_MARKER,
    DISK_PRESSURE_MAX_DEFERRALS,
    DOCKER_ROOT_DEVICES,
    SNAPSHOT_SECONDS,
    SYNCFS_POLL_SECONDS,
    DiskPressureReading,
    disk_pressure_probe,
    parse_disk_pressure,
)
from ..messages import RentedThroughputMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from ..runner import SSHCommandResult
from .container_download_speed import (
    _FETCH_SCRIPT,
    _TIMEOUT_EXITS,
    NODE_SIDE_ERRORS,
    ContainerDownloadSpeedCheck,
    download_speed_command,
    parse_download_speed,
)
from .rented_gpu_drop import POD_STATUS_RUNNING
from .task_time import task_seconds_left

logger = logging.getLogger(__name__)

DISK_PROBE_BYTES = 256 * 1024 * 1024
DISK_TIMEOUT_SECONDS = 60
# the 2 s busy window, the disk-pressure reading (2 s snapshots, syncfs poll cap), the disk probe (direct, then
# buffered), both fetches and a second container start
COMMAND_TIMEOUT_SECONDS = 270 + SNAPSHOT_SECONDS + SYNCFS_POLL_SECONDS
PROBE_SKIPPED_MARKER = "lium_dp_skipped"
# a reading needs its full command timeout plus the checks after it; started later, the task deadline cuts it off
# and the node's whole cycle is dropped without a result (316223c6, 30 Oct 17:49Z)
MIN_TASK_SECONDS_LEFT = COMMAND_TIMEOUT_SECONDS + 30
# due readings skipped for lack of cycle time, in a row, that count as a failed reading (the provider controls how long
# the earlier checks take, so endless late cycles must not keep the score)
LATE_SKIPS_BEFORE_VERDICT = 3
# cycle time kept after a Redis save, to finish the cycle
SAVE_RESERVE_SECONDS = 5
RETRY_SECONDS = 30 * 60
READINGS_BEFORE_VERDICT = 2
# a failed reading older than this (its retry never ran, e.g. the node was unrented in between) does not count
# "in a row" with the next one: RETRY_SECONDS plus LATE_SKIPS_BEFORE_VERDICT late cycles fit well inside it
STREAK_GAP_SECONDS = 3 * RETRY_SECONDS
BUSY_WINDOW_SECONDS = 2
DISK_MARKER = "lium_disk"
SNAP_MARKER = "lium_tp"
_REDIS_PREFIX = "rented_throughput"
_REDIS_TTL_SECONDS = 7 * 24 * 3600
# The rented pods' own traffic while the probe measured a side (their network namespace's interfaces around the fetch
# container, their container cgroup's io.stat around the disk container) at or above these is renter load on what the
# probe measured. Only the pods' own counters count, so traffic the host makes anywhere else cannot excuse a slow node.
# The host can still attach its own process to a pod's cgroup or network namespace, so the processes there are checked
# per pod: only the container's own = its IO is renter load, no verdict (owner decision: a renter keeping the disk or
# link busy is never SLOW); a foreign one = its IO does not count; not readable = inconclusive.
DURING_NET_MBPS = 5.0
DURING_DISK_MB_PER_S = 2.0
DURING_DISK_IOPS = 20.0
# A failing side is also inconclusive when its pod counters are incomplete (any rented pod not readable both when that
# side began and when it ended: stopped or restarted mid-probe, a counter that went backwards, never readable, or no
# snapshot): the renter's IO is unknown. Inconclusive = no strike and no pass however many in a row (owner decision: IO
# that cannot be attributed never fails a node), and the node is probed again after RETRY_SECONDS.
# executor uuid -> a reading Redis refused to record; keeps the cadence (no probe per cycle) until a save succeeds
_UNSAVED: dict[str, _State] = {}

# The executor shares the host PID namespace but not its network (bridge, `pid: host`), so the NIC counters are
# init's (/proc/1/net/dev), read only when /proc/1 is the host's; else no rx/tx and a slow network side is inconclusive.
# The devices under the docker root (the probe's writable layer) are host_disk.DOCKER_ROOT_DEVICES. Then bytes
# received and sent on the physical NICs and sectors
# moved (and requests completed) per whole disk (/proc/diskstats is not namespaced), twice BUSY_WINDOW_SECONDS apart
# before the probe, then again after the disk container and after the fetch container (throughput_command).
_COUNTER_SNAPSHOTS = (
    # Only physical NICs (a sysfs `device` link) are counted: disjoint, so a bond, bridge, VLAN or OVS port can neither
    # repeat nor hide an uplink. Without a host sysfs (this shell's own is the container's), names and bond members.
    'ph=$(for i in /proc/1/root/sys/class/net/*; do [ -e $i/device ] && [ "${i##*/}" != lo ] && '
    'printf " %s" "${i##*/}"; done); '
    "sl=$(for i in /proc/1/root/sys/class/net/*; do [ -e $i/master ] && m=$(readlink $i/master) && "
    'printf " %s:%s" "${i##*/}" "${m##*/}"; done; '
    'for f in /proc/1/net/bonding/*; do [ -f "$f" ] && '
    'awk -v b="${f##*/}" \'/^Slave Interface:/{printf " %s:%s", $3, b}\' "$f"; done 2>/dev/null); '
    # Provenance: own PID is the container's (init $p): its host-side ancestry reaches the init (a shared PID namespace
    # alone is not enough); 0 = yes, 1 = not (a host process attached to its cgroup or netns), 2 = gone mid-read or the
    # init itself gone (the container restarted, so its new processes cannot descend from the init we hold). prov
    # prints 0 when any listed PID is not the container's, nothing when one could not be read (never own), else 1.
    'own() { [ -d /proc/$p ] || return 2; q=$1; while [ "$q" != "$p" ]; do '
    "q=$(awk '/^PPid:/{print $2}' /proc/$q/status 2>/dev/null); [ -n \"$q\" ] || return 2; "
    '[ "$q" -gt 0 ] || return 1; done; return 0; }; '
    "prov() { r=1; while read i; do own $i; case $? in 1) r=0; break;; 2) r=;; esac; done; echo $r; }; "
    # Per rented pod, keyed by its position in $pp (pid/container id, set by throughput_command): rx:tx of its own network namespace (not
    # the host's), bytes:requests of its container cgroup (systemd or cgroupfs driver), then the provenance of the
    # processes in that cgroup (its subtree) and in that network namespace; a field left empty is not readable.
    # Nothing when /proc/1 is not the host's, where a docker pid means nothing here.
    "pods() { [ /proc/1/root/ -ef / ] && return; k=0; for x in $pp; do k=$((k+1)); p=${x%%/*}; c=${x#*/}; "
    "n=$([ /proc/$p/ns/net -ef /proc/1/ns/net ] || sed 's/:/ /' /proc/$p/net/dev 2>/dev/null | "
    'awk \'NR>2 && $1!="lo"{r+=$2; t+=$10; k++} END{if (k) printf "%.0f:%.0f", r, t}\'); d=; g=; '
    "for f in /proc/1/root/sys/fs/cgroup/system.slice/docker-$c.scope/io.stat /proc/1/root/sys/fs/cgroup/docker/$c/io.stat; "
    'do [ -r $f ] && d=$(awk \'{for (i=2; i<=NF; i++) {split($i, a, "="); '
    'if (a[1]=="rbytes" || a[1]=="wbytes") b+=a[2]; if (a[1]=="rios" || a[1]=="wios") n+=a[2]}} '
    'END{printf "%.0f:%.0f", b, n}\' $f) && g=${f%/io.stat} && break; done; '
    'cg=; [ -r "$g/cgroup.procs" ] && cg=$(find "$g" -name cgroup.procs -exec cat {} + 2>/dev/null | prov); '
    "nt=; [ -e /proc/$p/ns/net ] && ! [ /proc/$p/ns/net -ef /proc/1/ns/net ] && nt=$(for q in /proc/[0-9]*; do "
    "[ $q/ns/net -ef /proc/$p/ns/net ] && echo ${q#/proc/}; done 2>/dev/null | prov); "
    'printf "%s/%s:%s:%s:%s:%s," "$k" "$p" "${n:-:}" "${d:-:}" "$cg" "$nt"; done; }; '
    f'snap() {{ echo "{SNAP_MARKER} t=$(cut -d" " -f1 /proc/uptime) '
    '$([ /proc/1/root/ -ef / ] || sed \'s/:/ /\' /proc/1/net/dev 2>/dev/null | awk -v sl="$sl" -v ph="$ph" '
    '\'function keep(x) {return x!="lo" && x!~/[.@]/ && '
    "x!~/^(docker|veth|br-|virbr|cni|flannel|cali|tun|wg|lxc|vxlan|kube)/} "
    'BEGIN{n=split(sl, a, " "); for (i=1; i<=n; i++) {split(a[i], p, ":"); mst[p[1]]=p[2]}; '
    'np=split(ph, q, " "); for (i=1; i<=np; i++) phy[q[i]]=1} '
    "NR>2 && (np ? ($1 in phy) : keep($1) && !(($1 in mst) && keep(mst[$1]))){r+=$2; t+=$10; k++} "
    'END{if (k) printf "rx=%.0f tx=%.0f", r, t}\') '
    "io=$(awk '$3~/^(sd[a-z]+|nvme[0-9]+n[0-9]+|vd[a-z]+|xvd[a-z]+|md[0-9]+)$/{printf \"%s:%.0f:%.0f,\", $3, $6+$10, $4+$8}' "
    '/proc/diskstats) pod=$(pods)"; }; '
    f"snap; sleep {BUSY_WINDOW_SECONDS}; snap; "
)
_SNAPSHOT = DOCKER_ROOT_DEVICES + _COUNTER_SNAPSHOTS
# In the probe container, whose writable layer sits on the disk a renter's container writes to. O_DIRECT where the
# filesystem takes it, else buffered with the fsync timed in; a timed-out write is the slow reading itself.
_DISK_SCRIPT = (
    "up() { cut -d' ' -f1 /proc/uptime; }; "
    "dprobe() { rm -f /root/lium_disk; s=$(up); "
    f"timeout {DISK_TIMEOUT_SECONDS} dd if=/dev/zero of=/root/lium_disk bs=1M "
    f"count={DISK_PROBE_BYTES // (1024 * 1024)} $1 conv=fsync 2>/dev/null; rc=$?; e=$(up); "
    "n=$(wc -c 2>/dev/null </root/lium_disk || echo 0); rm -f /root/lium_disk; "
    f'echo "{DISK_MARKER} bytes=$n start=$s end=$e rc=$rc"; }}; '
    "dprobe oflag=direct; case $rc in 0|124|137|143) ;; *) dprobe '';; esac; "
)


def throughput_command(
    sysbox: bool, containers: list[str], *, read_pressure: bool = True, skip_under_pressure: bool = False
) -> str:
    """The disk-pressure reading (read_pressure), the busy snapshot, then one probe container per side with the
    counters after each: what the renter's pods moved while that side was measured. The reading goes first so its
    seconds stay out of the disk side's interval and do not dilute the renter's rate there. With skip_under_pressure both
    containers are left out when the reading shows pressure (the shell's `dp`), since their start would stall."""
    names = " ".join(shlex.quote(c) for c in containers)
    pod_pids = (
        f"pp=$(for c in {names}; do /usr/bin/docker inspect -f '{{{{.State.Pid}}}}/{{{{.Id}}}}' \"$c\" 2>/dev/null || echo 0/-; done); "
    )
    probe = (
        download_speed_command(sysbox, _DISK_SCRIPT)
        + "; snap; "
        + download_speed_command(sysbox, _FETCH_SCRIPT)
        + "; snap"
    )
    if not read_pressure:
        return pod_pids + _SNAPSHOT + probe
    if skip_under_pressure:
        probe = f'if [ "$dp" = 1 ]; then echo "{PROBE_SKIPPED_MARKER}"; else {probe}; fi'
    return pod_pids + DOCKER_ROOT_DEVICES + disk_pressure_probe() + _COUNTER_SNAPSHOTS + probe


@dataclass(frozen=True)
class DiskReading:
    bytes: int = 0
    seconds: float | None = None
    mb_per_s: float | None = None
    # None = measured; else why not: no_output, write_failed
    error: str | None = None


def parse_disk(stdout: str) -> DiskReading:
    """The last dd line: the buffered retry when O_DIRECT was refused."""
    lines = [ln for ln in (stdout or "").splitlines() if ln.startswith(f"{DISK_MARKER} ")]
    if not lines:
        return DiskReading(error="no_output")
    tokens = dict(t.partition("=")[::2] for t in lines[-1][len(DISK_MARKER) :].split())
    try:
        got = int(tokens.get("bytes") or 0)
        seconds = float(tokens["end"]) - float(tokens["start"])
        rc = int(tokens.get("rc") or -1)
    except (KeyError, ValueError):
        return DiskReading(error="no_output")
    if rc == 0 or rc in _TIMEOUT_EXITS:
        return DiskReading(got, seconds, got / max(seconds, 0.01) / 1e6)
    return DiskReading(got, seconds, error="write_failed")


# (uptime, (rx, tx) or None, {disk: (sectors, requests)}, {pod pid: ((rx, tx) or None, (bytes, requests) or None,
# its cgroup's processes all the container's: True/False/None unreadable, the same for its network namespace)})
_Snap = tuple[float, Any, dict[str, tuple[int, int]], dict[str, tuple[Any, Any, Any, Any]]]
_OWN = {"1": True, "0": False, "": None}


def _pair(a: str, b: str) -> tuple[int, int] | None:
    return (int(a), int(b)) if a and b else None


def _both(c0: Any, c1: Any) -> bool:
    """A pod counter readable at both ends and never lower at the end: a lower one was reset (the container restarted
    and its cgroup was recreated), so what moved in between is unknown."""
    return c0 is not None and c1 is not None and c1[0] >= c0[0] and c1[1] >= c0[1]


def _parse_snaps(stdout: str) -> tuple[list[_Snap], set[str]]:
    snaps: list[_Snap] = []
    devices: set[str] = set()
    for line in (stdout or "").splitlines():
        if line.startswith(f"{DEVICE_MARKER} name="):
            devices.add(line.partition("=")[2].strip())
        elif line.startswith(f"{SNAP_MARKER} "):
            tokens = dict(t.partition("=")[::2] for t in line[len(SNAP_MARKER) :].split())
            try:
                io = {}
                for e in tokens.get("io", "").split(","):
                    if e:
                        d, _, rest = e.partition(":")
                        sectors, _, ios = rest.partition(":")
                        io[d] = (int(sectors), int(ios or 0))
                net = (int(tokens["rx"]), int(tokens["tx"])) if "rx" in tokens else None
                pods = {}
                for e in tokens.get("pod", "").split(","):
                    if e:
                        pid, rx, tx, moved, ios, *own = e.split(":")
                        cg, nt = own or (
                            "",
                            "",
                        )  # the older five-field format: provenance not readable
                        pods[pid] = (_pair(rx, tx), _pair(moved, ios), _OWN[cg], _OWN[nt])
                snaps.append((float(tokens["t"]), net, io, pods))
            except (KeyError, ValueError):
                return [], devices
    return snaps, devices


def _moved(a: _Snap, b: _Snap, devices: set[str]) -> tuple[float | None, float | None]:
    """Other traffic between two snapshots: (Mbps on the host NICs, the busier direction; MB/s on the busiest disk
    under the docker root). None = not measured."""
    (t0, net0, io0, _), (t1, net1, io1, _) = a, b
    dt = max(t1 - t0, 0.01)
    # docker root devices unresolved (btrfs/zfs, no host view): another disk's traffic says nothing, so not measured
    relevant = [(io1[d], io0[d]) for d in devices if d in io1 and d in io0]
    disk = (
        max(max(0, s1 - s0) for (s1, _), (s0, _) in relevant) * 512 / dt / 1e6 if relevant else None
    )
    if net0 is None or net1 is None:
        return None, disk
    return max(net1[0] - net0[0], net1[1] - net0[1], 0) * 8 / dt / 1e6, disk


def _own(a: _Snap, b: _Snap, pid: str, side: int) -> bool | None:
    """Provenance of one pod's processes for that side (cgroup for disk, netns for network) in both snapshots."""
    marks = [snap[3].get(pid, (None,) * 4)[3 - side] for snap in (a, b)]
    return False if False in marks else (True if marks == [True, True] else None)


def _pods_moved(
    a: _Snap, b: _Snap, side: int, only: tuple[bool | None, ...] = (True, False, None)
) -> tuple[float, float] | None:
    """What the rented pods themselves moved between two snapshots, summed per second: (received, sent)
    for side 0 (their network namespaces' bytes), (bytes, requests) per second for side 1 (their cgroups' disk
    IO). Only pods readable in both snapshots count: one that restarted (a new pid, or a counter reset) or whose
    counter went unreadable adds nothing but does not void the others. None = not measured: no pod readable in both."""
    (t0, *_, pods0), (t1, *_, pods1) = a, b
    both = [
        (c[side], pods1[pid][side])
        for pid, c in pods0.items()
        if pid in pods1 and _both(c[side], pods1[pid][side]) and _own(a, b, pid, side) in only
    ]
    if not both:
        return None
    dt = max(t1 - t0, 0.01)
    return (
        sum(c1[0] - c0[0] for c0, c1 in both) / dt,
        sum(c1[1] - c0[1] for c0, c1 in both) / dt,
    )


def pods_incomplete(stdout: str, pods: int) -> tuple[bool, bool]:
    """(network, disk): one of the first `pods` rented pods (the running ones, listed first) was not readable both when
    that side's measurement began and when it ended (stopped or restarted mid-probe, a counter reset, never readable,
    or a snapshot missing), so what the renter moved while that side was measured is not fully known."""
    snaps, _ = _parse_snaps(stdout)
    if len(snaps) < 4:
        return pods > 0, pods > 0

    def incomplete(a: _Snap, b: _Snap, side: int) -> bool:
        first = list(a[3].items())[:pods]
        both = [pid for pid, c in first if _both(c[side], b[3].get(pid, (None, None))[side])]
        return len(both) < pods

    return incomplete(snaps[2], snaps[3], 0), incomplete(snaps[1], snaps[2], 1)


def pods_provenance(stdout: str) -> tuple[str | None, str | None]:
    """(network, disk): whose processes the rented pods' network namespaces and cgroups held while that side was
    measured. "own" = only the containers' own, "foreign" = a process that is not the container's (the host attached
    it to make IO count as the renter's), None = not readable for every pod in both snapshots."""
    snaps, _ = _parse_snaps(stdout)
    if len(snaps) < 4:
        return None, None

    def provenance(a: _Snap, b: _Snap, index: int) -> str | None:
        own = [c[index] for snap in (a, b) for c in snap[3].values()]
        if any(o is False for o in own):
            return "foreign"
        return "own" if own and all(o is True for o in own) else None

    return provenance(snaps[2], snaps[3], 3), provenance(snaps[1], snaps[2], 2)


def parse_busy(stdout: str) -> tuple[float | None, float | None]:
    """Other traffic in the window before the probe: (Mbps on the host NICs, the busier direction; MB/s on the busiest
    disk under the docker root). None = not measured; a slow side is then inconclusive."""
    snaps, devices = _parse_snaps(stdout)
    if len(snaps) < 2:
        return None, None
    return _moved(snaps[0], snaps[1], devices)


def parse_during(
    stdout: str, only: tuple[bool | None, ...] = (True, False, None)
) -> tuple[float | None, tuple[float | None, float | None]]:
    """The rented pods' own traffic while each side was measured: (network Mbps, the busier direction, during the
    fetch container; (disk MB/s, disk IO/s) during the disk container). The probe and anything else on the host is
    outside those counters. `only`: the pods whose provenance for that side is one of these. None = not measured."""
    snaps, _ = _parse_snaps(stdout)
    if len(snaps) < 4:
        return None, (None, None)
    net = _pods_moved(snaps[2], snaps[3], 0, only)
    disk = _pods_moved(snaps[1], snaps[2], 1, only)
    return (
        None if net is None else max(net) * 8 / 1e6,
        (None, None) if disk is None else (disk[0] / 1e6, disk[1]),
    )


def phase_offset_seconds(uuid: str) -> float:
    fraction = int(hashlib.sha256(uuid.encode()).hexdigest()[:12], 16) / 16**12
    return fraction * settings.RENTED_THROUGHPUT_INTERVAL_HOURS * 3600


def _excuse(side: str, renter_io: bool, maybe_renter_io: bool, unsure: bool) -> str:
    """A failing side: no verdict when pods holding only their own processes carried renter load while it was
    measured (owner decision: a renter keeping the disk or link busy is never SLOW); inconclusive when pods of unknown
    provenance could have, or when the measurement is unsure; else it fails. Pods with a foreign process never excuse."""
    if side != "fail":
        return side
    if renter_io:
        return "none"
    return "inconclusive" if maybe_renter_io or unsure else "fail"


def _step(failures: int, side: str) -> int:
    """One streak per side: a side that measures and passes ends its own streak whatever the other side says; no
    verdict and inconclusive keep it."""
    return failures + 1 if side == "fail" else (0 if side == "pass" else failures)


def slow_detail(state: _State) -> str:
    """Which confirmed side failed and the values behind it, e.g. "network: pypi 1.5 Mbps measured, 10 Mbps
    required". Read by the score warning and the provider portal, so no "|" (the warning separator)."""
    reading = state.reading
    parts = []
    if state.net_failures >= READINGS_BEFORE_VERDICT:
        need = reading.get("min_mbps")
        slow = [
            f"{e['endpoint']} {e['mbps']:.1f} Mbps" if e.get("mbps") is not None else f"{e['endpoint']} {e.get('error')}"
            for e in reading.get("endpoints") or []
            if (e.get("mbps") is not None and need is not None and e["mbps"] < need) or e.get("error") in NODE_SIDE_ERRORS
        ]
        parts.append(f"network: {', '.join(slow)} measured, {need:g} Mbps required" if slow and need else "network")
    if state.disk_failures >= READINGS_BEFORE_VERDICT:
        mb, need = (reading.get("disk") or {}).get("mb_per_s"), reading.get("disk_min_mb_per_s")
        parts.append(f"disk: {mb:.1f} MB/s measured, {need:g} MB/s required" if mb is not None and need else "disk")
    return "; ".join(parts)


@dataclass
class _State:
    at: float
    failures_in_a_row: int = 0
    net_failures: int = 0
    disk_failures: int = 0
    reading: dict[str, Any] = field(default_factory=dict)
    measured: bool = False
    # DAH-5244: readings in a row whose probe container was left out under disk pressure; at
    # DISK_PRESSURE_MAX_DEFERRALS the probe runs regardless, so the sides are measured again
    probe_skips_in_a_row: int = 0
    late_skips: int = 0

    def next_due_at(self) -> float:
        if self.failures_in_a_row > 0 or self.reading.get("inconclusive"):
            return self.at + RETRY_SECONDS
        return self.at + settings.RENTED_THROUGHPUT_INTERVAL_HOURS * 3600

    @classmethod
    def from_raw(cls, raw: Any) -> _State | None:
        try:
            data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            return cls(
                at=float(data["at"]),
                failures_in_a_row=int(data.get("failures_in_a_row", 0)),
                net_failures=int(data.get("net_failures", 0)),
                disk_failures=int(data.get("disk_failures", 0)),
                reading=dict(data.get("reading") or {}),
                measured=bool(data.get("measured")),
                probe_skips_in_a_row=int(data.get("probe_skips_in_a_row", 0)),
                late_skips=int(data.get("late_skips", 0)),
            )
        except (TypeError, ValueError, KeyError, AttributeError):
            return None


class RentedThroughputCheck:
    """What a renter's pod gets from the network and the disk on a rented node (a renter saw 0.7-2 MB/s downloads
    and a crawling disk on a node listed at ~3 Gbps). ContainerDownloadSpeedCheck measures idle nodes only, so a
    host that shapes or starves a node once it is rented kept its score.

    Once per RENTED_THROUGHPUT_INTERVAL_HOURS at a per-node phase, a throwaway container on the rentals bridge (the
    renter's runtime) writes DISK_PROBE_BYTES with fsync and reads the two pinned files of ContainerDownloadSpeedCheck.
    A reading fails when a download fails that check's test (CONTAINER_DOWNLOAD_SPEED_MIN_MBPS, or unreachable while
    the validator reaches it) or the write runs under RENTED_THROUGHPUT_DISK_MIN_MB_PER_S. The renter's own traffic
    is no outright excuse below the floors. When the 2 s before the probe already moved RENTED_THROUGHPUT_BUSY_MBPS on
    the host NICs (either direction), or RENTED_THROUGHPUT_DISK_BUSY_MB_PER_S on a disk under the docker root, a slow
    side is inconclusive (the provider can make that load itself), and so is a slow network side without host NIC
    counters or off lium-rentals (bridge fallback) and a slow disk whose docker root devices are unresolved; a fetch
    container that never reports and a disk write that fails or reports nothing are inconclusive too. Only an HTTP
    error or an endpoint the validator cannot reach either stays no verdict (not the node's doing). When the
    rented pods themselves moved DURING_* on that resource while the probe measured it (their own network namespace
    and container cgroup counters, not the host's), a failing side is no verdict, counting only pods whose cgroup or
    namespace held just the container's own processes; a pod with a foreign (host) process adds nothing, and one that
    cannot be read makes it inconclusive; an unsure measurement stays inconclusive. Incomplete pod counters (a pod stopped, restarted or unreadable) make a
    failing side inconclusive too. Inconclusive = no strike and no pass however many in a row (owner decision: IO that
    cannot be attributed never fails a node), re-probed after RETRY_SECONDS.
    READINGS_BEFORE_VERDICT failing readings of one side in a row (RETRY_SECONDS apart; a side's lone failure older than
    STREAK_GAP_SECONDS starts no streak) zero the score
    (RENTED_THROUGHPUT_SLOW) until that side passes a reading; so do LATE_SKIPS_BEFORE_VERDICT due readings in a row
    skipped for too little cycle time (RENTED_THROUGHPUT_NOT_RUN), until a reading completes. Non-fatal: the rental keeps running.

    DAH-5244: the command also times `sync -f` on the Docker root (host_disk). Under
    DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED a stalling sync leaves the probe container out - its start would wait
    on the same pages - and both sides are no verdict, DISK_PRESSURE_MAX_DEFERRALS readings in a row at most before
    the probe runs regardless; a probe that does run under pressure measures and logs the disk but judges no disk
    side (busy, not slow, as the busy window excuses it); in shadow the probe runs and the reading says what the
    verdict would have been (`disk_verdict_shadow`).
    """

    check_id = "host.validate.rented_throughput"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.RENTED_THROUGHPUT_CHECK_ENABLED:
            return self._skipped(ctx, "RENTED_THROUGHPUT_CHECK_ENABLED is off")
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        if rented_executor is None or rented_executor.miner_hotkey != ctx.miner_hotkey:
            return self._skipped(ctx, "not rented")
        if not any(pod.status in (None, POD_STATUS_RUNNING) for pod in rented_executor.pods):
            return self._skipped(ctx, "no running pod")

        key = f"{_REDIS_PREFIX}:{ctx.executor.uuid}"
        try:
            raw = await ctx.services.redis.get(key)
        except Exception:
            logger.warning(
                _m("Rented throughput: Redis unreadable", extra=get_extra_info(ctx.default_extra))
            )
            return self._skipped(ctx, "last reading unreadable in Redis")
        state = _State.from_raw(raw) if raw is not None else None
        unsaved = _UNSAVED.get(ctx.executor.uuid)
        # a late skip keeps .at, so a newer unsaved skip count wins on a tie
        if unsaved is not None and (
            state is None or (unsaved.at, unsaved.late_skips) > (state.at, state.late_skips)
        ):
            state = unsaved
        now = time.time()
        if state is None:
            state = _State(at=now - phase_offset_seconds(ctx.executor.uuid))
            # kept even unsaved, else each cycle recreates it and the first reading never comes due
            await self._bounded_save(ctx, key, state)
        if now < state.next_due_at():
            return self._verdict(ctx, state, probed=False)
        if task_seconds_left(ctx) < MIN_TASK_SECONDS_LEFT:
            # no reading this cycle: a standing SLOW stays, otherwise skip; still due, so the next cycle measures
            if state.failures_in_a_row >= READINGS_BEFORE_VERDICT:
                return self._verdict(ctx, state, probed=False)
            state.late_skips += 1
            await self._bounded_save(ctx, key, state)
            if state.late_skips >= LATE_SKIPS_BEFORE_VERDICT:
                return self._verdict(ctx, state, probed=False)
            return self._skipped(ctx, "too little cycle time left")

        skip_under_pressure = await self._reserve_probe_skip(ctx, key, state)
        reading, net, disk = await self._measure_and_judge(ctx, skip_under_pressure=skip_under_pressure)
        reading["inconclusive"] = "inconclusive" in (net, disk)
        net_failures, disk_failures = state.net_failures, state.disk_failures
        if now - state.at > STREAK_GAP_SECONDS:
            # per side, only a strike short of a verdict expires: a standing SLOW stays until a reading passes
            net_failures, disk_failures = (n if n >= READINGS_BEFORE_VERDICT else 0 for n in (net_failures, disk_failures))
        net_failures, disk_failures = _step(net_failures, net), _step(disk_failures, disk)
        failures = max(net_failures, disk_failures)
        state = _State(
            at=now,
            failures_in_a_row=failures,
            net_failures=net_failures,
            disk_failures=disk_failures,
            reading=reading,
            measured=not {"none", "inconclusive"} & {net, disk},
            probe_skips_in_a_row=state.probe_skips_in_a_row + 1 if "probe_skipped" in reading else 0,
        )
        logger.info(
            _m(
                "RENTED_THROUGHPUT",
                extra=get_extra_info(
                    {
                        **ctx.default_extra,
                        **reading,
                        "failing": "fail" in (net, disk),
                        "failures_in_a_row": failures,
                    }
                ),
            )
        )
        await self._bounded_save(ctx, key, state)
        return self._verdict(ctx, state, probed=True)

    async def _reserve_probe_skip(self, ctx: Context, key: str, state: _State) -> bool:
        """Whether this reading may leave the probe container out under pressure: enforced, skips left,
        and the skip already on record (a Redis that takes reads but refuses writes would never count it,
        so the probe runs instead)."""
        if not (settings.DISK_PRESSURE_CHECK_ENABLED and settings.DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED):
            return False
        if state.probe_skips_in_a_row >= DISK_PRESSURE_MAX_DEFERRALS:
            return False
        budget = task_seconds_left(ctx) - SAVE_RESERVE_SECONDS
        if budget <= 0:
            return False
        reserved = replace(state, probe_skips_in_a_row=state.probe_skips_in_a_row + 1)
        return await self._save(ctx, key, reserved, timeout=budget)

    async def _measure_and_judge(
        self, ctx: Context, *, skip_under_pressure: bool = False
    ) -> tuple[dict[str, Any], str, str]:
        """The reading and each side's verdict: "fail", "pass", "none" (no verdict) or "inconclusive"
        (a slow side that may not be the host's fault: no strike, re-probed sooner)."""
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        all_pods = rented_executor.pods if rented_executor else []
        # every rented pod's IO is the renter's, whatever its backend status (a running one may be reported as in
        # transition); completeness counts the running ones, whose counters must be readable
        pods = [p for p in all_pods if p.status in (None, POD_STATUS_RUNNING)]
        result = await ctx.runner.run(
            throughput_command(
                bool(ctx.state.sysbox_runtime),
                # running pods first: pods_incomplete checks them by position
                [
                    pod.docker_container_name
                    for pod in pods + [p for p in all_pods if p not in pods]
                ],
                read_pressure=settings.DISK_PRESSURE_CHECK_ENABLED,
                skip_under_pressure=skip_under_pressure,
            ),
            timeout=COMMAND_TIMEOUT_SECONDS,
            retryable=False,
        )
        stdout = result.stdout or ""
        pressure = parse_disk_pressure(stdout) if settings.DISK_PRESSURE_CHECK_ENABLED else None
        skip_marker_present = any(line.strip() == PROBE_SKIPPED_MARKER for line in stdout.splitlines())
        if skip_marker_present and skip_under_pressure and pressure is not None and pressure.pressure:
            # DAH-5244: no container beside a Docker root whose sync already stalls; nothing measured, the
            # streaks stand (a node already at two failures stays zeroed until a measured pass)
            return self._unprobed_reading(result, pressure), "none", "none"
        has_unsolicited_skip_marker = PROBE_SKIPPED_MARKER in stdout
        if has_unsolicited_skip_marker:
            # only the command this reading sent, and only under the pressure it read, leaves the probe out;
            # a marker from anywhere else grants nothing and the rest of the output is judged as always
            logger.warning(
                _m("Rented throughput: unsolicited skip marker in the output", extra=get_extra_info(ctx.default_extra))
            )
        endpoints = parse_download_speed(stdout)
        disk = parse_disk(stdout)
        net_mbps, disk_busy = parse_busy(stdout)
        during_net, (during_disk, during_iops) = parse_during(stdout)
        own = parse_during(stdout, (True,))
        maybe = parse_during(stdout, (True, None))
        net_incomplete, disk_incomplete = pods_incomplete(stdout, len(pods))
        net_provenance, disk_provenance = pods_provenance(stdout)
        # A busy host before the probe, no host counters, docker root devices unresolved or a bridge fallback make a
        # slow side inconclusive: who moved that IO, or what the probe measured, is unknown. A fast side stays none.
        net_busy = net_mbps is not None and net_mbps >= settings.RENTED_THROUGHPUT_BUSY_MBPS
        disk_is_busy = (
            disk_busy is not None and disk_busy >= settings.RENTED_THROUGHPUT_DISK_BUSY_MB_PER_S
        )
        # DAH-5244, enforced: a Docker root whose sync stalls is busy, not slow - the dd waited on the tenant's
        # pages - so a probe that ran under pressure (the skip budget spent or its reservation refused) still
        # judges no disk side; what `disk_verdict_shadow` promises below is what enforcement delivers
        disk_under_pressure = (
            settings.DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED and pressure is not None and pressure.pressure
        )

        judged = await ContainerDownloadSpeedCheck()._judge(endpoints)
        # a bridge fallback (rentals network not inspectable) did not measure the renter's path
        off_rentals = not endpoints or any(r.network != RENTAL_NETWORK_NAME for r in endpoints)
        net_unsure = net_busy or net_mbps is None or off_rentals
        if judged.is_failing:
            net = "fail"
        elif any(r.error == "no_output" for r in endpoints):
            # the fetch container did not start or died before reporting: nothing measured, no strike
            net = "inconclusive"
        elif judged.has_no_verdict:
            # the validator cannot reach that endpoint either, or it answered with an HTTP error: not the node's doing
            net = "none"
        else:
            net = "none" if net_unsure else "pass"

        def net_load(during: tuple[Any, Any]) -> bool:
            return during[0] is not None and during[0] >= DURING_NET_MBPS

        def disk_load(during: tuple[Any, Any]) -> bool:
            mb, iops = during[1]
            return (mb or 0) >= DURING_DISK_MB_PER_S or (iops or 0) >= DURING_DISK_IOPS

        net = _excuse(net, net_load(own), net_load(maybe), net_unsure or net_incomplete)
        disk_unsure = disk_is_busy or disk_busy is None
        if disk.mb_per_s is None:
            # the disk write failed or reported nothing (the renter can saturate the disk past the timeout): no strike
            disk_side = "inconclusive"
        elif disk.mb_per_s < settings.RENTED_THROUGHPUT_DISK_MIN_MB_PER_S:
            disk_side = "fail"
        else:
            disk_side = "none" if disk_unsure else "pass"
        disk_side = _excuse(
            disk_side, disk_load(own), disk_load(maybe), disk_unsure or disk_incomplete
        )
        if disk_under_pressure:
            disk_side = "none"
        reading = {
            "endpoints": [asdict(r) for r in endpoints],
            "disk": asdict(disk),
            "other_net_mbps": net_mbps,
            "other_disk_mb_per_s": disk_busy,
            "during_net_mbps": during_net,
            "during_disk_mb_per_s": during_disk,
            "during_disk_iops": during_iops,
            "pod_provenance": [net_provenance, disk_provenance],
            "network_verdict": net,
            "disk_verdict": disk_side,
            "min_mbps": settings.CONTAINER_DOWNLOAD_SPEED_MIN_MBPS,
            "disk_min_mb_per_s": settings.RENTED_THROUGHPUT_DISK_MIN_MB_PER_S,
        }
        if pressure is not None:
            # shadow: what the disk side would be with the gate enforced, next to what it is
            reading["disk_pressure"] = pressure.as_spec()
            reading["disk_verdict_shadow"] = "none" if pressure.pressure else disk_side
        if has_unsolicited_skip_marker:
            reading["unsolicited_skip_marker"] = True
        if result.error_type is not None:
            reading["error"] = f"{result.error_type}: {(result.error_message or '')[-200:]}"
        return reading, net, disk_side

    @staticmethod
    def _unprobed_reading(result: SSHCommandResult, pressure: DiskPressureReading | None) -> dict[str, Any]:
        reading: dict[str, Any] = {
            "endpoints": [],
            "disk": asdict(DiskReading(error="not_probed")),
            "network_verdict": "none",
            "disk_verdict": "none",
            "probe_skipped": "disk pressure",
            "disk_pressure": pressure.as_spec() if pressure is not None else None,
            "disk_verdict_shadow": "none",
        }
        if result.error_type is not None:
            reading["error"] = f"{result.error_type}: {(result.error_message or '')[-200:]}"
        return reading

    def _verdict(self, ctx: Context, state: _State, *, probed: bool) -> CheckResult:
        what: dict[str, Any] = {
            **state.reading,
            "probed_this_cycle": probed,
            "failures_in_a_row": state.failures_in_a_row,
            "probe_skips_in_a_row": state.probe_skips_in_a_row,
            "late_skips_in_a_row": state.late_skips,
        }
        slow = state.failures_in_a_row >= READINGS_BEFORE_VERDICT
        if slow or state.late_skips >= LATE_SKIPS_BEFORE_VERDICT:
            template = Msg.SLOW if slow else Msg.NOT_RUN
            detail = slow_detail(state) if slow else ""
            event = render_message(
                template,
                ctx=ctx,
                check_id=self.check_id,
                what={**what, "failed": detail} if detail else what,
                remediation=f"failed: {detail}. {template.remediation}" if detail else None,
            )
            # NOT_RUN measured nothing, so scoring and the portal must not call it slow
            updates = (
                {"rented_throughput_passed": False, "rented_throughput_slow_detail": detail}
                if slow
                else {
                    "rented_throughput_passed": False,
                    "rented_throughput_not_run": True,
                }
            )
            return CheckResult(passed=False, event=event, updates=updates)
        if not probed:
            return self._skipped(
                ctx, "not due", last={**what, "read_at": state.at, "next_at": state.next_due_at()}
            )
        if state.failures_in_a_row:
            template = Msg.SLOW_ONCE
        elif state.measured:
            template = Msg.OK
        else:
            template = Msg.UNMEASURED
        return CheckResult(
            passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )

    async def _bounded_save(self, ctx: Context, key: str, state: _State) -> None:
        """In memory first: a refused (READONLY failover) or slow save must not lose the state. The Redis save is
        bounded by the cycle time left so it cannot itself run into the task deadline and drop the cycle."""
        # bounded like Redis itself: a node not measured for _REDIS_TTL_SECONDS is gone; a due node keeps its
        # unsaved streak until its own next reading replaces it
        now = time.time()
        for uuid in [u for u, s in _UNSAVED.items() if now - s.at > _REDIS_TTL_SECONDS]:
            del _UNSAVED[uuid]
        _UNSAVED[ctx.executor.uuid] = state
        budget = task_seconds_left(ctx) - SAVE_RESERVE_SECONDS
        if budget > 0 and await self._save(ctx, key, state, timeout=budget):
            _UNSAVED.pop(ctx.executor.uuid, None)

    async def _save(self, ctx: Context, key: str, state: _State, *, timeout: float) -> bool:
        payload = {
            "at": state.at,
            "failures_in_a_row": state.failures_in_a_row,
            "net_failures": state.net_failures,
            "disk_failures": state.disk_failures,
            "reading": state.reading,
            "measured": state.measured,
            "probe_skips_in_a_row": state.probe_skips_in_a_row,
            "late_skips": state.late_skips,
        }
        try:
            await asyncio.wait_for(
                ctx.services.redis.set(key, json.dumps(payload), ex=_REDIS_TTL_SECONDS), timeout
            )
            return True
        except Exception:
            logger.warning(
                _m(
                    "Rented throughput: could not record the reading",
                    extra=get_extra_info(ctx.default_extra),
                )
            )
            return False

    def _skipped(
        self, ctx: Context, reason: str, *, last: dict[str, Any] | None = None
    ) -> CheckResult:
        what: dict[str, Any] = {"reason": reason}
        if last is not None:
            what["last"] = last
        return CheckResult(
            passed=True,
            event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what=what),
        )
