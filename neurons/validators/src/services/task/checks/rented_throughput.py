from __future__ import annotations

import hashlib
import json
import logging
import shlex
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from services.rental_docker_sdk import RENTAL_NETWORK_NAME

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import RentedThroughputMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .container_download_speed import (
    _FETCH_SCRIPT,
    _TIMEOUT_EXITS,
    ContainerDownloadSpeedCheck,
    download_speed_command,
    parse_download_speed,
)
from .rented_gpu_drop import POD_STATUS_RUNNING

logger = logging.getLogger(__name__)

DISK_PROBE_BYTES = 256 * 1024 * 1024
DISK_TIMEOUT_SECONDS = 60
# the 2 s busy window, the disk probe (direct, then buffered), both fetches and a second container start
COMMAND_TIMEOUT_SECONDS = 270
RETRY_SECONDS = 30 * 60
READINGS_BEFORE_VERDICT = 2
BUSY_WINDOW_SECONDS = 2
DISK_MARKER = "lium_disk"
SNAP_MARKER = "lium_tp"
_REDIS_PREFIX = "rented_throughput"
_REDIS_TTL_SECONDS = 7 * 24 * 3600
# The rented pods' own traffic while the probe measured a side (their network namespace's interfaces around the fetch
# container, their container cgroup's io.stat around the disk container) at or above these makes a failing side
# inconclusive: the renter may have been using what the probe measured. Only the pods' own counters count, so traffic
# the host makes anywhere else cannot excuse a slow node. The host can still attach its own process to a pod's cgroup or
# network namespace, so the counters prove nothing: an excused side is only inconclusive (INCONCLUSIVE_BEFORE_STRIKE).
DURING_NET_MBPS = 5.0
DURING_DISK_MB_PER_S = 2.0
DURING_DISK_IOPS = 20.0
# A failing side is also inconclusive when its pod counters are incomplete (any rented pod not readable both when that
# side began and when it ended: stopped or restarted mid-probe, never readable, or no snapshot): the renter's IO is
# unknown. Inconclusive = no strike and no pass, and the node is probed again after RETRY_SECONDS. The provider can make
# a side inconclusive too (restart the pod, or feed IO into its cgroup or netns), so this many inconclusive readings of
# one side in a row (no pass or fail in between) count as a failing reading, and every further one in that run does too.
INCONCLUSIVE_BEFORE_STRIKE = 3
# executor uuid -> a reading Redis refused to record; keeps the cadence (no probe per cycle) until a save succeeds
_UNSAVED: dict[str, _State] = {}

# The executor shares the host PID namespace but not its network (bridge, `pid: host`), so the NIC counters are
# init's (/proc/1/net/dev), read only when /proc/1 is the host's; else no rx/tx and the network side is no verdict.
# The devices under the docker root (the probe's writable layer) come from init's mountinfo through sysfs: the
# device, its dm/md slaves and a partition's disk. Then bytes received and sent on the physical NICs and sectors
# moved (and requests completed) per whole disk (/proc/diskstats is not namespaced), twice BUSY_WINDOW_SECONDS apart
# before the probe, then again after the disk container and after the fetch container (throughput_command).
_SNAPSHOT = (
    "r=$(/usr/bin/docker info -f '{{.DockerRootDir}}' 2>/dev/null); "
    'm=$(awk -v r="${r:-/var/lib/docker}" \'r==$5 || $5=="/" || index(r, $5 "/")==1 '
    "{ if (length($5) >= n) { n=length($5); m=$3 } } END{print m}' /proc/1/mountinfo 2>/dev/null); "
    'ds=; [ -n "$m" ] && ds=$(readlink -f /sys/dev/block/$m 2>/dev/null) && ds=${ds##*/}; '
    "for i in 1 2 3 4; do n=; for d in $ds; do [ -e /sys/class/block/$d ] || continue; "
    f'echo "{SNAP_MARKER}_device name=$d"; '
    '[ -e /sys/class/block/$d/partition ] && p=$(readlink -f /sys/class/block/$d) && p=${p%/*} && n="$n ${p##*/}"; '
    'n="$n $(ls /sys/class/block/$d/slaves 2>/dev/null)"; done; ds=$n; done; '
    # Only physical NICs (a sysfs `device` link) are counted: disjoint, so a bond, bridge, VLAN or OVS port can neither
    # repeat nor hide an uplink. Without a host sysfs (this shell's own is the container's), names and bond members.
    'ph=$(for i in /proc/1/root/sys/class/net/*; do [ -e $i/device ] && [ "${i##*/}" != lo ] && '
    'printf " %s" "${i##*/}"; done); '
    "sl=$(for i in /proc/1/root/sys/class/net/*; do [ -e $i/master ] && m=$(readlink $i/master) && "
    'printf " %s:%s" "${i##*/}" "${m##*/}"; done; '
    'for f in /proc/1/net/bonding/*; do [ -f "$f" ] && '
    'awk -v b="${f##*/}" \'/^Slave Interface:/{printf " %s:%s", $3, b}\' "$f"; done 2>/dev/null); '
    # Per rented pod ($pp: pid/container id, set by throughput_command): rx:tx of its own network namespace (not
    # the host's) and bytes:requests of its container cgroup (systemd or cgroupfs driver); a field left empty is not
    # readable. Nothing when /proc/1 is not the host's, where a docker pid means nothing here.
    "pods() { [ /proc/1/root/ -ef / ] && return; for x in $pp; do p=${x%%/*}; c=${x#*/}; "
    "n=$([ /proc/$p/ns/net -ef /proc/1/ns/net ] || sed 's/:/ /' /proc/$p/net/dev 2>/dev/null | "
    'awk \'NR>2 && $1!="lo"{r+=$2; t+=$10; k++} END{if (k) printf "%.0f:%.0f", r, t}\'); d=; '
    "for f in /proc/1/root/sys/fs/cgroup/system.slice/docker-$c.scope/io.stat /proc/1/root/sys/fs/cgroup/docker/$c/io.stat; "
    'do [ -r $f ] && d=$(awk \'{for (i=2; i<=NF; i++) {split($i, a, "="); '
    'if (a[1]=="rbytes" || a[1]=="wbytes") b+=a[2]; if (a[1]=="rios" || a[1]=="wios") n+=a[2]}} '
    'END{printf "%.0f:%.0f", b, n}\' $f) && break; done; '
    'printf "%s:%s:%s," "$p" "${n:-:}" "${d:-:}"; done; }; '
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


def throughput_command(sysbox: bool, containers: list[str]) -> str:
    # one container per side, counters after each: what the renter's pods moved while that side was measured
    names = " ".join(shlex.quote(c) for c in containers)
    return (
        f"pp=$(for c in {names}; do /usr/bin/docker inspect -f '{{{{.State.Pid}}}}/{{{{.Id}}}}' \"$c\" 2>/dev/null; done); "
        + _SNAPSHOT
        + download_speed_command(sysbox, _DISK_SCRIPT)
        + "; snap; "
        + download_speed_command(sysbox, _FETCH_SCRIPT)
        + "; snap"
    )


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


# (uptime, (rx, tx) or None, {disk: (sectors, requests)}, {pod pid: ((rx, tx) or None, (bytes, requests) or None)})
_Snap = tuple[float, Any, dict[str, tuple[int, int]], dict[str, tuple[Any, Any]]]


def _pair(a: str, b: str) -> tuple[int, int] | None:
    return (int(a), int(b)) if a and b else None


def _parse_snaps(stdout: str) -> tuple[list[_Snap], set[str]]:
    snaps: list[_Snap] = []
    devices: set[str] = set()
    for line in (stdout or "").splitlines():
        if line.startswith(f"{SNAP_MARKER}_device name="):
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
                        pid, rx, tx, moved, ios = e.split(":")
                        pods[pid] = (_pair(rx, tx), _pair(moved, ios))
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


def _pods_moved(a: _Snap, b: _Snap, side: int) -> tuple[float, float] | None:
    """What the rented pods themselves moved between two snapshots, summed per second: (received, sent)
    for side 0 (their network namespaces' bytes), (bytes, requests) per second for side 1 (their cgroups' disk
    IO). Only pods readable in both snapshots count: one that restarted (a new pid) or whose counter went unreadable
    adds nothing but does not void the others. None = not measured: no pod readable in both."""
    (t0, *_, pods0), (t1, *_, pods1) = a, b
    both = [
        (c[side], pods1[pid][side])
        for pid, c in pods0.items()
        if pid in pods1 and c[side] is not None and pods1[pid][side] is not None
    ]
    if not both:
        return None
    dt = max(t1 - t0, 0.01)
    return (
        sum(max(0, c1[0] - c0[0]) for c0, c1 in both) / dt,
        sum(max(0, c1[1] - c0[1]) for c0, c1 in both) / dt,
    )


def pods_incomplete(stdout: str, pods: int) -> tuple[bool, bool]:
    """(network, disk): fewer than `pods` rented pods were readable both when that side's measurement began and when
    it ended (a pod stopped or restarted mid-probe, was never readable, or a snapshot is missing), so what the renter
    moved while that side was measured is not fully known."""
    snaps, _ = _parse_snaps(stdout)
    if len(snaps) < 4:
        return pods > 0, pods > 0

    def incomplete(a: _Snap, b: _Snap, side: int) -> bool:
        both = [
            pid
            for pid, c in a[3].items()
            if c[side] is not None and b[3].get(pid, (None, None))[side] is not None
        ]
        return len(both) < pods

    return incomplete(snaps[2], snaps[3], 0), incomplete(snaps[1], snaps[2], 1)


def parse_busy(stdout: str) -> tuple[float | None, float | None]:
    """Other traffic in the window before the probe: (Mbps on the host NICs, the busier direction; MB/s on the busiest
    disk under the docker root). None = not measured; the disk side then has no verdict."""
    snaps, devices = _parse_snaps(stdout)
    if len(snaps) < 2:
        return None, None
    return _moved(snaps[0], snaps[1], devices)


def parse_during(stdout: str) -> tuple[float | None, tuple[float | None, float | None]]:
    """The rented pods' own traffic while each side was measured: (network Mbps, the busier direction, during the
    fetch container; (disk MB/s, disk IO/s) during the disk container). The probe and anything else on the host is
    outside those counters. None = not measured."""
    snaps, _ = _parse_snaps(stdout)
    if len(snaps) < 4:
        return None, (None, None)
    net = _pods_moved(snaps[2], snaps[3], 0)
    disk = _pods_moved(snaps[1], snaps[2], 1)
    return (
        None if net is None else max(net) * 8 / 1e6,
        (None, None) if disk is None else (disk[0] / 1e6, disk[1]),
    )


def phase_offset_seconds(uuid: str) -> float:
    fraction = int(hashlib.sha256(uuid.encode()).hexdigest()[:12], 16) / 16**12
    return fraction * settings.RENTED_THROUGHPUT_INTERVAL_HOURS * 3600


def _inconclusive(count: int, side: str) -> tuple[int, str]:
    """A side's inconclusive run after this reading, and the side as the streak counts it: a pass or fail ends the
    run, no verdict keeps it, and from INCONCLUSIVE_BEFORE_STRIKE in a row an inconclusive reading is a failure."""
    if side != "inconclusive":
        return (count if side == "none" else 0), side
    count += 1
    return count, ("fail" if count >= INCONCLUSIVE_BEFORE_STRIKE else "none")


def _step(failures: int, side: str) -> int:
    """One streak per side: a side that measures and passes ends its own streak whatever the other side says."""
    return failures + 1 if side == "fail" else (failures if side == "none" else 0)


@dataclass
class _State:
    at: float
    failures_in_a_row: int = 0
    net_failures: int = 0
    disk_failures: int = 0
    reading: dict[str, Any] = field(default_factory=dict)
    measured: bool = False
    net_inconclusive: int = 0
    disk_inconclusive: int = 0

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
                net_inconclusive=int(data.get("net_inconclusive", 0)),
                disk_inconclusive=int(data.get("disk_inconclusive", 0)),
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
    is no excuse below the floors: only when the 2 s before the probe already moved RENTED_THROUGHPUT_BUSY_MBPS on the
    host NICs (either direction), or RENTED_THROUGHPUT_DISK_BUSY_MB_PER_S on a disk under the docker root, is that side
    no verdict (a link or disk doing that much is not the defect); so is the network side without host NIC counters.
    A failing side is inconclusive (no strike, no pass, re-probed after RETRY_SECONDS) when the rented pods themselves
    moved DURING_* on that resource while the probe measured it (their own network namespace and container cgroup
    counters, not the host's), or when those pod counters are incomplete (a pod stopped, restarted or unreadable): one
    renter-induced slow reading never fails a node. The host can feed or hide those counters, so
    INCONCLUSIVE_BEFORE_STRIKE inconclusive readings of one side in a row count as a failing reading.
    READINGS_BEFORE_VERDICT failing readings of one side in a row (RETRY_SECONDS apart) zero the score
    (RENTED_THROUGHPUT_SLOW) until that side passes a reading. Non-fatal: the rental keeps running.
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
        if unsaved is not None and (state is None or unsaved.at > state.at):
            state = unsaved
        now = time.time()
        if state is None:
            state = _State(at=now - phase_offset_seconds(ctx.executor.uuid))
            if not await self._save(ctx, key, state):
                # else each cycle recreates it and the first reading never comes due
                _UNSAVED[ctx.executor.uuid] = state
        if now < state.next_due_at():
            return self._verdict(ctx, state, probed=False)

        reading, net, disk = await self._measure_and_judge(ctx)
        reading["inconclusive"] = "inconclusive" in (net, disk)
        net_inconclusive, net = _inconclusive(state.net_inconclusive, net)
        disk_inconclusive, disk = _inconclusive(state.disk_inconclusive, disk)
        net_failures, disk_failures = (
            _step(state.net_failures, net),
            _step(state.disk_failures, disk),
        )
        failures = max(net_failures, disk_failures)
        state = _State(
            at=now,
            failures_in_a_row=failures,
            net_failures=net_failures,
            disk_failures=disk_failures,
            reading=reading,
            measured="none" not in (net, disk),
            net_inconclusive=net_inconclusive,
            disk_inconclusive=disk_inconclusive,
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
        if await self._save(ctx, key, state):
            _UNSAVED.pop(ctx.executor.uuid, None)
        else:
            # bounded like Redis itself: a node not measured for _REDIS_TTL_SECONDS is gone; a due node keeps its
            # unsaved streak until its own next reading replaces it
            for uuid in [u for u, s in _UNSAVED.items() if now - s.at > _REDIS_TTL_SECONDS]:
                del _UNSAVED[uuid]
            _UNSAVED[ctx.executor.uuid] = state
        return self._verdict(ctx, state, probed=True)

    async def _measure_and_judge(self, ctx: Context) -> tuple[dict[str, Any], str, str]:
        """The reading and each side's verdict: "fail", "pass" or "none" (no verdict)."""
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        pods = [
            p
            for p in (rented_executor.pods if rented_executor else [])
            if p.status in (None, POD_STATUS_RUNNING)
        ]
        result = await ctx.runner.run(
            throughput_command(
                bool(ctx.state.sysbox_runtime),
                [pod.docker_container_name for pod in pods],
            ),
            timeout=COMMAND_TIMEOUT_SECONDS,
            retryable=False,
        )
        stdout = result.stdout or ""
        endpoints = parse_download_speed(stdout)
        disk = parse_disk(stdout)
        net_mbps, disk_busy = parse_busy(stdout)
        during_net, (during_disk, during_iops) = parse_during(stdout)
        net_incomplete, disk_incomplete = pods_incomplete(stdout, len(pods))
        # no host counters: a slow download may be the renter's, so the network side is no verdict
        net_busy = net_mbps is None or net_mbps >= settings.RENTED_THROUGHPUT_BUSY_MBPS
        disk_is_busy = (
            disk_busy is None or disk_busy >= settings.RENTED_THROUGHPUT_DISK_BUSY_MB_PER_S
        )

        judged = await ContainerDownloadSpeedCheck()._judge(endpoints)
        # a bridge fallback (rentals network not inspectable) did not measure the renter's path
        off_rentals = not endpoints or any(r.network != RENTAL_NETWORK_NAME for r in endpoints)
        if net_busy or off_rentals or judged.has_no_verdict:
            net = "none"
        else:
            net = "fail" if judged.is_failing else "pass"
        # a failing side is inconclusive when the renter's own pods used that resource while it was measured, or their
        # counters are incomplete; never an outright excuse, the host can feed or hide those counters
        if net == "fail" and (
            net_incomplete or (during_net is not None and during_net >= DURING_NET_MBPS)
        ):
            net = "inconclusive"
        if disk_is_busy or disk.mb_per_s is None:
            disk_side = "none"
        else:
            disk_side = (
                "fail" if disk.mb_per_s < settings.RENTED_THROUGHPUT_DISK_MIN_MB_PER_S else "pass"
            )
        if disk_side == "fail" and (
            disk_incomplete
            or (during_disk or 0) >= DURING_DISK_MB_PER_S
            or (during_iops or 0) >= DURING_DISK_IOPS
        ):
            disk_side = "inconclusive"
        reading = {
            "endpoints": [asdict(r) for r in endpoints],
            "disk": asdict(disk),
            "other_net_mbps": net_mbps,
            "other_disk_mb_per_s": disk_busy,
            "during_net_mbps": during_net,
            "during_disk_mb_per_s": during_disk,
            "during_disk_iops": during_iops,
            "network_verdict": net,
            "disk_verdict": disk_side,
            "min_mbps": settings.CONTAINER_DOWNLOAD_SPEED_MIN_MBPS,
            "disk_min_mb_per_s": settings.RENTED_THROUGHPUT_DISK_MIN_MB_PER_S,
        }
        if result.error_type is not None:
            reading["error"] = f"{result.error_type}: {(result.error_message or '')[-200:]}"
        return reading, net, disk_side

    def _verdict(self, ctx: Context, state: _State, *, probed: bool) -> CheckResult:
        what: dict[str, Any] = {
            **state.reading,
            "probed_this_cycle": probed,
            "failures_in_a_row": state.failures_in_a_row,
        }
        if state.failures_in_a_row >= READINGS_BEFORE_VERDICT:
            event = render_message(Msg.SLOW, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(
                passed=False, event=event, updates={"rented_throughput_passed": False}
            )
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

    async def _save(self, ctx: Context, key: str, state: _State) -> bool:
        payload = {
            "at": state.at,
            "failures_in_a_row": state.failures_in_a_row,
            "net_failures": state.net_failures,
            "disk_failures": state.disk_failures,
            "reading": state.reading,
            "measured": state.measured,
            "net_inconclusive": state.net_inconclusive,
            "disk_inconclusive": state.disk_inconclusive,
        }
        try:
            await ctx.services.redis.set(key, json.dumps(payload), ex=_REDIS_TTL_SECONDS)
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
