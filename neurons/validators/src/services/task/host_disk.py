"""The host's Docker-root disk as the validator's containers are about to meet it (DAH-5244).

A `docker run` on an overlay root waits in `ovl_sync_fs -> sync_inodes_sb` for every dirty page on the Docker
filesystem's superblock. On the 2026-10-28 stand a tenant writing through the page cache cost 50-370 s per
`docker run`; the same writer in a direct-I/O loop volume on the same saturated device cost 1 s. Device
utilisation and host Dirty were the same in both, so neither predicts the stall; the time `sync -f` takes on
the Docker root does, because it is that very wait. The diskstats and meminfo fields around it are telemetry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# a healthy Docker root syncs in well under a second; beyond this the port-check containers (60 s) are at risk
DISK_PRESSURE_SYNCFS_SECONDS = 5.0
# how long the shell waits for the sync before reporting it as timed out: a third of the port-check timeout
SYNCFS_POLL_SECONDS = 20
# consecutive cycles a rented or filler node may skip its port check under pressure before it runs regardless
DISK_PRESSURE_MAX_DEFERRALS = 3
# `await` over fewer completed I/Os than this is a few slow requests, not a latency
AWAIT_MIN_COMPLETED_IOS = 20
DEVICE_MARKER = "lium_tp_device"
PRESSURE_MARKER = "lium_dp"
SNAPSHOT_SECONDS = 2
_SYNC_UNAVAILABLE_RC = 127

# The devices under the docker root (the probe's writable layer) from init's mountinfo through sysfs: the
# device itself (a partition, dm, md, nbd or loop), its dm/md slaves and a partition's disk, four levels deep.
# `docker info` is bounded: on the hosts this reading is for, dockerd itself may be stuck in `sync_inodes_sb`.
DOCKER_ROOT_DEVICES = (
    "r=$(timeout 5 /usr/bin/docker info -f '{{.DockerRootDir}}' 2>/dev/null); r=${r:-/var/lib/docker}; "
    'm=$(awk -v r="$r" \'r==$5 || $5=="/" || index(r, $5 "/")==1 '
    "{ if (length($5) >= n) { n=length($5); m=$3 } } END{print m}' /proc/1/mountinfo 2>/dev/null); "
    'ds=; dv=; [ -n "$m" ] && ds=$(readlink -f /sys/dev/block/$m 2>/dev/null) && ds=${ds##*/}; '
    "for i in 1 2 3 4; do n=; for d in $ds; do [ -e /sys/class/block/$d ] || continue; "
    f'echo "{DEVICE_MARKER} name=$d"; dv="$dv $d"; '
    '[ -e /sys/class/block/$d/partition ] && p=$(readlink -f /sys/class/block/$d) && p=${p%/*} && n="$n ${p##*/}"; '
    'n="$n $(ls /sys/class/block/$d/slaves 2>/dev/null)"; done; ds=$n; done; '
)

# `sync -f` on the Docker root through the host's root (`/proc/1/root`: the executor shares the host PID
# namespace), detached with its descriptors closed so this command returns at the poll cap while a sync in D
# state finishes on its own; it only writes back pages the kernel was going to write back. Meanwhile two
# snapshots SNAPSHOT_SECONDS apart of /proc/diskstats (sectors, io_ticks, ms spent, I/Os completed) for the
# Docker-root devices, with the host's Dirty/Writeback and I/O pressure stall. `dp` is left set for the caller.
_PRESSURE_PROBE = (
    'h=/proc/1/root; [ $h/ -ef / ] && h=; sr="$h$r"; [ -d "$sr" ] || sr=$r; '
    "sf=/tmp/lium_syncfs.$$; rm -f $sf $sf.t; find /tmp -maxdepth 1 -name 'lium_syncfs.*' -mmin +60 -delete 2>/dev/null; "
    "ts=$(cut -d' ' -f1 /proc/uptime); "
    # the sync's own end time next to its rc, written whole: a half-written file would read as a timeout
    '( sync -f "$sr" 2>/dev/null; rc=$?; echo "$rc $(cut -d\' \' -f1 /proc/uptime)" > $sf.t && mv $sf.t $sf ) '
    "</dev/null >/dev/null 2>&1 & "
    f'dps() {{ echo "{PRESSURE_MARKER} t=$(cut -d\' \' -f1 /proc/uptime) '
    "dirty=$(awk '/^Dirty:/{d=$2} /^Writeback:/{w=$2} END{print d\":\"w}' $h/proc/meminfo 2>/dev/null) "
    "psi=$(awk '/^full/{sub(\"avg10=\",\"\",$2); print $2}' $h/proc/pressure/io 2>/dev/null) "
    'io=$(awk -v ds=" $(echo $dv) " \'index(ds, " "$3" "){printf "%s:%.0f:%.0f:%.0f:%.0f,", '
    "$3, $6+$10, $13, $7+$11, $4+$8}' /proc/diskstats)\"; }; "
    f"dps; sleep {SNAPSHOT_SECONDS}; dps; i=0; "
    f"while [ $i -lt {SYNCFS_POLL_SECONDS * 4} ] && [ ! -e $sf ]; do sleep 0.25; i=$((i+1)); done; "
    "src=$(cat $sf 2>/dev/null); rm -f $sf; rc=${src%% *}; te=${src#* }; "
    "[ -n \"$src\" ] || te=$(cut -d' ' -f1 /proc/uptime); "
    f'echo "{PRESSURE_MARKER}_sync start=$ts end=$te rc=$rc"; '
    f'dp=0; if [ -z "$rc" ]; then dp=1; elif [ "$rc" = 0 ] && '
    f'awk -v s=$ts -v e=$te "BEGIN{{exit !(e-s >= {DISK_PRESSURE_SYNCFS_SECONDS})}}"; then dp=1; fi; '
)


def disk_pressure_command() -> str:
    return DOCKER_ROOT_DEVICES + _PRESSURE_PROBE


def disk_pressure_probe() -> str:
    """The probe alone, for a command that has already resolved the devices (`$r`, `$dv`)."""
    return _PRESSURE_PROBE


@dataclass(frozen=True)
class DiskPressureReading:
    syncfs_seconds: float | None = None
    syncfs_timed_out: bool | None = None
    # the Docker-root device with the most io_ticks in the window, and its numbers
    device: str | None = None
    mb_per_s: float | None = None
    util_percent: float | None = None
    await_ms: float | None = None
    dirty_mb: int | None = None
    writeback_mb: int | None = None
    psi_io_full_avg10: float | None = None
    pressure: bool = False
    # "syncfs" when pressure, else None
    reason: str | None = None
    # None = read; else why the sync part could not be: no_output, syncfs_unavailable, syncfs_failed, read_failed
    error: str | None = None

    def as_spec(self) -> dict[str, Any]:
        return asdict(self)


def _tokens(line: str, marker: str) -> dict[str, str]:
    return dict(t.partition("=")[::2] for t in line[len(marker) :].split())


def parse_disk_pressure(stdout: str) -> DiskPressureReading:
    """The reading from the probe's output; the disk fields are filled even when the sync part failed."""
    devices: list[str] = []
    snaps: list[dict[str, str]] = []
    sync: dict[str, str] | None = None
    for line in (stdout or "").splitlines():
        if line.startswith(f"{DEVICE_MARKER} name="):
            devices.append(line.partition("=")[2].strip())
        elif line.startswith(f"{PRESSURE_MARKER} "):
            snaps.append(_tokens(line, PRESSURE_MARKER))
        elif line.startswith(f"{PRESSURE_MARKER}_sync "):
            sync = _tokens(line, f"{PRESSURE_MARKER}_sync")
    fields: dict[str, Any] = {}
    if len(snaps) == 2:
        fields.update(_disk_fields(snaps[0], snaps[1], set(devices)))
    # no resolved device only blanks the disk fields: `sync -f` on the Docker root needs none of them
    if sync is None:
        return DiskPressureReading(**fields, error="no_output")
    try:
        seconds = float(sync["end"]) - float(sync["start"])
    except (KeyError, ValueError):
        return DiskPressureReading(**fields, error="no_output")
    rc = sync.get("rc", "")
    if rc == str(_SYNC_UNAVAILABLE_RC):
        return DiskPressureReading(**fields, error="syncfs_unavailable")
    if rc not in ("", "0"):
        return DiskPressureReading(**fields, error="syncfs_failed")
    timed_out = rc == ""
    pressure = timed_out or (rc == "0" and seconds >= DISK_PRESSURE_SYNCFS_SECONDS)
    return DiskPressureReading(
        **fields,
        syncfs_seconds=round(seconds, 2),
        syncfs_timed_out=timed_out,
        pressure=pressure,
        reason="syncfs" if pressure else None,
    )


def _disk_fields(first: dict[str, str], second: dict[str, str], devices: set[str]) -> dict[str, Any]:
    # the busiest (io_ticks) of the Docker-root devices over the window; another disk's traffic says nothing
    fields: dict[str, Any] = {}
    try:
        dt = max(float(second["t"]) - float(first["t"]), 0.01)
    except (KeyError, ValueError):
        return fields
    dirty, _, writeback = second.get("dirty", "").partition(":")
    if dirty.isdigit() and writeback.isdigit():
        fields["dirty_mb"] = int(dirty) // 1024
        fields["writeback_mb"] = int(writeback) // 1024
    try:
        fields["psi_io_full_avg10"] = float(second["psi"])
    except (KeyError, ValueError):
        pass
    before = _io_counters(first.get("io", ""))
    busiest: tuple[int, str, list[int]] | None = None
    for name, after in _io_counters(second.get("io", "")).items():
        if name not in before or name not in devices:
            continue
        delta = [a - b for a, b in zip(after, before[name])]
        if min(delta) < 0:  # a counter reset or a device that came back
            continue
        if busiest is None or delta[1] > busiest[0]:
            busiest = (delta[1], name, delta)
    if busiest is not None:
        _, name, (sectors, ticks, ms, ios) = busiest
        fields["device"] = name
        fields["mb_per_s"] = round(sectors * 512 / dt / 1e6, 1)
        fields["util_percent"] = round(min(ticks / (dt * 1000) * 100, 100.0), 1)
        fields["await_ms"] = round(ms / ios, 1) if ios >= AWAIT_MIN_COMPLETED_IOS else None
    return fields


def _io_counters(raw: str) -> dict[str, list[int]]:
    """device -> [sectors moved, io_ticks ms, ms spent, I/Os completed]."""
    counters: dict[str, list[int]] = {}
    for entry in raw.split(","):
        name, _, numbers = entry.partition(":")
        parts = numbers.split(":")
        if name and len(parts) == 4 and all(p.lstrip("-").isdigit() for p in parts):
            counters[name] = [int(p) for p in parts]
    return counters
