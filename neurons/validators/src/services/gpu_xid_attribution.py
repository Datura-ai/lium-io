"""DAH-3490: who broke the card — the renter's workload or the provider's hardware.

Rustam's rule (ticket DAH-3490, 16 Sep 2026): a card that is stuck after a rental and was healthy before
it proves nothing; a bad card often fails only under load. Attribute by the NVIDIA Xid lines the kernel
logged during the rental window:

- Xid 13, 31, 43, 45 (application errors on a healthy card) with a timestamp inside the rental, and the
  PID of the renter's container when the line names one: the workload broke it.
- Xid 48, 79, 94, 95 and every other hardware Xid, or uncorrected ECC errors on a card: provider hardware,
  even inside a rental.
- No Xid line in the window: not attributed from this rental. The backend then falls back to repeats
  (the same node after different renters: provider; the same renter on different nodes: renter).

The kernel log is read on the host, never from inside the container (the executor's monitor.py already
runs dmesg the same way). The Xid split is the one the kernel-fault probe uses (SOFTWARE_XIDS in
miner_jobs/gpu_fault_probe.py); this module adds the rental window, the PID match and the ECC read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from miner_jobs.gpu_fault_probe import SOFTWARE_XIDS

WORKLOAD_XIDS: frozenset[int] = frozenset(SOFTWARE_XIDS)
# Named so the table in the event reads the same as the ticket; any Xid outside WORKLOAD_XIDS is hardware.
HARDWARE_XIDS: frozenset[int] = frozenset({48, 62, 63, 64, 74, 79, 92, 94, 95})

ATTRIBUTION_WORKLOAD = "workload"
ATTRIBUTION_HARDWARE = "hardware"
ATTRIBUTION_NONE = "none"

# `--time-format=iso` gives a wall-clock stamp per line (util-linux ≥ 2.29); `-T` is the fallback for older
# hosts. Only the NVRM lines travel, the last 200 of them: dmesg is host-wide and executor-controlled.
XID_LOG_COMMAND = (
    "(dmesg --time-format=iso 2>/dev/null || dmesg -T 2>/dev/null) | grep -F 'NVRM: Xid' | tail -n 200"
)
ECC_QUERY_COMMAND = (
    "nvidia-smi --query-gpu=pci.bus_id,uuid,ecc.errors.uncorrected.volatile.total"
    " --format=csv,noheader,nounits"
)
CONTAINER_STARTED_AT_COMMAND = "/usr/bin/docker inspect -f '{{.State.StartedAt}}' {name}"
CONTAINER_PIDS_COMMAND = "/usr/bin/docker top {name} -eo pid"
HOST_COMMAND_TIMEOUT_SECONDS = 15

MAX_LINES_KEPT = 20
MAX_LINE_CHARS = 200

# `2026-09-16T10:29:01,123456+00:00` (iso) or `[Wed Sep 16 10:29:01 2026]` (-T)
_ISO_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:,(\d{1,6}))?([+-]\d{2}:?\d{2}|Z)?")
_CTIME_STAMP = re.compile(r"^\[(\w{3}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})\]")
_XID_CODE = re.compile(r"NVRM: Xid \((?:PCI:)?([0-9a-fA-F:.]+)\): (\d+)")
_XID_PID = re.compile(r"\bpid=(\d+)")
# nvidia-smi's answers when a card is gone: the query exits non-zero and says so on stdout or stderr.
_NODE_BROKEN_MARKERS = ("Unable to determine the device handle", "GPU is lost", "Unknown Error", "has fallen off the bus")


@dataclass(frozen=True)
class XidLine:
    raw: str
    timestamp: datetime | None
    pci: str | None
    code: int | None
    pid: int | None


@dataclass(frozen=True)
class XidAttribution:
    attribution: str
    workload: list[str] = field(default_factory=list)
    hardware: list[str] = field(default_factory=list)
    outside_window: int = 0
    other_container: int = 0
    unparsed: int = 0
    ecc_uncorrected: dict[str, int] = field(default_factory=dict)

    def as_report(self) -> dict[str, Any]:
        return {
            "attribution": self.attribution,
            "workload_xids": self.workload[:MAX_LINES_KEPT],
            "hardware_xids": self.hardware[:MAX_LINES_KEPT],
            "outside_window": self.outside_window,
            "other_container": self.other_container,
            "unparsed": self.unparsed,
            "ecc_uncorrected": dict(list(self.ecc_uncorrected.items())[:MAX_LINES_KEPT]),
        }


def parse_timestamp(line: str) -> datetime | None:
    iso = _ISO_STAMP.match(line)
    if iso:
        stamp = datetime.strptime(iso.group(1), "%Y-%m-%dT%H:%M:%S")
        if iso.group(2):
            stamp = stamp.replace(microsecond=int(iso.group(2).ljust(6, "0")))
        zone = iso.group(3)
        if zone and zone != "Z":
            sign = 1 if zone[0] == "+" else -1
            digits = zone[1:].replace(":", "")
            offset = timedelta(hours=int(digits[:2]), minutes=int(digits[2:]))
            stamp = stamp - sign * offset
        return stamp.replace(tzinfo=UTC)
    ctime = _CTIME_STAMP.match(line)
    if ctime:
        # `-T` prints the host's local time without a zone; the executor image runs on UTC, as the
        # validator's own clock does, so the stamp is read as UTC.
        return datetime.strptime(ctime.group(1), "%a %b %d %H:%M:%S %Y").replace(tzinfo=UTC)
    return None


def parse_xid_lines(text: str) -> list[XidLine]:
    lines: list[XidLine] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if "NVRM: Xid" not in raw:
            continue
        code_match = _XID_CODE.search(raw)
        pid_match = _XID_PID.search(raw)
        lines.append(
            XidLine(
                raw=raw[:MAX_LINE_CHARS],
                timestamp=parse_timestamp(raw),
                pci=code_match.group(1).lower() if code_match else None,
                code=int(code_match.group(2)) if code_match else None,
                pid=int(pid_match.group(1)) if pid_match else None,
            )
        )
    return lines


def parse_container_pids(text: str) -> set[int]:
    pids: set[int] = set()
    for token in text.split():
        if token.isdigit():
            pids.add(int(token))
    return pids


def parse_docker_started_at(text: str) -> datetime | None:
    """`docker inspect -f '{{.State.StartedAt}}'`: RFC 3339 with nanoseconds, `Z` zone."""
    value = text.strip()
    if not value or value.startswith("0001-01-01"):
        return None
    match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})$", value)
    if not match:
        return None
    stamp = datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S")
    if match.group(2):
        stamp = stamp.replace(microsecond=int(match.group(2)[:6].ljust(6, "0")))
    zone = match.group(3)
    if zone != "Z":
        sign = 1 if zone[0] == "+" else -1
        hours, minutes = zone[1:].split(":")
        stamp = stamp - sign * timedelta(hours=int(hours), minutes=int(minutes))
    return stamp.replace(tzinfo=UTC)


def parse_ecc_uncorrected(text: str) -> dict[str, int]:
    """`pci.bus_id, uuid, ecc.errors.uncorrected.volatile.total` rows -> {uuid: count} for counts above 0.

    `[N/A]` (ECC off, consumer cards) and unparsable rows are skipped: no reading is not an error.
    """
    counts: dict[str, int] = {}
    for row in text.strip().splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 3 or not parts[1].startswith("GPU-"):
            continue
        try:
            count = int(parts[2])
        except ValueError:
            continue
        if count > 0:
            counts[parts[1]] = count
    return counts


def node_answers(exit_code: int, stdout: str, stderr: str) -> bool:
    """True when nvidia-smi listed the cards; False is what a card that fell off the bus looks like."""
    if exit_code != 0:
        return False
    text = f"{stdout}\n{stderr}"
    return not any(marker in text for marker in _NODE_BROKEN_MARKERS)


def attribute(
    lines: list[XidLine],
    *,
    window_start: datetime | None,
    window_end: datetime,
    container_pids: set[int] | None = None,
    ecc_uncorrected: dict[str, int] | None = None,
) -> XidAttribution:
    """Split the rental window's Xid lines into workload and hardware and name the verdict.

    A line without a readable timestamp, or outside [window_start, window_end], is not this rental's.
    When the renter's container PIDs are known (mid-rental) a workload Xid that names a PID outside them is
    another container's and is not counted; at rental end the container is gone, so the timestamp alone
    places the line. Hardware beats workload: one hardware Xid or an uncorrected ECC count makes the
    verdict "hardware" whatever else the renter's process logged. No window start means no verdict.
    """
    workload: list[str] = []
    hardware: list[str] = []
    outside = other = unparsed = 0
    for line in lines:
        if line.code is None:
            unparsed += 1
            continue
        if window_start is None or line.timestamp is None or not (window_start <= line.timestamp <= window_end):
            outside += 1
            continue
        if line.code in WORKLOAD_XIDS:
            if container_pids is not None and line.pid is not None and line.pid not in container_pids:
                other += 1
                continue
            workload.append(line.raw)
        else:
            hardware.append(line.raw)
    ecc = dict(ecc_uncorrected or {})
    if hardware or ecc:
        verdict = ATTRIBUTION_HARDWARE
    elif workload:
        verdict = ATTRIBUTION_WORKLOAD
    else:
        verdict = ATTRIBUTION_NONE
    return XidAttribution(
        attribution=verdict,
        workload=workload,
        hardware=hardware,
        outside_window=outside,
        other_container=other,
        unparsed=unparsed,
        ecc_uncorrected=ecc,
    )
