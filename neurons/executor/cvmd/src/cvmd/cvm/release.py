"""Is this node's hardware actually free? Four conditions, each measured on the host.

A stopped CVM is not a free node. `lium-cvm.sh stop` reports success as soon as its wait loop
gives up, and the fleet's standing bug is exactly that: the shell says the CVM is gone while
QEMU is still holding the guest's RAM. Confirming the process group is gone — all DAH-2576
could do — is necessary and nowhere near sufficient, because the expensive part of a TDX
teardown happens *after* the process exits.

So "destroyed" is a predicate over host facts, not over cvmd's own records:

    1. process_reaped   the process group has no members left, zombies included, and no TDX
                        guest is running anywhere on this host
    2. vfio_released    no process holds a descriptor on /dev/vfio, so the GPUs can be claimed
                        by the next guest
    3. memory_returned  the host has the guest's RAM back, and its hugepages are back in the pool
    4. ports_free       every port this CVM forwarded can be bound again

Each is polled until it holds or the budget runs out; the first moment each one holds is
recorded, which is what turns "the switch took 41s" into "the process went at 12s and the
memory took another 29". That per-condition breakdown is the point — the switch window is what
FR-I3 has to budget for, and a single total tells an operator nothing about which hardware
class is slow.

Reading `/proc` is parameterised so the tests can point these functions at a fixture tree
instead of the machine running them. Everything else here reads the real host.
"""

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cvmd.cvm import ports, supervisor

logger = logging.getLogger(__name__)

PROC = Path("/proc")
MEMINFO = Path("/proc/meminfo")
VFIO_PREFIX = "/dev/vfio/"

# How often the four conditions are re-evaluated. The first evaluation happens before any
# sleep, so a node that is already free answers immediately; this only paces the waiting.
POLL_SECONDS = 2

PROCESS_REAPED = "process_reaped"
VFIO_RELEASED = "vfio_released"
MEMORY_RETURNED = "memory_returned"
PORTS_FREE = "ports_free"

CONDITION_NAMES = (PROCESS_REAPED, VFIO_RELEASED, MEMORY_RETURNED, PORTS_FREE)

# `dstack.py:_convert_memory_to_mb`. Mirrored rather than imported: dstack.py is loaded from a
# host path that may not exist, and this has to work on a node whose launcher is missing.
_MEMORY_SUFFIXES = {"T": 1024 * 1024, "G": 1024, "M": 1}


class MemoryUnreadable(Exception):
    """/proc/meminfo could not be read or did not carry the field asked for."""


def _gib(kib: int) -> str:
    return f"{kib / 1024 / 1024:.1f} GiB"


@dataclass(frozen=True)
class Condition:
    """One condition, evaluated once. `detail` says what was observed, satisfied or not."""

    name: str
    satisfied: bool
    detail: str


@dataclass
class Timing:
    """When a condition first held, and what was last observed about it."""

    satisfied_after_seconds: float | None = None
    detail: str = "not evaluated"

    def to_json(self) -> dict:
        return {
            "satisfied_after_seconds": self.satisfied_after_seconds,
            "detail": self.detail,
        }


@dataclass
class ReleaseReport:
    """The outcome of one verification, with per-condition timings."""

    complete: bool = False
    duration_seconds: float = 0.0
    timings: dict[str, Timing] = field(default_factory=dict)

    @property
    def unsatisfied(self) -> list[str]:
        return [name for name, t in self.timings.items() if t.satisfied_after_seconds is None]

    def why_incomplete(self) -> str:
        """The failing conditions, named, with what was observed about each."""
        return "; ".join(f"{name}: {self.timings[name].detail}" for name in self.unsatisfied)

    def to_json(self) -> dict:
        return {
            "complete": self.complete,
            "duration_seconds": round(self.duration_seconds, 1),
            "conditions": {name: t.to_json() for name, t in self.timings.items()},
        }


# ------------------------------------------------------------------ host facts


def _stat(proc: Path, pid: int) -> tuple[str, str, int] | None:
    """(comm, state, pgrp) for `pid`, or None if it is gone.

    Read from `stat` rather than `cmdline` because a zombie has an empty `cmdline` — which is
    the whole reason this file exists. The command name is parenthesised and may itself contain
    spaces and parentheses, so the fields after it are found from the LAST `)`.
    """
    try:
        raw = (proc / str(pid) / "stat").read_text()
    except OSError:
        return None
    opened, closed = raw.find("("), raw.rfind(")")
    if opened == -1 or closed == -1 or closed < opened:
        return None
    fields = raw[closed + 1 :].split()
    if len(fields) < 3:
        return None
    try:
        return raw[opened + 1 : closed], fields[0], int(fields[2])
    except ValueError:
        return None


def _pids(proc: Path) -> list[int]:
    if not proc.is_dir():
        return []
    return sorted(int(e.name) for e in proc.iterdir() if e.name.isdigit())


def group_members(pgid: int, *, proc: Path = PROC) -> list[str]:
    """Every process still in process group `pgid`, described, zombies included.

    A zombie is why this is not `os.killpg(pgid, 0)`. It is still a group member, so killpg
    keeps succeeding against a process that has already exited and holds nothing — measured on
    au11, where it made every teardown burn the full signal ladder and then report a failure
    that had not happened. Naming the state is what lets a reader tell "the guest is still
    running" from "something never reaped its child".
    """
    members = []
    for pid in _pids(proc):
        entry = _stat(proc, pid)
        if entry is None:
            continue
        comm, state, group = entry
        if group == pgid:
            members.append(f"pid {pid} ({comm}{', zombie' if state == 'Z' else ''})")
    return members


def vfio_holders(*, proc: Path = PROC) -> tuple[list[str], list[int]]:
    """Who holds a descriptor under /dev/vfio, and whose descriptors could not be read.

    The open descriptor is the binding that matters. A GPU stays bound to `vfio-pci` across a
    teardown by design — that is the day-zero setup, not a leak — so the driver a device sits
    on says nothing about whether the next guest can claim it. An open group descriptor says
    exactly that, and it is released by the kernel only once the holder is fully gone.

    Descriptors this process may not read are returned separately rather than ignored: cvmd
    runs as root, so that list is empty in production, and treating "could not look" as "found
    nothing" would be the one direction this check must not fail in.
    """
    holders: list[str] = []
    unreadable: list[int] = []
    for pid in _pids(proc):
        fd_dir = proc / str(pid) / "fd"
        try:
            entries = list(fd_dir.iterdir())
        except PermissionError:
            unreadable.append(pid)
            continue
        except OSError:
            # Gone between listing /proc and reading it. Nothing that exited is holding a
            # descriptor.
            continue
        for entry in entries:
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            if target.startswith(VFIO_PREFIX):
                comm = _stat(proc, pid)
                holders.append(f"pid {pid} ({comm[0] if comm else '?'}) holds {target}")
                break
    return holders, unreadable


def meminfo(*, path: Path = MEMINFO) -> dict[str, int]:
    """`/proc/meminfo` as a name -> kibibytes mapping. Lines without a kB value are skipped."""
    try:
        raw = path.read_text()
    except OSError as exc:
        raise MemoryUnreadable(f"cannot read {path}: {exc}") from exc

    values: dict[str, int] = {}
    for line in raw.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            continue
        try:
            values[name.strip()] = int(parts[0])
        except ValueError:
            continue
    return values


def mem_available_kib(*, path: Path = MEMINFO) -> int:
    values = meminfo(path=path)
    if "MemAvailable" not in values:
        raise MemoryUnreadable(f"{path} carries no MemAvailable")
    return values["MemAvailable"]


def memory_kib(spec: str) -> int:
    """A dstack memory string — "2G", "512M", "1T", or a bare number of mebibytes — in KiB.

    Mirrors `DStackManager._convert_memory_to_mb`, which is what turned this string into the
    number QEMU was started with. Reading it any other way would compare the host's recovery
    against a size the guest never had.
    """
    text = spec.strip()
    factor = _MEMORY_SUFFIXES.get(text[-1:].upper())
    number = text[:-1] if factor else text
    try:
        return int(number) * (factor or 1) * 1024
    except ValueError as exc:
        raise MemoryUnreadable(f"{spec!r} is not a memory size dstack would accept") from exc


# ------------------------------------------------------------------ the four conditions


@dataclass(frozen=True)
class ReleaseChecks:
    """Everything the four conditions need, bound to one CVM that has been asked to stop.

    `baseline_mem_available_kib` is sampled while the guest is still running, so the memory
    condition can be answered two ways (see `memory_returned`). It is optional because a cvmd
    that restarted mid-teardown has no baseline and must still be able to finish the job.
    """

    supervisor_pid: int
    mappings: list[ports.PortMapping]
    guest_memory_kib: int | None = None
    baseline_mem_available_kib: int | None = None
    memory_tolerance: float = 0.9
    hugepages: bool = False
    proc: Path = PROC
    meminfo_path: Path = MEMINFO

    def process_reaped(self) -> Condition:
        members = group_members(self.supervisor_pid, proc=self.proc)
        if members:
            return Condition(
                PROCESS_REAPED,
                False,
                f"process group {self.supervisor_pid} still has {', '.join(members)}",
            )
        # Host-wide, not just this group: a guest that escaped its group — or one `lium-cvm.sh`
        # started — holds the same GPUs and the same TDX capacity, so the node is not free
        # while it runs.
        running = supervisor.running_cvms()
        if running:
            return Condition(
                PROCESS_REAPED,
                False,
                "a confidential guest is still running on this host: "
                + ", ".join(f"pid {pid} ({name})" for pid, name in running),
            )
        return Condition(PROCESS_REAPED, True, "no process group members and no TDX guest")

    def vfio_released(self) -> Condition:
        holders, unreadable = vfio_holders(proc=self.proc)
        if holders:
            return Condition(VFIO_RELEASED, False, "; ".join(holders))
        if unreadable:
            return Condition(
                VFIO_RELEASED,
                False,
                f"{len(unreadable)} process(es) would not let their open descriptors be read "
                f"(first: pid {unreadable[0]}), so no VFIO group can be proven closed",
            )
        return Condition(VFIO_RELEASED, True, "nothing holds a /dev/vfio descriptor")

    def memory_returned(self) -> Condition:
        """Has the host got the guest's RAM back, and its hugepages?

        Answered two ways, either of which is enough, because neither alone is honest on every
        host:

        * **absolute** — MemAvailable is at least the size the guest was configured with, so
          this node could host that CVM again. This is the operational meaning of "the memory
          is free", and it is the one that catches the slow case: a 1.13 TB guest's memory
          comes back to the host over tens of minutes after QEMU has exited.
        * **relative** — MemAvailable has risen by the guest's size since teardown began. This
          is what answers on a host that is legitimately short of memory for other reasons: the
          guest gave back what it held, which is all a teardown can be responsible for.

        A relative-only rule would fail a perfectly clean teardown, because a guest that never
        faulted in its full allocation has less to give back than it was configured with — and
        under TDX the private memory is not in QEMU's RSS, so there is no way to ask how much
        it really held.
        """
        try:
            values = meminfo(path=self.meminfo_path)
        except MemoryUnreadable as exc:
            return Condition(MEMORY_RETURNED, False, str(exc))

        hugepages = self._hugepages(values)
        if hugepages is not None:
            return hugepages

        available = values.get("MemAvailable")
        if available is None:
            return Condition(MEMORY_RETURNED, False, f"{self.meminfo_path} carries no MemAvailable")
        if self.guest_memory_kib is None:
            # Nothing to compare against. The process is what held the memory, and that is
            # condition 1's to answer.
            return Condition(
                MEMORY_RETURNED, True, "no guest memory size is configured, so nothing to reclaim"
            )

        want = int(self.guest_memory_kib * self.memory_tolerance)
        detail = f"MemAvailable {_gib(available)} of the {_gib(want)} this guest needs"
        recovered = None
        if self.baseline_mem_available_kib is not None:
            recovered = available - self.baseline_mem_available_kib
            detail += f", up {_gib(recovered)} since teardown began"

        satisfied = available >= want or (recovered is not None and recovered >= want)
        return Condition(MEMORY_RETURNED, satisfied, detail)

    def _hugepages(self, values: dict[str, int]) -> Condition | None:
        """The unsatisfied hugepage condition, or None when there is nothing to wait for.

        Only meaningful on a host configured to back its guests with hugepages: elsewhere the
        pool is empty and `HugePages_Free == HugePages_Total == 0` says nothing.
        """
        if not self.hugepages:
            return None
        total = values.get("HugePages_Total", 0)
        free = values.get("HugePages_Free", 0)
        if total == 0 or free >= total:
            return None
        return Condition(
            MEMORY_RETURNED,
            False,
            f"{total - free} of {total} hugepages are still out of the pool",
        )

    def ports_free(self) -> Condition:
        if not self.mappings:
            return Condition(PORTS_FREE, True, "this CVM forwarded no ports")
        try:
            ports.assert_free(self.mappings)
        except ports.PortError as exc:
            return Condition(PORTS_FREE, False, str(exc))
        return Condition(
            PORTS_FREE, True, f"all {len(self.mappings)} forwarded port(s) can be bound again"
        )

    def evaluate(self) -> list[Condition]:
        """All four, in the order DAH-2577 numbers them. Every one is evaluated every poll.

        None of them short-circuits on an earlier failure: a condition that is already
        satisfied while another is still pending has a real first-satisfied moment, and losing
        it would make the per-condition timings useless for exactly the hosts they exist for.
        """
        return [
            self.process_reaped(),
            self.vfio_released(),
            self.memory_returned(),
            self.ports_free(),
        ]


# ------------------------------------------------------------------ the wait


def verify_released(
    checks: ReleaseChecks,
    *,
    timeout: float,
    poll: float = POLL_SECONDS,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ReleaseReport:
    """Poll the four conditions until all hold together, or the budget runs out.

    "Together" is the whole point. A condition that held ten seconds ago is not evidence now —
    a port can be re-bound by something else, and a guest can still be running while its
    memory reads as free — so completion requires one evaluation in which all four are true.
    The first-satisfied timings are still recorded per condition, because they are what the
    per-hardware-class budget is measured from.

    Returns rather than raises: a teardown that did not complete is a report the caller has to
    put in the node's state and in its answer, not an exception to interpret.
    """
    started = now()
    deadline = started + timeout
    report = ReleaseReport(timings={name: Timing() for name in CONDITION_NAMES})

    while True:
        conditions = checks.evaluate()
        elapsed = now() - started
        for condition in conditions:
            timing = report.timings[condition.name]
            timing.detail = condition.detail
            if condition.satisfied and timing.satisfied_after_seconds is None:
                timing.satisfied_after_seconds = round(elapsed, 1)
            elif not condition.satisfied:
                # It held earlier and does not now, so the earlier moment was not release. The
                # report has to describe the host as it is at the end, not at its best moment.
                timing.satisfied_after_seconds = None

        report.duration_seconds = elapsed
        if all(condition.satisfied for condition in conditions):
            report.complete = True
            return report

        if now() >= deadline:
            logger.warning(
                "the node was not released in %.0fs: %s", elapsed, report.why_incomplete()
            )
            return report
        sleep(min(poll, max(deadline - now(), 0)))
