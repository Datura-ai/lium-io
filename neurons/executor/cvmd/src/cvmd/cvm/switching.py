"""The switch record: what this node's last mode change cost, and whether one is still running.

A node spends its life crossing between validation and rented (FR-C5), and every crossing means
destroying one CVM before launching the next. That crossing is not instant — on a large-memory
guest it is the slowest thing the node does — so it is an explicit window with its own state
(`SWITCHING`) and its own measured duration.

The record is written twice: once when a teardown begins, and once when the four conditions in
`cvm/release.py` settle. The first write is what makes the window survivable. cvmd is restartable
by design, and a daemon that came back mid-teardown with no record would have to choose between
declaring a node free it had not checked and failing one that was fine — so the facts the
conditions need (which process, which ports, how much memory the guest had, and what the host had
free before it stopped) are on disk before the first signal is sent.

The second write is the deliverable: per-condition timings for FR-I3's availability accounting,
which needs to know how long a node is legitimately unreachable before it starts counting as
offline. Kept out of `state.json`, whose small schema DAH-2575 pinned and its tests assert field
by field, and out of `instance.json`, which is deleted by the very operation being measured.
"""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from cvmd.atomic import read_json, write_json_durable
from cvmd.cvm.instance import PortReport, now_iso

SWITCH_FILENAME = "switch.json"
SCHEMA_VERSION = 1

RELEASED = "released"
TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class SwitchRecord:
    """One teardown. `outcome` is None while it is still running."""

    instance_id: str
    kind: str
    vm_dir: str
    supervisor_pid: int
    started_at: str
    guest_memory_kib: int | None = None
    baseline_mem_available_kib: int | None = None
    ports: list[PortReport] = field(default_factory=list)
    completed_at: str | None = None
    outcome: str | None = None
    detail: str | None = None
    release: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"version": SCHEMA_VERSION, **asdict(self)}

    def report(self) -> dict:
        """What `/v1/state` and the teardown response say about the switch window."""
        return {
            "instance_id": self.instance_id,
            "kind": self.kind,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "outcome": self.outcome,
            "detail": self.detail,
            **self.release,
        }


def _decode(raw) -> SwitchRecord | None:
    if not isinstance(raw, dict) or raw.get("version") != SCHEMA_VERSION:
        return None
    try:
        return SwitchRecord(
            instance_id=str(raw["instance_id"]),
            kind=str(raw["kind"]),
            vm_dir=str(raw["vm_dir"]),
            supervisor_pid=int(raw["supervisor_pid"]),
            started_at=str(raw["started_at"]),
            guest_memory_kib=raw.get("guest_memory_kib"),
            baseline_mem_available_kib=raw.get("baseline_mem_available_kib"),
            ports=[PortReport(**port) for port in raw.get("ports", [])],
            completed_at=raw.get("completed_at"),
            outcome=raw.get("outcome"),
            detail=raw.get("detail"),
            release=raw.get("release") or {},
        )
    except (KeyError, TypeError, ValueError):
        return None


class SwitchStore:
    """Holds the last switch, running or finished.

    Unlike the instance record this is never cleared. A finished switch is the node's only
    account of how long its last crossing took, and the next teardown overwrites it — so there
    is exactly one, always the most recent.

    An undecodable file is treated as absent. Losing the timings of a past switch costs a
    measurement; refusing to start over it would cost the node.
    """

    def __init__(self, state_dir: Path) -> None:
        self._path = state_dir / SWITCH_FILENAME
        self._record = _decode(read_json(self._path)) if self._path.exists() else None

    @property
    def current(self) -> SwitchRecord | None:
        return self._record

    @property
    def unfinished(self) -> SwitchRecord | None:
        """The switch this node still owes, if it owes one.

        Anything but `released` counts: a teardown that timed out has not ended the window it
        opened — the node is still holding hardware the platform asked it to give back — and a
        record with no outcome at all is a cvmd that died mid-teardown. Both are continued by
        the next teardown rather than replaced by it.
        """
        record = self._record
        return None if record is None or record.outcome == RELEASED else record

    def begin(self, record: SwitchRecord) -> SwitchRecord:
        return self._write(record)

    def finish(self, **changes) -> SwitchRecord:
        if self._record is None:
            raise ValueError("no switch to finish")
        return self._write(replace(self._record, completed_at=now_iso(), **changes))

    def _write(self, record: SwitchRecord) -> SwitchRecord:
        self._record = record
        write_json_durable(self._path, record.to_json())
        return record
