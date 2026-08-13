"""What CVM this node is running, persisted beside the state document.

Kept out of `state.json` on purpose. The state document is a small, fully-pinned contract that
DAH-2575 shipped and its tests assert byte for byte; the instance record is a bag of facts that
will keep growing as DAH-2577 adds teardown timings and DAH-2580 adds renter metadata. Two
files with separate lifetimes cost one extra write and stop every future field from being a
change to a contract other components read.

The record is the node's answer to "what is running here". It is written before the supervisor
is spawned — a record with no process is a recoverable inconsistency, while a process with no
record is an orphan cvmd cannot find, stop, or report.
"""

from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from cvmd.atomic import read_json, write_json_durable

INSTANCE_FILENAME = "instance.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PortReport:
    protocol: str
    address: str
    host_port: int
    guest_port: int


@dataclass(frozen=True)
class Instance:
    """One CVM. `ssh_fingerprint` is None until the guest is up and has been probed."""

    instance_id: str
    kind: str
    artifact_id: str
    vm_dir: str
    supervisor_pid: int
    created_at: str
    qemu: str
    os_image_hash: str
    compose_hash: str
    ports: list[PortReport] = field(default_factory=list)
    ssh_fingerprint: str | None = None
    # DAH-2580, renter CVMs only: what the platform called the rental this CVM belongs to,
    # carried so an operator can match a running guest back to it. Defaults to None, which is
    # what a validation CVM has and what a record written before this task has — so an older
    # instance.json still decodes rather than reading as corrupt and failing the node.
    rental_id: str | None = None
    # DAH-2679: how many times reconciliation has re-spawned this CVM's supervisor over the
    # same directory (host reboot, QEMU exit). Cumulative for the instance's whole life, never
    # reset — reconciliation has no "the guest is healthy again" signal it could reset on, and
    # a cap on the total is what stops a dying guest from crash-looping through every cvmd
    # restart. Defaults to 0 so an older instance.json still decodes.
    relaunch_attempts: int = 0

    def to_json(self) -> dict:
        return {"version": SCHEMA_VERSION, **asdict(self)}

    def report(self) -> dict:
        """The launch report the API returns: identity, reachability, and what it measures as."""
        report = {
            "instance_id": self.instance_id,
            "kind": self.kind,
            "artifact_id": self.artifact_id,
            "ports": [asdict(port) for port in self.ports],
            "ssh_host_key_fingerprint": self.ssh_fingerprint,
            "measurements": {
                "qemu": self.qemu,
                "os_image_hash": self.os_image_hash,
                "compose_hash": self.compose_hash,
            },
        }
        # Only present on a renter CVM, so a validation report keeps the exact shape DAH-2576
        # shipped and its tests pin.
        if self.rental_id is not None:
            report["rental_id"] = self.rental_id
        # Same reasoning: only present once a relaunch has actually happened, so every report
        # for a CVM that booted once and stayed up keeps its pinned shape.
        if self.relaunch_attempts:
            report["relaunch_attempts"] = self.relaunch_attempts
        return report


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _decode(raw) -> Instance | None:
    if not isinstance(raw, dict) or raw.get("version") != SCHEMA_VERSION:
        return None
    try:
        ports = [PortReport(**port) for port in raw.get("ports", [])]
        return Instance(
            instance_id=str(raw["instance_id"]),
            kind=str(raw["kind"]),
            artifact_id=str(raw["artifact_id"]),
            vm_dir=str(raw["vm_dir"]),
            supervisor_pid=int(raw["supervisor_pid"]),
            created_at=str(raw["created_at"]),
            qemu=str(raw["qemu"]),
            os_image_hash=str(raw["os_image_hash"]),
            compose_hash=str(raw["compose_hash"]),
            ports=ports,
            ssh_fingerprint=raw.get("ssh_fingerprint"),
            rental_id=raw.get("rental_id"),
            relaunch_attempts=int(raw.get("relaunch_attempts") or 0),
        )
    except (KeyError, TypeError, ValueError):
        return None


class InstanceStore:
    """Holds the current instance record, or None when the node is running no CVM.

    A record that cannot be decoded is treated as **absent but noted**, never as a fresh node:
    the caller gets None and `load_error` explains why, so an unreadable file surfaces as a
    refusal to launch over the top of something rather than as a clean slate.
    """

    def __init__(self, state_dir: Path) -> None:
        self._path = state_dir / INSTANCE_FILENAME
        self._instance: Instance | None = None
        self.load_error: str | None = None

        if self._path.exists():
            decoded = _decode(read_json(self._path))
            if decoded is None:
                self.load_error = f"{self._path} is unreadable or malformed"
            self._instance = decoded

    @property
    def current(self) -> Instance | None:
        return self._instance

    def set(self, instance: Instance) -> Instance:
        self._instance = instance
        write_json_durable(self._path, instance.to_json())
        self.load_error = None
        return instance

    def update(self, **changes) -> Instance:
        if self._instance is None:
            raise ValueError("no instance to update")
        return self.set(replace(self._instance, **changes))

    def clear(self) -> None:
        self._instance = None
        self.load_error = None
        self._path.unlink(missing_ok=True)
