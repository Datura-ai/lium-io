from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortPair:
    """Immutable port mapping."""
    internal: int
    external: int


@dataclass(frozen=True)
class PortProbeResult:
    successful: tuple[PortPair, ...]
    failed: tuple[PortPair, ...]


@dataclass(frozen=True)
class DindLogCause:
    """Why the DinD probe failed — the container's sshd never answered, or `docker run` itself was refused
    by the NVIDIA container hook: a stable code and the words the provider reads."""

    code: str
    message: str
    # dockerd's own line when the cause was read from the container log. None when the code was
    # named without one, so the event must not claim the fix is on the host and not in sysbox.
    dockerd_line: str | None = None

    @property
    def text(self) -> str:
        """`CODE: message`, the one-line form for logs and the event's what-we-saw."""
        return f"{self.code}: {self.message}"


# DAH-2856: the cause codes read from the inner dockerd's own log. The fix for these is on the
# host, sysbox is not the cause; the sysbox check reads the set to word its remediation.
DIND_INNER_DOCKERD_IPTABLES = "DIND_INNER_DOCKERD_IPTABLES"
DIND_INNER_DOCKERD_DOWN = "DIND_INNER_DOCKERD_DOWN"
DIND_INNER_DOCKERD_CODES = frozenset({DIND_INNER_DOCKERD_IPTABLES, DIND_INNER_DOCKERD_DOWN})


@dataclass(frozen=True)
class DindProbeResult:
    success: bool
    sysbox_runtime: bool
    port: PortPair | None
    log_text: str | None = None
    # DAH-2856: the cause when the container started but sshd never answered, read from the
    # container's own logs before removal. DAH-3634: also when `docker run` itself was refused by
    # the NVIDIA container hook (docker's stderr). None when the probe passed or the cause is unknown.
    error: DindLogCause | None = None


@dataclass(frozen=True)
class PortRangeResult:
    """One declared port range's tally for a cycle: how many ports it declares, how many were probed
    and how many answered, so a partial forward shows where it sits."""

    first: int
    last: int
    declared: int
    probed: int
    answered: int
    # the buckets past PORT_RANGE_MAX_ENTRIES, summed into one
    other: bool = False

    def as_dict(self) -> dict[str, object]:
        label = str(self.first) if self.first == self.last else f"{self.first}-{self.last}"
        if self.other:
            label = f"other {label}"
        return {"range": label, "declared": self.declared, "probed": self.probed, "answered": self.answered}


@dataclass(frozen=True)
class PortVerificationResult:
    selected_ports: tuple[PortPair, ...]
    successful_ports: tuple[PortPair, ...]
    failed_ports: tuple[PortPair, ...]
    dind_port: PortPair | None
    dind_ok: bool
    sysbox_runtime: bool
    status: str
    error: str | None = None
    elapsed_sec: float | None = None
    # DAH-2856: DindProbeResult.error carried through, so the sysbox verdict can name the real cause.
    dind_error: DindLogCause | None = None
    # One tally per declared range (split at PORT_RANGE_BUCKET_WIDTH boundaries), ascending.
    port_ranges: tuple[PortRangeResult, ...] = ()


@dataclass(frozen=True)
class ContainerStartResult:
    ok: bool
    container_id: str | None
    status: str | None
    logs: str | None = None
