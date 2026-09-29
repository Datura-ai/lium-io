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
    # the tiers that produced `successful`, joined with "+" when a top-up merged two
    tier: str = "batch"


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
    probe_tier: str | None = None
    # len(get_all_ports(...)) for the executor's declared range or mappings; None when they could not be parsed
    declared_port_count: int | None = None


@dataclass(frozen=True)
class ContainerStartResult:
    ok: bool
    container_id: str | None
    status: str | None
    logs: str | None = None
