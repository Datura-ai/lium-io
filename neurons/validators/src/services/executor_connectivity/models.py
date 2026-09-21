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
    """Why the DinD container's sshd never answered: a stable code and the words the provider reads."""

    code: str
    message: str

    @property
    def text(self) -> str:
        """`CODE: message`, the one-line form for logs and the event's what-we-saw."""
        return f"{self.code}: {self.message}"


@dataclass(frozen=True)
class DindProbeResult:
    success: bool
    sysbox_runtime: bool
    port: PortPair | None
    log_text: str | None = None
    # DAH-2856: the cause when the container started but sshd never answered, read from the
    # container's own logs before removal (None when the probe passed or never got that far).
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


@dataclass(frozen=True)
class ContainerStartResult:
    ok: bool
    container_id: str | None
    status: str | None
    logs: str | None = None
