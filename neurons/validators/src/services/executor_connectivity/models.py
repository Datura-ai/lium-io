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
class DindProbeResult:
    success: bool
    sysbox_runtime: bool
    port: PortPair | None
    log_text: str | None = None


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


@dataclass(frozen=True)
class ContainerStartResult:
    ok: bool
    container_id: str | None
    status: str | None
    logs: str | None = None
