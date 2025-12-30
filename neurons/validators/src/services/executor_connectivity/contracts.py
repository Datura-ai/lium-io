from __future__ import annotations

from typing import Protocol

from services.executor_connectivity.models import (
    DindProbeResult,
    PortPair,
    PortProbeResult,
    PortVerificationResult,
)


class PortSelectorProtocol(Protocol):
    def select(self, executor_info, size: int, rented: set[int]) -> list[PortPair]:
        ...


class PortProbeProtocol(Protocol):
    async def probe(
        self,
        ports: list[PortPair],
        *,
        ssh_client,
        host: str,
    ) -> PortProbeResult:
        ...


class DindProbeProtocol(Protocol):
    async def verify(
        self,
        port: PortPair,
        *,
        ssh_client,
        host: str,
        container_name_prefix: str,
        sysbox_runtime: bool,
    ) -> DindProbeResult:
        ...


class ResultPersisterProtocol(Protocol):
    async def save(
        self,
        result: PortVerificationResult,
        executor_id: str,
        miner_hotkey: str,
    ) -> None:
        ...


class OrchestratorProtocol(Protocol):
    async def verify(
        self,
        *,
        ssh_client,
        executor_info,
        miner_hotkey: str,
        sysbox_runtime: bool,
        rented_ports: list[int] | None,
    ) -> PortVerificationResult:
        ...


class ContainerCleanupProtocol(Protocol):
    async def cleanup(
        self,
        ssh,
        active_pod_names: list[str],
        container_name_prefix: str,
    ) -> None:
        ...
