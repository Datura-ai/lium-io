import logging
import time

import asyncssh
from datura.requests.miner_requests import ExecutorSSHInfo
from daos.port_mapping_dao import PortMappingDao

from services.executor_connectivity.cleanup_service import ContainerCleanupService
from services.executor_connectivity.contracts import (
    ContainerCleanupProtocol,
    OrchestratorProtocol,
    ResultPersisterProtocol,
)
from services.executor_connectivity.models import PortVerificationResult
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.persister import PortResultPersister
from services.redis_service import RedisService
from services.ssh_service import SSHService

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN SERVICE - Thin Orchestrator
# ============================================================================

class ExecutorConnectivityService:
    """Orchestrates port verification workflow."""

    def __init__(
        self,
        redis_service: "RedisService",
        port_mapping_dao: PortMappingDao,
        ssh_service: SSHService,
        orchestrator: OrchestratorProtocol,
        persister: ResultPersisterProtocol,
        cleanup_service: ContainerCleanupProtocol,
    ):
        self.redis = redis_service
        self.dao = port_mapping_dao
        self.ssh_service = ssh_service
        self.orchestrator = orchestrator
        self.persister = persister
        self.cleanup_service = cleanup_service

    async def verify_ports(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo,
        sysbox_runtime: bool = False,
        rented_ports: list[int] | None = None,
        rented_pod_names: list[str] | None = None,
    ) -> PortVerificationResult:
        """Verify executor port connectivity and DinD capability."""
        t1 = time.monotonic()
        try:
            await self.cleanup_service.cleanup(
                ssh_client,
                rented_pod_names or [],
                "container_",
            )

            verification = await self.orchestrator.verify(
                ssh_client=ssh_client,
                executor_info=executor_info,
                miner_hotkey=miner_hotkey,
                sysbox_runtime=sysbox_runtime,
                rented_ports=rented_ports,
            )
            result = PortVerificationResult(
                selected_ports=verification.selected_ports,
                successful_ports=verification.successful_ports,
                failed_ports=verification.failed_ports,
                dind_port=verification.dind_port,
                dind_ok=verification.dind_ok,
                sysbox_runtime=verification.sysbox_runtime,
                status=verification.status,
                error=verification.error,
                elapsed_sec=time.monotonic() - t1,
            )

            if result.status == "ok":
                await self.persister.save(result, executor_info.uuid, miner_hotkey)
            return result
        except Exception as e:
            logger.error(
                "verification failed: %s executor=%s",
                str(e),
                executor_info.address,
                exc_info=True,
            )
            return PortVerificationResult(
                selected_ports=tuple(),
                successful_ports=tuple(),
                failed_ports=tuple(),
                dind_port=None,
                dind_ok=False,
                sysbox_runtime=sysbox_runtime,
                status="error",
                error=str(e),
                elapsed_sec=time.monotonic() - t1,
            )
