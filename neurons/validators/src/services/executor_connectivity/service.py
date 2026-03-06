import logging
import time

import asyncssh
from datura.requests.miner_requests import ExecutorSSHInfo
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.models import PortVerificationResult

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN SERVICE - Thin Orchestrator
# ============================================================================

class ExecutorConnectivityService:
    """Orchestrates port verification workflow."""

    def __init__(
        self,
        orchestrator: ConnectivityOrchestrator,
    ):
        self.orchestrator = orchestrator

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
            # Cleanup removed - test containers are ephemeral and short-lived anyway
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
