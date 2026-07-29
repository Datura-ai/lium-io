import logging
import time

import asyncssh
from core.utils import _m, get_extra_info
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
        excluded_ports: list[int] | None = None,
        log_ctx: dict | None = None,
    ) -> PortVerificationResult:
        """Verify executor port connectivity and DinD capability."""
        log_ctx = log_ctx or {}
        t1 = time.monotonic()
        try:
            # Cleanup removed - test containers are ephemeral and short-lived anyway
            verification = await self.orchestrator.verify(
                ssh_client=ssh_client,
                executor_info=executor_info,
                miner_hotkey=miner_hotkey,
                sysbox_runtime=sysbox_runtime,
                rented_ports=rented_ports,
                excluded_ports=excluded_ports,
                log_ctx=log_ctx,
            )
            sysbox_result = verification.sysbox_runtime
            if not sysbox_result and rented_ports and sysbox_runtime:
                logger.info(
                    _m(
                        "Sysbox runtime fallback: using known value for rented executor",
                        extra=get_extra_info({
                            **log_ctx,
                            "miner_hotkey": miner_hotkey,
                            "rented_ports": rented_ports,
                            "verification_sysbox": verification.sysbox_runtime,
                            "fallback_sysbox": sysbox_runtime,
                        }),
                    )
                )
                sysbox_result = sysbox_runtime

            result = PortVerificationResult(
                selected_ports=verification.selected_ports,
                successful_ports=verification.successful_ports,
                failed_ports=verification.failed_ports,
                dind_port=verification.dind_port,
                dind_ok=verification.dind_ok,
                sysbox_runtime=sysbox_result,
                status=verification.status,
                error=verification.error,
                elapsed_sec=time.monotonic() - t1,
            )

            return result
        except Exception as e:
            logger.error(
                _m(
                    "verification failed",
                    extra=get_extra_info({
                        **log_ctx,
                        "error": str(e),
                    }),
                ),
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
