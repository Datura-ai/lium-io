import random
import logging

from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from services.const import BATCH_PORT_VERIFICATION_SIZE, MIN_PORT_COUNT
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import PortVerificationResult
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import PortSelector

logger = logging.getLogger(__name__)


class ConnectivityOrchestrator:
    """Coordinates selection, probing, and DinD verification."""

    def __init__(
        self,
        port_selector: PortSelector,
        port_probe: PortProbe,
        dind_probe: DindProbe,
    ):
        self.port_selector = port_selector
        self.port_probe = port_probe
        self.dind_probe = dind_probe

    async def verify(
        self,
        *,
        executor_info: ExecutorSSHInfo,
        miner_hotkey: str,
        sysbox_runtime: bool,
        unavailable_ports: list[int] | None,
        ssh_client,
        log_ctx: dict | None = None,
    ) -> PortVerificationResult:
        log_ctx = {
            **(log_ctx or {}),
            "executor_uuid": executor_info.uuid,
            "executor_ip": executor_info.address,
        }
        ports = self.port_selector.select(
            executor_info, BATCH_PORT_VERIFICATION_SIZE, set(unavailable_ports or [])
        )
        declared_port_count = self.port_selector.declared_count(executor_info)

        if not ports:
            return PortVerificationResult(
                selected_ports=tuple(),
                successful_ports=tuple(),
                failed_ports=tuple(),
                dind_port=None,
                dind_ok=False,
                sysbox_runtime=sysbox_runtime,
                status="no_ports",
                declared_port_count=declared_port_count,
            )

        probe_result = await self.port_probe.probe(
            ports,
            ssh_client=ssh_client,
            host=executor_info.address,
            log_ctx=log_ctx,
        )

        successful = list(probe_result.successful)
        failed = list(probe_result.failed)

        dind_port = successful.pop(0) if successful else random.choice(ports)
        dind_result = await self.dind_probe.verify(
            dind_port,
            ssh_client=ssh_client,
            host=executor_info.address,
            container_name_prefix=f"container_{miner_hotkey}",
            sysbox_runtime=sysbox_runtime,
            log_ctx=log_ctx,
        )

        if dind_result.success:
            successful.append(dind_port)
            sysbox_runtime = dind_result.sysbox_runtime
        else:
            failed.append(dind_port)
            sysbox_runtime = False

        # After DinD, not before: a batch of exactly MIN_PORT_COUNT whose DinD port fails publishes one fewer.
        tier = probe_result.tier
        if tier == "batch" and len(successful) < MIN_PORT_COUNT and settings.PORT_PROBE_TOPUP_BELOW_FLOOR:
            successful, failed, tier = await self.port_probe.top_up(
                successful,
                failed,
                ssh_client=ssh_client,
                host=executor_info.address,
                log_ctx=log_ctx,
            )

        status = "ok" if successful else "no_working_ports"
        return PortVerificationResult(
            selected_ports=tuple(ports),
            successful_ports=tuple(successful),
            failed_ports=tuple(failed),
            dind_port=dind_port,
            dind_ok=dind_result.success,
            sysbox_runtime=sysbox_runtime,
            status=status,
            dind_error=dind_result.error,
            probe_tier=tier,
            declared_port_count=declared_port_count,
        )
