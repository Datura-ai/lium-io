import random

from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import BATCH_PORT_VERIFICATION_SIZE
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import PortVerificationResult
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import PortSelector


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
        rented_ports: list[int] | None,
        ssh_client,
    ) -> PortVerificationResult:
        rented = set(rented_ports) if rented_ports else set()
        ports = self.port_selector.select(executor_info, BATCH_PORT_VERIFICATION_SIZE, rented)

        if not ports:
            return PortVerificationResult(
                selected_ports=tuple(),
                successful_ports=tuple(),
                failed_ports=tuple(),
                dind_port=None,
                dind_ok=False,
                sysbox_runtime=sysbox_runtime,
                status="no_ports",
            )

        probe_result = await self.port_probe.probe(
            ports,
            ssh_client=ssh_client,
            host=executor_info.address,
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
        )

        if dind_result.success:
            successful.append(dind_port)
            sysbox_runtime = dind_result.sysbox_runtime
        else:
            failed.append(dind_port)
            sysbox_runtime = False

        status = "ok" if successful else "no_working_ports"
        return PortVerificationResult(
            selected_ports=tuple(ports),
            successful_ports=tuple(successful),
            failed_ports=tuple(failed),
            dind_port=dind_port,
            dind_ok=dind_result.success,
            sysbox_runtime=sysbox_runtime,
            status=status,
        )
