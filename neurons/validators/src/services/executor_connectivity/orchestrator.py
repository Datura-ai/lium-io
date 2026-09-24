import random
import logging

from datura.requests.miner_requests import ExecutorSSHInfo

from core.utils import _m, get_extra_info
from services.const import BATCH_PORT_VERIFICATION_SIZE, SAMPLED_PORTS_LOWEST_PASS_BELOW
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import PortVerificationResult
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import (
    SELECTION_STRATIFIED_LOWEST,
    PortSelector,
    estimate_usable_ports,
)

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
        probe_seed: str | None = None,
    ) -> PortVerificationResult:
        log_ctx = {
            **(log_ctx or {}),
            "executor_uuid": executor_info.uuid,
            "executor_ip": executor_info.address,
        }
        sample = self.port_selector.select(
            executor_info,
            BATCH_PORT_VERIFICATION_SIZE,
            set(unavailable_ports or []),
            seed=probe_seed,
        )
        ports = sample.ports

        if not ports:
            return PortVerificationResult(
                selected_ports=tuple(),
                successful_ports=tuple(),
                failed_ports=tuple(),
                dind_port=None,
                dind_ok=False,
                sysbox_runtime=sysbox_runtime,
                status="no_ports",
                declared_port_count=sample.declared_count,
                estimated_usable_port_count=0,
                port_selection=sample.selection,
            )

        probe_result = await self.port_probe.probe(
            ports,
            ssh_client=ssh_client,
            host=executor_info.address,
            log_ctx=log_ctx,
        )

        successful = list(probe_result.successful)
        failed = list(probe_result.failed)
        # the estimate is scaled from the random sample alone: the lowest ports below are not random
        sample_estimate = estimate_usable_ports(
            len(successful), len(set(successful) | set(failed)), sample.free_count
        )
        selected = list(ports)
        selection = sample.selection

        if len(successful) < SAMPLED_PORTS_LOWEST_PASS_BELOW and sample.lowest_ports:
            # The probe set before sampling gets its turn too, so a host whose working ports sit only
            # at the low end of a wide declared range publishes no fewer than it did.
            verified = set(successful)
            lowest = [p for p in sample.lowest_ports if p not in verified]
            logger.warning(
                _m(
                    f"sample verified {len(successful)}/{len(ports)}, below {SAMPLED_PORTS_LOWEST_PASS_BELOW}; "
                    f"probing the lowest {len(lowest)} free ports too",
                    extra=get_extra_info(log_ctx),
                )
            )
            lowest_result = await self.port_probe.probe(
                lowest,
                ssh_client=ssh_client,
                host=executor_info.address,
                log_ctx=log_ctx,
            )
            successful += [p for p in lowest_result.successful if p not in verified]
            verified = set(successful)
            failed = list(
                dict.fromkeys(p for p in failed + list(lowest_result.failed) if p not in verified)
            )
            sampled = set(ports)
            selected += [p for p in lowest if p not in sampled]
            selection = SELECTION_STRATIFIED_LOWEST

        dind_port = successful.pop(0) if successful else random.choice(selected)
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

        probed_port_count = len(set(successful) | set(failed))
        status = "ok" if successful else "no_working_ports"
        return PortVerificationResult(
            selected_ports=tuple(selected),
            successful_ports=tuple(successful),
            failed_ports=tuple(failed),
            dind_port=dind_port,
            dind_ok=dind_result.success,
            sysbox_runtime=sysbox_runtime,
            status=status,
            dind_error=dind_result.error,
            declared_port_count=sample.declared_count,
            probed_port_count=probed_port_count,
            estimated_usable_port_count=min(
                sample.free_count, max(len(successful), sample_estimate)
            ),
            port_selection=selection,
        )
