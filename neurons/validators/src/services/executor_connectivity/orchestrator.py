import random
import logging

from core.utils import _m, get_extra_info
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from services.const import BATCH_PORT_VERIFICATION_SIZE, MIN_PORT_COUNT
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import (
    PortPair,
    PortProbeResult,
    PortVerificationResult,
    SecondPass,
    SecondPassRun,
)
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import (
    PortSelector,
    declared_ports,
    tally_port_ranges,
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
    ) -> PortVerificationResult:
        log_ctx = {
            **(log_ctx or {}),
            "executor_uuid": executor_info.uuid,
            "executor_ip": executor_info.address,
        }
        declared = declared_ports(executor_info)
        unavailable = set(unavailable_ports or [])
        ports = self.port_selector.select(
            executor_info,
            BATCH_PORT_VERIFICATION_SIZE,
            unavailable,
            declared=declared,
        )
        declared_port_count = len(declared)

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
                port_ranges=tally_port_ranges(declared, (), ()),
                second_pass=SecondPass.NO_PORTS_LEFT if settings.PORT_PROBE_TOPUP_BELOW_FLOOR else None,
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
            topped_up = await self.port_probe.top_up(
                successful,
                failed,
                ssh_client=ssh_client,
                host=executor_info.address,
                log_ctx=log_ctx,
            )
            successful, failed, tier = list(topped_up.successful), list(topped_up.failed), topped_up.tier

        # After the top-up, which re-probes only `ports`: a host whose forwarded ports all sit above
        # them stays below the floor however the -p tiers do.
        second_pass: SecondPassRun | None = None
        if settings.PORT_PROBE_TOPUP_BELOW_FLOOR:
            second_pass = await self._second_pass(
                len(successful),
                probe_result,
                declared,
                unavailable,
                ports,
                ssh_client=ssh_client,
                host=executor_info.address,
                log_ctx=log_ctx,
            )
            successful += second_pass.answered
            failed += second_pass.failed
            if second_pass.outcome != SecondPass.NOT_NEEDED:
                logger.info(
                    _m(
                        f"second port pass: {second_pass.outcome}, {len(second_pass.probed)} ports",
                        extra=get_extra_info(log_ctx),
                    )
                )

        spread = second_pass.probed if second_pass else []
        port_ranges = tally_port_ranges(declared, ports, successful)
        if spread:
            port_ranges += tally_port_ranges(declared, spread, second_pass.answered, pass_number=2)

        status = "ok" if successful else "no_working_ports"
        return PortVerificationResult(
            selected_ports=tuple(ports) + tuple(spread),
            successful_ports=tuple(successful),
            failed_ports=tuple(failed),
            dind_port=dind_port,
            dind_ok=dind_result.success,
            sysbox_runtime=sysbox_runtime,
            status=status,
            dind_error=dind_result.error,
            probe_tier=tier,
            declared_port_count=declared_port_count,
            port_ranges=port_ranges,
            second_pass=second_pass.outcome if second_pass else None,
        )

    async def _second_pass(
        self,
        verified_count: int,
        pass_one: PortProbeResult,
        declared: list[PortPair],
        unavailable: set[int],
        pass_one_ports: list[PortPair],
        *,
        ssh_client,
        host: str,
        log_ctx: dict,
    ) -> SecondPassRun:
        if verified_count >= MIN_PORT_COUNT:
            return SecondPassRun(SecondPass.NOT_NEEDED)
        if not pass_one.batch_completed:
            return SecondPassRun(SecondPass.SKIPPED_BATCH_FAILED)
        spread = self.port_selector.select_spread(
            declared, BATCH_PORT_VERIFICATION_SIZE, unavailable, pass_one_ports=pass_one_ports
        )
        if not spread:
            return SecondPassRun(SecondPass.NO_PORTS_LEFT)
        result = await self.port_probe.probe_spread(
            spread, ssh_client=ssh_client, host=host, log_ctx=log_ctx
        )
        return SecondPassRun(
            SecondPass.RAN if result.batch_completed else SecondPass.BATCH_FAILED,
            probed=spread,
            answered=list(result.successful),
            failed=list(result.failed),
        )
