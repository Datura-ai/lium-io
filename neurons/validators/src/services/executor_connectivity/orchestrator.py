import random
import logging

from core.utils import _m, get_extra_info
from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import BATCH_PORT_VERIFICATION_SIZE, MIN_PORT_COUNT
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import (
    DindCheck,
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

        if not ports:
            return PortVerificationResult(
                selected_ports=tuple(),
                successful_ports=tuple(),
                failed_ports=tuple(),
                dind_port=None,
                dind_ok=False,
                sysbox_runtime=sysbox_runtime,
                status="no_ports",
                port_ranges=tally_port_ranges(declared, (), ()),
            )

        probe_result = await self.port_probe.probe(
            ports,
            ssh_client=ssh_client,
            host=executor_info.address,
            log_ctx=log_ctx,
        )

        successful = list(probe_result.successful)
        failed = list(probe_result.failed)

        async def probe_dind(port: PortPair) -> DindCheck:
            result = await self.dind_probe.verify(
                port,
                ssh_client=ssh_client,
                host=executor_info.address,
                container_name_prefix=f"container_{miner_hotkey}",
                sysbox_runtime=sysbox_runtime,
                log_ctx=log_ctx,
            )
            return DindCheck(port, result)

        # Pass two restores the port count only; it never overrides a failed container check. With a
        # pass-one answer the check runs first, on that port, as the one-pass check did, and a
        # failure publishes that check's result with no pass two. With none (the one-pass check would
        # have run on a port that did not answer) the check waits for pass two and runs on one of its
        # answers; if it fails, none of pass two's answers count: 0 verified, DinD failed.
        dind: DindCheck | None = None
        if successful:
            dind = await probe_dind(successful.pop(0))
            (successful if dind.result.success else failed).append(dind.port)

        skip_reason = self._second_pass_skip_reason(len(successful), dind, probe_result)
        if skip_reason is not None:
            second_pass = SecondPassRun(skip_reason)
        else:
            second_pass = await self._run_second_pass(
                declared,
                unavailable,
                ports,
                ssh_client=ssh_client,
                host=executor_info.address,
                log_ctx=log_ctx,
            )

        outcome = second_pass.outcome
        if dind is None and second_pass.answered:
            dind = await probe_dind(second_pass.answered[0])
            if not dind.result.success:
                outcome = SecondPass.DISCARDED_CONTAINER_FAILED
                failed.append(dind.port)
        second_pass_counted = outcome != SecondPass.DISCARDED_CONTAINER_FAILED
        if second_pass_counted:
            successful += second_pass.answered
            failed += second_pass.failed
        if dind is None:
            # no answer in either pass: DinD runs on a random pass-one port, as the one-pass check did
            dind = await probe_dind(random.choice(ports))
            (successful if dind.result.success else failed).append(dind.port)
        if outcome != SecondPass.NOT_NEEDED:
            logger.info(
                _m(
                    f"second port pass: {outcome}, {len(second_pass.probed)} ports",
                    extra=get_extra_info(log_ctx),
                )
            )

        port_ranges = tally_port_ranges(declared, ports, successful)
        if second_pass.probed:
            port_ranges += tally_port_ranges(
                declared,
                second_pass.probed,
                second_pass.answered,
                pass_number=2,
                answers_counted=second_pass_counted,
            )
        return PortVerificationResult(
            selected_ports=tuple(ports) + tuple(second_pass.probed),
            successful_ports=tuple(successful),
            failed_ports=tuple(failed),
            dind_port=dind.port,
            dind_ok=dind.result.success,
            sysbox_runtime=dind.result.sysbox_runtime if dind.result.success else False,
            status="ok" if successful else "no_working_ports",
            dind_error=dind.result.error,
            port_ranges=port_ranges,
            second_pass=outcome,
        )

    @staticmethod
    def _second_pass_skip_reason(
        verified_count: int, pass_one_dind: DindCheck | None, pass_one: PortProbeResult
    ) -> SecondPass | None:
        if verified_count >= MIN_PORT_COUNT:
            return SecondPass.NOT_NEEDED
        if pass_one_dind is not None and not pass_one_dind.result.success:
            return SecondPass.SKIPPED_CONTAINER_FAILED
        if not pass_one.batch_completed:
            return SecondPass.SKIPPED_BATCH_FAILED
        return None

    async def _run_second_pass(
        self,
        declared: list[PortPair],
        unavailable: set[int],
        pass_one_ports: list[PortPair],
        *,
        ssh_client,
        host: str,
        log_ctx: dict,
    ) -> SecondPassRun:
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
