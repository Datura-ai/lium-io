import random
import logging

from core.utils import _m, get_extra_info
from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import BATCH_PORT_VERIFICATION_SIZE, MIN_PORT_COUNT
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import (
    SECOND_PASS_BATCH_FAILED,
    SECOND_PASS_DISCARDED_CONTAINER_FAILED,
    SECOND_PASS_NO_PORTS_LEFT,
    SECOND_PASS_NOT_NEEDED,
    SECOND_PASS_RAN,
    SECOND_PASS_SKIPPED_BATCH_FAILED,
    SECOND_PASS_SKIPPED_CONTAINER_FAILED,
    DindProbeResult,
    PortPair,
    PortVerificationResult,
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
        container_check = {
            "ssh_client": ssh_client,
            "host": executor_info.address,
            "container_name_prefix": f"container_{miner_hotkey}",
            "sysbox_runtime": sysbox_runtime,
            "log_ctx": log_ctx,
        }

        # Pass two restores the port count only; it never overrides a failed container check. With a
        # pass-one answer the check runs first, on that port, as the one-pass check did, and a
        # failure publishes that check's result with no pass two. With none (the one-pass check would
        # have run on a port that did not answer) the check waits for pass two and runs on one of its
        # answers; if it fails, none of pass two's answers count: 0 verified, DinD failed.
        dind = None
        if successful:
            dind = await self._check_container(successful, failed, ports, **container_check)

        spread: list[PortPair] = []
        spread_answers: list[PortPair] = []
        spread_failed: list[PortPair] = []
        if len(successful) >= MIN_PORT_COUNT:
            second_pass = SECOND_PASS_NOT_NEEDED
        elif dind is not None and not dind[1].success:
            second_pass = SECOND_PASS_SKIPPED_CONTAINER_FAILED
        elif not probe_result.batch_ran:
            second_pass = SECOND_PASS_SKIPPED_BATCH_FAILED
        else:
            spread = self.port_selector.select_spread(
                declared, BATCH_PORT_VERIFICATION_SIZE, unavailable, tested=ports
            )
            if not spread:
                second_pass = SECOND_PASS_NO_PORTS_LEFT
            else:
                spread_result = await self.port_probe.probe_spread(
                    spread,
                    ssh_client=ssh_client,
                    host=executor_info.address,
                    log_ctx=log_ctx,
                )
                second_pass = (
                    SECOND_PASS_RAN if spread_result.batch_ran else SECOND_PASS_BATCH_FAILED
                )
                spread_answers = list(spread_result.successful)
                spread_failed = list(spread_result.failed)

        if dind is None and spread_answers:
            dind_port = spread_answers[0]
            dind_result = await self.dind_probe.verify(dind_port, **container_check)
            dind = dind_port, dind_result
            if not dind_result.success:
                second_pass = SECOND_PASS_DISCARDED_CONTAINER_FAILED
                failed.append(dind_port)
        if second_pass != SECOND_PASS_DISCARDED_CONTAINER_FAILED:
            successful += spread_answers
            failed += spread_failed
        if dind is None:
            dind = await self._check_container(successful, failed, ports, **container_check)
        dind_port, dind_result = dind
        sysbox_runtime = dind_result.sysbox_runtime if dind_result.success else False
        if second_pass != SECOND_PASS_NOT_NEEDED:
            logger.info(
                _m(
                    f"second port pass: {second_pass}, {len(spread)} ports",
                    extra=get_extra_info(log_ctx),
                )
            )

        status = "ok" if successful else "no_working_ports"
        port_ranges = tally_port_ranges(declared, ports, successful)
        if spread:
            port_ranges += tally_port_ranges(
                declared,
                spread,
                spread_answers,
                pass_number=2,
                counted=second_pass != SECOND_PASS_DISCARDED_CONTAINER_FAILED,
            )
        return PortVerificationResult(
            selected_ports=tuple(ports) + tuple(spread),
            successful_ports=tuple(successful),
            failed_ports=tuple(failed),
            dind_port=dind_port,
            dind_ok=dind_result.success,
            sysbox_runtime=sysbox_runtime,
            status=status,
            dind_error=dind_result.error,
            port_ranges=port_ranges,
            second_pass=second_pass,
        )

    async def _check_container(
        self,
        successful: list[PortPair],
        failed: list[PortPair],
        ports: list[PortPair],
        **kwargs,
    ) -> tuple[PortPair, DindProbeResult]:
        dind_port = successful.pop(0) if successful else random.choice(ports)
        dind_result = await self.dind_probe.verify(dind_port, **kwargs)
        (successful if dind_result.success else failed).append(dind_port)
        return dind_port, dind_result
