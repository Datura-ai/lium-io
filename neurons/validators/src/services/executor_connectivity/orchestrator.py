import logging
import random

from datura.requests.miner_requests import ExecutorSSHInfo

from core.docker_utils import DockerCommand
from core.utils import _m, get_extra_info
from services.const import BATCH_PORT_VERIFICATION_SIZE, MIN_PORT_COUNT
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import PortVerificationResult
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import (
    PortSelector,
    declared_ports,
    tally_port_ranges,
)

logger = logging.getLogger(__name__)

# Our own short-lived probe containers; any other container's published port belongs to a pod or a filler.
PROBE_CONTAINER_PREFIXES = ("/port_test", "/container_")


async def ports_held_by_containers(ssh_client) -> set[int]:
    """Host ports published by any non-probe container on the host, stopped or restarting ones included.

    A rental created while a validation job runs publishes its ports after the job picked its own. If its
    container exits and Docker restarts it while our DinD container sits on one of them, the restart fails
    with "port is already allocated" and the pod stays down. Read failures return an empty set."""
    try:
        result = await ssh_client.run(DockerCommand.host_port_bindings())
    except Exception:
        return set()
    held: set[int] = set()
    for line in (result.stdout or "").splitlines():
        name, _, ports = line.partition(" ")
        if name.startswith(PROBE_CONTAINER_PREFIXES):
            continue
        held.update(int(port) for port in ports.split() if port.isdigit())
    return held


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
        min_ports: int = MIN_PORT_COUNT,
        dind_deadline_monotonic: float | None = None,
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
        # the validator's rental snapshot can miss a pod created or restarting since it was read
        held_at_start = await ports_held_by_containers(ssh_client)
        ports = [port for port in ports if port.internal not in held_at_start]
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
            )

        probe_result = await self.port_probe.probe(
            ports,
            ssh_client=ssh_client,
            host=executor_info.address,
            log_ctx=log_ctx,
            min_ports=min_ports,
        )

        successful = list(probe_result.successful)
        failed = list(probe_result.failed)

        held = await ports_held_by_containers(ssh_client)
        free_successful = [port for port in successful if port.internal not in held]
        free_ports = [port for port in ports if port.internal not in held]
        if held & {port.internal for port in ports}:
            logger.info(
                _m(
                    "DinD port choice skips ports a container already publishes",
                    extra=get_extra_info({**log_ctx, "held_ports": sorted(held & {p.internal for p in ports})}),
                )
            )
        if free_successful:
            dind_port = free_successful[0]
            successful.remove(dind_port)
        else:
            # shortcut: if every selected port is held we still bind one (old behaviour); skip DinD if this shows up
            dind_port = random.choice(free_ports or ports)
        sysbox_requested = sysbox_runtime
        dind_result = await self.dind_probe.verify(
            dind_port,
            ssh_client=ssh_client,
            host=executor_info.address,
            container_name_prefix=f"container_{miner_hotkey}",
            sysbox_runtime=sysbox_runtime,
            log_ctx=log_ctx,
            deadline_monotonic=dind_deadline_monotonic,
        )

        if dind_result.success:
            successful.append(dind_port)
            sysbox_runtime = dind_result.sysbox_runtime
        else:
            failed.append(dind_port)
            sysbox_runtime = False

        port_ranges = tally_port_ranges(declared, ports, successful)
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
            probe_tier=probe_result.tier,
            external_probe=probe_result.external,
            declared_port_count=declared_port_count,
            port_ranges=port_ranges,
            host_runtime_fault=dind_result.sshd_never_answered,
            sysbox_measured=dind_result.sysbox_runtime
            if dind_result.success and sysbox_requested and not dind_result.sysbox_timed_out
            else None,
        )
