import logging
import random
import time
from datetime import UTC, datetime

from datura.requests.miner_requests import ExecutorSSHInfo

from core.docker_utils import DockerCommand
from core.utils import _m, get_extra_info
from services.const import BATCH_PORT_VERIFICATION_SIZE, MIN_PORT_COUNT
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import PortPair, PortVerificationResult
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import (
    PortSelector,
    declared_ports,
    tally_port_ranges,
)

logger = logging.getLogger(__name__)

# a port check's hold outlives its own deadline only by this much if the release is lost
PROBE_HOLD_TTL_SECONDS = 600

# Our own short-lived probe containers; any other container's published port belongs to a pod or a filler.
PROBE_CONTAINER_PREFIXES = ("/port_test", "/container_")


async def ports_held_by_containers(ssh_client) -> set[int] | None:
    """Host ports published by any non-probe container on the host, stopped or restarting ones included.

    A rental created while a validation job runs publishes its ports after the job picked its own. If its
    container exits and Docker restarts it while our DinD container sits on one of them, the restart fails
    with "port is already allocated" and the pod stays down. None when the inventory is unknown: the read
    raised, exited non-zero or printed a line it cannot parse."""
    try:
        result = await ssh_client.run(DockerCommand.host_port_bindings())
    except Exception:
        return None
    if result.exit_status != 0 or not isinstance(result.stdout, str):
        return None
    held: set[int] = set()
    for line in result.stdout.splitlines():
        name, _, ports = line.partition(" ")
        if not name.startswith("/") or not all(port.isdigit() for port in ports.split()):
            return None
        if name.startswith(PROBE_CONTAINER_PREFIXES):
            continue
        held.update(int(port) for port in ports.split())
    return held


class ConnectivityOrchestrator:
    """Coordinates selection, probing, and DinD verification."""

    def __init__(
        self,
        port_selector: PortSelector,
        port_probe: PortProbe,
        dind_probe: DindProbe,
        port_claims=None,
    ):
        self.port_selector = port_selector
        self.port_probe = port_probe
        self.dind_probe = dind_probe
        # RedisService: the probe publishes the ports it binds and reads the ports a new pod was just given
        self.port_claims = port_claims

    async def _claims(self, method: str, *args, default=None):
        """A claims read or write that fails is logged and treated as nothing claimed: it never fails the check."""
        if self.port_claims is None:
            return default
        try:
            return await getattr(self.port_claims, method)(*args)
        except Exception as e:
            logger.warning(_m(f"port claims {method} failed: {str(e)[:100]}"))
            return default

    async def verify(self, **kwargs) -> PortVerificationResult:
        try:
            return await self._verify(**kwargs)
        finally:
            await self._claims("release_probe_ports", kwargs["executor_info"].uuid)

    async def _verify(
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
        # the validator's rental snapshot can miss a pod created or restarting since it was read
        held_at_start = await ports_held_by_containers(ssh_client)
        # a pod given its ports after the backend's rental list was read has no container yet, or a stopped one
        pending = await self._claims("pending_pod_ports", executor_info.uuid, default=set())
        # the inventory lists host (internal) ports; the selector matches unavailable_ports on the external side
        ports = self.port_selector.select(
            executor_info,
            BATCH_PORT_VERIFICATION_SIZE,
            set(unavailable_ports or []) | pending,
            declared=[port for port in declared if port.internal not in (held_at_start or set())],
        )
        # the batch test binds every selected port on the host network: a pod created meanwhile must not get one
        hold_ttl = PROBE_HOLD_TTL_SECONDS
        if dind_deadline_monotonic is not None:
            hold_ttl = max(60, min(hold_ttl, int(dind_deadline_monotonic - time.monotonic()) + 60))
        await self._claims("hold_probe_ports", executor_info.uuid, {port.external for port in ports}, hold_ttl)
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

        # read again right before the bind: a rental may have published a port while the batch probe ran.
        # Residual race, not fixed: a container can still publish the chosen port between this read and the bind.
        held = await ports_held_by_containers(ssh_client) if held_at_start is not None else None
        pending = await self._claims("pending_pod_ports", executor_info.uuid, default=set())

        def free(port: PortPair) -> bool:
            return held is not None and port.internal not in held and port.external not in pending

        free_successful = [port for port in successful if free(port)]
        free_ports = [port for port in ports if free(port)]
        if free_successful:
            dind_port = free_successful[0]
            successful.remove(dind_port)
        elif free_ports:
            dind_port = random.choice(free_ports)
        else:
            dind_port = None
        probe_log = {
            **log_ctx,
            "probe_container": f"container_{miner_hotkey}_{dind_port.external}" if dind_port else None,
            "internal_port": dind_port.internal if dind_port else None,
            "external_port": dind_port.external if dind_port else None,
            "held_ports": sorted(held & {p.internal for p in ports}) if held is not None else None,
            "inventory_known": held is not None,
        }
        if dind_port is None:
            logger.info(_m("DinD probe skipped: no candidate port known free", extra=get_extra_info(probe_log)))
            port_ranges = tally_port_ranges(declared, ports, successful)
            return PortVerificationResult(
                selected_ports=tuple(ports),
                successful_ports=tuple(successful),
                failed_ports=tuple(failed),
                dind_port=None,
                dind_ok=False,
                sysbox_runtime=sysbox_runtime,
                status="ok" if successful else "no_working_ports",
                probe_tier=probe_result.tier,
                external_probe=probe_result.external,
                declared_port_count=declared_port_count,
                port_ranges=port_ranges,
                dind_skipped=True,
            )
        sysbox_requested = sysbox_runtime
        started_at = datetime.now(UTC).isoformat()
        dind_result = await self.dind_probe.verify(
            dind_port,
            ssh_client=ssh_client,
            host=executor_info.address,
            container_name_prefix=f"container_{miner_hotkey}",
            sysbox_runtime=sysbox_runtime,
            log_ctx=log_ctx,
            deadline_monotonic=dind_deadline_monotonic,
        )
        logger.info(
            _m(
                "DinD probe finished",
                extra=get_extra_info(
                    {**probe_log, "started_at": started_at, "ended_at": datetime.now(UTC).isoformat(),
                     "dind_ok": dind_result.success}
                ),
            )
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
