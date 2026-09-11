from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from datura.requests.miner_requests import ExecutorSSHInfo
from services.const import BATCH_PORT_VERIFICATION_SIZE
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.models import PortPair, PortProbeResult, PortVerificationResult
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import PortSelector

from core.utils import _m, get_extra_info

if TYPE_CHECKING:  # annotation only: local_verify_facts imports this package's models
    from services.local_verify_facts import PreparedDind

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

    def _choose_ports(
        self,
        executor_info: ExecutorSSHInfo,
        unavailable_ports: list[int] | None,
        published_ports: list[int] | None,
        prestarted_dind: PreparedDind | None,
    ) -> tuple[list[PortPair], list[PortPair]]:
        """(ports this run judges, ports the batch probe binds).

        The backend's `unavailable_ports` decide WHICH window of the range is probed. The two
        host-reported inputs may only shrink that window, never shift it: `published_ports` (liumd
        phase 2, the executor's own docker) because a bind on a published port fails as one on a
        rented port does, and the prestarted DinD's port (phase 2c) because the validator's own
        container already holds it — the DinD probe connects to it, the batch skips it. An empty
        remainder after the published list keeps the full window: the connect-back proves it, not
        the executor's word. When the prestarted port is the only port there is, the DinD probe
        alone decides (no batch bind on a port the container holds: it would count the same port
        as failed and successful).
        """
        unavailable = set(unavailable_ports or [])
        if prestarted_dind is not None:
            unavailable.add(prestarted_dind.port.external)
        ports = self.port_selector.select(executor_info, BATCH_PORT_VERIFICATION_SIZE, unavailable)
        if published_ports:
            published = set(published_ports)
            if prestarted_dind is not None:
                published.discard(prestarted_dind.port.external)  # ours, not "taken by the host"
            ports = [pair for pair in ports if pair.external not in published] or ports
        if not ports and prestarted_dind is not None:
            return [prestarted_dind.port], []
        return ports, ports

    async def verify(
        self,
        *,
        executor_info: ExecutorSSHInfo,
        miner_hotkey: str,
        sysbox_runtime: bool,
        unavailable_ports: list[int] | None,
        ssh_client,
        log_ctx: dict | None = None,
        published_ports: list[int] | None = None,
        prestarted_dind=None,
    ) -> PortVerificationResult:
        log_ctx = {
            **(log_ctx or {}),
            "executor_uuid": executor_info.uuid,
            "executor_ip": executor_info.address,
        }
        ports, batch = self._choose_ports(
            executor_info, unavailable_ports, published_ports, prestarted_dind
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
            )

        if batch:
            probe_result = await self.port_probe.probe(
                batch,
                ssh_client=ssh_client,
                host=executor_info.address,
                log_ctx=log_ctx,
            )
        else:
            probe_result = PortProbeResult(successful=tuple(), failed=tuple())

        successful = list(probe_result.successful)
        failed = list(probe_result.failed)

        dind_result = None
        if prestarted_dind is not None:
            dind_port = prestarted_dind.port
            dind_result = await self.dind_probe.verify(
                dind_port,
                ssh_client=ssh_client,
                host=executor_info.address,
                container_name_prefix=f"container_{miner_hotkey}",
                sysbox_runtime=sysbox_runtime,
                log_ctx=log_ctx,
                prestarted=prestarted_dind,
            )
            if not dind_result.success:
                # The executor's container did not answer the validator's key: it is gone (the
                # verifier removed it) and today's `docker run` probe runs on the port the removal
                # just freed — not on one of the batch's proven ports, which all stay in the count.
                # A host with exactly MIN_PORT_COUNT free ports must not fail the fatal port check
                # because our own container was the one that did not answer.
                logger.info(_m("DinD prestart not usable; probing the freed port as today", extra=get_extra_info(log_ctx)))
                dind_result = None
        if dind_result is None:
            if prestarted_dind is not None:
                dind_port = prestarted_dind.port
            else:
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
