"""DAH-5932: measure which AWS EFA hosts share a fabric.

EFA has no subnet manager and no IP segment, so nothing in a host's specs says which other hosts it
can reach over EFA: two instances in one placement group look exactly like two in different
regions. The validator answers it the way it answers RoCE (`roce_link_probe`): a real transfer
between two free hosts of one miner, `fi_pingpong` over the libfabric efa provider, recorded in the
specs of BOTH hosts under `efa_link_measurements`. The backend groups an EFA pair only when each
side carries a fresh measurement naming the other.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
from datetime import datetime, timezone
from ipaddress import IPv4Address

import asyncssh

from core.utils import _m, get_extra_info
from services.const import PROBE_CONTAINER_NAME
from services.roce_link_probe import (
    PROBE_HANDSHAKE_PORT,
    PROBE_ITERATIONS,
    PROBE_MINIMUM_BUDGET_SECONDS,
    PROBE_TIMEOUT_SECONDS,
    RDMA_DOCKER_FLAGS,
    RoceLinkMeasurement,
    _connected,
    _is_active,
    attach_measurement,
    budget_left_for_probe,
)
from services.task.models import JobResult

logger = logging.getLogger(__name__)

# The reviewed manifest digest of 0.0.2, the first tag that carries libfabric and `lium-efa-probe`; run by digest,
# never by tag, since the probe gets host networking and the fabric devices.
EFA_PROBE_IMAGE_DIGEST = "sha256:8e6bf7f085c70c2f16c1862fdb0ab67b2c218c09fe4a91b41be79b235f5491e2"
EFA_PROBE_IMAGE_REPOSITORY = "daturaai/lium-rdma-probe"
EFA_PROBE_ENTRYPOINT = "/usr/local/bin/lium-efa-probe"
EFA_DRIVER = "efa"
SPEC_KEY = "efa_link_measurements"
# The server container starts detached, so the first client can knock before it listens.
CLIENT_ATTEMPTS = 2
CLIENT_RETRY_DELAY_SECONDS = 2
# The transfer takes well under a second; an unreachable peer must not eat the sweep's budget.
CLIENT_DEADLINE_SECONDS = 10


def _probe_image() -> str:
    return f"{EFA_PROBE_IMAGE_REPOSITORY}@{EFA_PROBE_IMAGE_DIGEST}"


def has_live_efa_port(result: JobResult) -> bool:
    return any(
        str(port.get("driver") or "").strip().lower() == EFA_DRIVER and _is_active(port)
        for port in (result.spec or {}).get("infiniband_ports") or []
    )


def pairs_to_measure(results: list[JobResult]) -> list[tuple[JobResult, JobResult]]:
    """Every pair of this miner's free EFA hosts: one miner's hosts are the only candidates, and a
    rented host is carrying someone's job."""
    free_efa_hosts = sorted(
        (result for result in results if not result.is_rented and has_live_efa_port(result)),
        key=lambda result: result.executor_info.uuid,
    )
    return [
        (server, client) for index, server in enumerate(free_efa_hosts) for client in free_efa_hosts[index + 1 :]
    ]


def measured_gigabits_per_second(fi_pingpong_output: str) -> float | None:
    """The rate of a finished run from fi_pingpong's table, or None when it printed no result row."""
    lines = [line.split() for line in fi_pingpong_output.splitlines() if line.strip()]
    for index, header in enumerate(lines[:-1]):
        if "MB/sec" not in header:
            continue
        row = lines[index + 1]
        column = header.index("MB/sec")
        try:
            return round(float(row[column]) * 8 / 1000, 3)
        except (IndexError, ValueError):
            return None
    return None


def _docker_run_command(arguments: str, detached: bool = False) -> str:
    return (
        f"docker run --rm {'-d --name ' + PROBE_CONTAINER_NAME + ' ' if detached else ''}--network host "
        f"--device /dev/infiniband {RDMA_DOCKER_FLAGS} {arguments}"
    )


async def _host_address(ssh: asyncssh.SSHClientConnection) -> str | None:
    """The host's own IPv4, where the peer's fi_pingpong reaches it for the handshake."""
    completed = await ssh.run(
        _docker_run_command(f"--entrypoint {EFA_PROBE_ENTRYPOINT} {_probe_image()} address"),
        check=False,
        timeout=PROBE_TIMEOUT_SECONDS,
    )
    try:
        address = IPv4Address(str(completed.stdout or "").strip())
    except ValueError:
        return None
    return None if address.is_loopback or address.is_unspecified or address.is_link_local else str(address)


async def _measure_pair(
    server: JobResult, client: JobResult, decrypted_private_key: str, addresses: dict[str, str | None]
) -> float | None:
    pkey = asyncssh.import_private_key(decrypted_private_key)
    async with _connected(server.executor_info, pkey) as server_ssh:
        async with _connected(client.executor_info, pkey) as client_ssh:
            for result, ssh in ((server, server_ssh), (client, client_ssh)):
                if result.executor_info.uuid not in addresses:
                    addresses[result.executor_info.uuid] = await _host_address(ssh)
            server_address = addresses[server.executor_info.uuid]
            if server_address is None:
                return None
            try:
                # inside the try: a cancel during startup must still stop a server that did start
                await server_ssh.run(
                    f"docker rm -f {PROBE_CONTAINER_NAME} >/dev/null 2>&1; "
                    + _docker_run_command(
                        f"--entrypoint timeout {_probe_image()} {PROBE_TIMEOUT_SECONDS} "
                        f"{EFA_PROBE_ENTRYPOINT} server {PROBE_HANDSHAKE_PORT} {PROBE_ITERATIONS}",
                        detached=True,
                    ),
                    check=False,
                    timeout=PROBE_TIMEOUT_SECONDS,
                )
                for attempt in range(CLIENT_ATTEMPTS):
                    if attempt:
                        await asyncio.sleep(CLIENT_RETRY_DELAY_SECONDS)
                    completed = await client_ssh.run(
                        _docker_run_command(
                            f"--entrypoint timeout {_probe_image()} {CLIENT_DEADLINE_SECONDS} "
                            f"{EFA_PROBE_ENTRYPOINT} client "
                            f"{PROBE_HANDSHAKE_PORT} {PROBE_ITERATIONS} {shlex.quote(server_address)}"
                        ),
                        check=False,
                        timeout=PROBE_TIMEOUT_SECONDS,
                    )
                    gigabits = measured_gigabits_per_second(str(completed.stdout or ""))
                    if gigabits is not None:
                        return gigabits
                return None
            finally:
                # The server holds the card until it is stopped; it must not outlive its measurement.
                # runs even when the sweep's budget cancels us: bounded, and its error must not replace the cancel
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        server_ssh.run(f"docker rm -f {PROBE_CONTAINER_NAME} >/dev/null 2>&1", check=False),
                        timeout=CLIENT_DEADLINE_SECONDS,
                    )


async def _measure_every_pair(
    pairs: list[tuple[JobResult, JobResult]], decrypted_private_key: str, log_extra: dict
) -> None:
    addresses: dict[str, str | None] = {}
    for server, client in pairs:
        try:
            gigabits = await _measure_pair(server, client, decrypted_private_key, addresses)
        except Exception as error:
            gigabits = None
            logger.warning(
                _m("EFA link probe failed", extra=get_extra_info({**log_extra, "error": str(error)}))
            )
        measured_at = datetime.now(timezone.utc).isoformat()
        for host, peer in ((server, client), (client, server)):
            attach_measurement(
                host,
                RoceLinkMeasurement(
                    peer_executor_uuid=peer.executor_info.uuid,
                    # unknown until the peer answered the address query; the uuid is what the backend keys on
                    peer_address=addresses.get(peer.executor_info.uuid) or peer.executor_info.uuid,
                    gigabits_per_second=gigabits,
                    measured_at=measured_at,
                ),
                spec_key=SPEC_KEY,
            )
        logger.info(
            _m(
                "EFA link measured" if gigabits is not None else "EFA link did not answer",
                extra=get_extra_info(
                    {
                        **log_extra,
                        "server_executor": server.executor_info.uuid,
                        "client_executor": client.executor_info.uuid,
                        "gigabits_per_second": gigabits,
                    }
                ),
            )
        )


async def measure_and_attach(
    results: list[JobResult], decrypted_private_key: str, log_extra: dict, job_started_at: float
) -> None:
    """Measure every free EFA pair of this miner, best effort, inside what the cycle has left.

    Nothing may reach the caller: its timeout cancels the miner's whole job and scores it zero.
    """
    budget_seconds = 0.0
    if not EFA_PROBE_IMAGE_DIGEST:
        return
    try:
        pairs = pairs_to_measure(results)
        if not pairs:
            return
        budget_seconds = budget_left_for_probe(job_started_at)
        if budget_seconds < PROBE_MINIMUM_BUDGET_SECONDS:
            logger.info(
                _m(
                    "EFA link probe skipped, the cycle has no time left",
                    extra=get_extra_info({**log_extra, "budget_seconds": budget_seconds}),
                )
            )
            return
        await asyncio.wait_for(
            _measure_every_pair(pairs, decrypted_private_key, log_extra), timeout=budget_seconds
        )
    except asyncio.TimeoutError:
        logger.warning(
            _m(
                "EFA link probe ran out of its budget",
                extra=get_extra_info({**log_extra, "budget_seconds": budget_seconds}),
            )
        )
    except Exception as error:
        logger.warning(
            _m("EFA link probe failed", extra=get_extra_info({**log_extra, "error": str(error)}))
        )
