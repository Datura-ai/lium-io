"""DAH-2667: measure a RoCE fabric instead of inferring it.

A RoCE fabric has no subnet manager, so the fleet can only infer one: same miner, same IPv4
segment, one address per host. That is evidence, not proof — a switch without lossless queueing
(PFC/ECN) carries no RDMA at all, and NCCL then falls back to TCP over the cluster overlay while
every device check still passes. The renter pays cluster price for a job nothing in the logs
explains.

So the validator measures the wire before the backend sells it: `ib_write_bw` between the two
hosts, over the same rail and GID the cluster image hands NCCL, run from
`daturaai/lium-rdma-probe`. The result is written into the specs of BOTH hosts, and the backend
groups a RoCE pair only when each side carries a fresh measurement naming the other.

Two facts shape when this runs. The measurement only matters for a pair that can be sold, and a
pair is sellable only while both hosts are free — which is exactly when the probe can take the
cards without touching a renter's job. So it runs at the end of a miner's cycle, over that miner's
idle executors, and never touches a rented one.
"""

from __future__ import annotations

import logging
import re
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from ipaddress import IPv4Address, IPv6Address, ip_network

import asyncssh
from datura.requests.miner_requests import ExecutorSSHInfo
from pydantic import BaseModel

from core.config import settings
from core.utils import _m, get_extra_info
from services.task.models import JobResult

logger = logging.getLogger(__name__)

PROBE_IMAGE = "daturaai/lium-rdma-probe:0.0.1"
PROBE_CONTAINER_NAME = "lium_roce_probe"
# Both sides open this for the out-of-band handshake; the data path is card to card and never
# touches it. Well above the ephemeral range so it cannot collide with a renter's published port.
PROBE_HANDSHAKE_PORT = 18515
PROBE_SECONDS = 5
# The client has to pull a 38 MB image on a host that has never seen it, hand the handshake to the
# server and push traffic. Generous, because the cost of a timeout is a pair we refuse to sell.
PROBE_TIMEOUT_SECONDS = 300
SPEC_KEY = "roce_link_measurement"

ROCE_LINK_LAYER = "ethernet"
# The prefix length is reported nowhere, so a segment is taken as a /24 — the same guess the
# backend's grouping makes, and it errs towards splitting rather than towards inventing a fabric.
ROCE_SEGMENT_PREFIX_LENGTH = 24
# `ib_write_bw --report_gbits` prints one data row: bytes, iterations, peak, average, message rate.
_BANDWIDTH_ROW = re.compile(r"^\s*\d+\s+\d+\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s*$", re.MULTILINE)


class RoceLinkMeasurement(BaseModel):
    """What one host measured against its neighbour, as it travels in that host's specs."""

    peer_executor_uuid: str
    peer_address: str
    gigabits_per_second: float
    measured_at: str


@dataclass(frozen=True, slots=True)
class RoceFabricHost:
    """A host that can be measured: one live RoCE rail, one address, one segment."""

    roce_address: str
    segment: str


def _is_active(port: dict) -> bool:
    return str(port.get("state") or "").split(":")[-1].strip().upper() == "ACTIVE"


def _is_roce(port: dict) -> bool:
    return str(port.get("link_layer") or "").strip().lower() == ROCE_LINK_LAYER


def _address_of(port: dict) -> IPv4Address | None:
    """The card's own IPv4 address, decoded from the GID table rather than taken by index.

    mlx5 puts the IPv4-mapped entry at 2-3 and Intel irdma at 1, so the index names nothing. An
    address the card gave itself names nothing either: 169.254/16 is what a NIC without a lease
    holds, and every unconfigured card of a provider would otherwise land on one invented segment.
    """
    for gid in port.get("gids") or []:
        try:
            mapped_address: IPv4Address | None = IPv6Address(str(gid).strip()).ipv4_mapped
        except ValueError:
            continue
        if mapped_address is None:
            continue
        if mapped_address.is_link_local or mapped_address.is_loopback or mapped_address.is_unspecified:
            continue
        return mapped_address
    return None


def roce_fabric_host_of(result: JobResult) -> RoceFabricHost | None:
    """The RoCE fabric this host sits on, or None when it holds none worth measuring.

    Silent for a host that also holds a live InfiniBand port — such a host is sold on THAT fabric,
    and `lium-fabric-env` inside the pod agrees. Silent too when its rails answer on two segments:
    which one a job would run on is ambiguous, so the backend refuses to sell it either way.
    """
    live_ports: list[dict] = [
        port for port in (result.spec or {}).get("infiniband_ports") or [] if _is_active(port)
    ]
    if any(not _is_roce(port) for port in live_ports):
        return None

    addresses: list[IPv4Address] = [
        address for port in live_ports if (address := _address_of(port)) is not None
    ]
    if not addresses:
        return None

    segments: set[str] = {
        str(ip_network(f"{address}/{ROCE_SEGMENT_PREFIX_LENGTH}", strict=False))
        for address in addresses
    }
    if len(segments) != 1:
        return None
    return RoceFabricHost(roce_address=str(addresses[0]), segment=segments.pop())


def pairs_to_measure(results: list[JobResult]) -> list[tuple[JobResult, JobResult]]:
    """Disjoint pairs of free hosts on one segment, in the order they will be measured.

    Free only: a rented host is carrying someone's job, and the pair could not be sold anyway. Two
    hosts claiming ONE address are not neighbours but two machines behind different NATs, so both
    are dropped — the same rule the backend's grouping applies, and for the same reason.
    """
    hosts_by_segment: dict[str, list[tuple[JobResult, RoceFabricHost]]] = {}
    for result in results:
        if result.is_rented:
            continue
        host = roce_fabric_host_of(result)
        if host is None:
            continue
        hosts_by_segment.setdefault(host.segment, []).append((result, host))

    pairs: list[tuple[JobResult, JobResult]] = []
    for members in hosts_by_segment.values():
        addressed = _without_addresses_claimed_twice(members)
        addressed.sort(key=lambda member: member[1].roce_address)
        for first, second in zip(addressed[::2], addressed[1::2]):
            pairs.append((first[0], second[0]))
    return pairs


def _without_addresses_claimed_twice(
    members: list[tuple[JobResult, RoceFabricHost]],
) -> list[tuple[JobResult, RoceFabricHost]]:
    claimants: dict[str, int] = {}
    for _, host in members:
        claimants[host.roce_address] = claimants.get(host.roce_address, 0) + 1
    return [member for member in members if claimants[member[1].roce_address] == 1]


def measured_gigabits_per_second(ib_write_bw_output: str) -> float | None:
    """The average bandwidth of a finished run, or None when the run printed no result row."""
    match = _BANDWIDTH_ROW.search(ib_write_bw_output)
    if match is None:
        return None
    return float(match.group(1))


def attach_measurement(result: JobResult, measurement: RoceLinkMeasurement) -> None:
    """Carry the measurement to the backend in this host's specs, where the grouping reads it."""
    if result.spec is None:
        result.spec = {}
    result.spec[SPEC_KEY] = measurement.model_dump(mode="json")


def _server_command() -> str:
    return (
        f"docker rm -f {PROBE_CONTAINER_NAME} >/dev/null 2>&1; "
        f"docker run --rm -d --name {PROBE_CONTAINER_NAME} --network host --device /dev/infiniband "
        f"{PROBE_IMAGE} server {PROBE_HANDSHAKE_PORT} {PROBE_SECONDS}"
    )


def _client_command(server_roce_address: str) -> str:
    return (
        f"docker run --rm --network host --device /dev/infiniband {PROBE_IMAGE} "
        f"client {PROBE_HANDSHAKE_PORT} {PROBE_SECONDS} {shlex.quote(server_roce_address)}"
    )


async def _measure_pair(
    server: JobResult,
    client: JobResult,
    decrypted_private_key: str,
    log_extra: dict,
) -> float | None:
    """Run the probe across one pair and return the bandwidth the client saw."""
    server_host = roce_fabric_host_of(server)
    if server_host is None:
        return None

    pkey = asyncssh.import_private_key(decrypted_private_key)
    async with _connected(server.executor_info, pkey) as server_ssh:
        await server_ssh.run(_server_command(), check=False, timeout=PROBE_TIMEOUT_SECONDS)
        try:
            async with _connected(client.executor_info, pkey) as client_ssh:
                completed = await client_ssh.run(
                    _client_command(server_host.roce_address),
                    check=False,
                    timeout=PROBE_TIMEOUT_SECONDS,
                )
        finally:
            # The server holds a card until it is stopped, and a probe must never outlive its own
            # measurement on a host a renter is about to take.
            await server_ssh.run(
                f"docker rm -f {PROBE_CONTAINER_NAME} >/dev/null 2>&1",
                check=False,
                timeout=PROBE_TIMEOUT_SECONDS,
            )

    gigabits = measured_gigabits_per_second(str(completed.stdout or ""))
    if gigabits is None:
        logger.info(
            _m(
                "RoCE link probe measured nothing",
                extra=get_extra_info(
                    {
                        **log_extra,
                        "server_executor": server.executor_info.uuid,
                        "client_executor": client.executor_info.uuid,
                        "stderr": str(completed.stderr or "")[:500],
                    }
                ),
            ),
        )
    return gigabits


@asynccontextmanager
async def _connected(
    executor_info: ExecutorSSHInfo, pkey: asyncssh.SSHKey
) -> AsyncIterator[asyncssh.SSHClientConnection]:
    async with asyncssh.connect(
        host=executor_info.address,
        port=executor_info.ssh_port,
        username=executor_info.ssh_username,
        client_keys=[pkey],
        known_hosts=None,
    ) as connection:
        yield connection


async def measure_and_attach(
    results: list[JobResult], decrypted_private_key: str, log_extra: dict
) -> None:
    """Measure every free RoCE pair of this miner and write the result into both hosts' specs.

    Best effort by design: a probe that fails leaves the specs without a measurement, the backend
    then does not sell that pair, and the validation cycle carries on untouched.
    """
    if not settings.ROCE_LINK_PROBE_ENABLED:
        return

    pairs = pairs_to_measure(results)
    if not pairs:
        return

    logger.info(
        _m(
            "Measuring RoCE links before the backend sells them",
            extra=get_extra_info({**log_extra, "pairs": len(pairs)}),
        ),
    )
    for server, client in pairs:
        try:
            gigabits = await _measure_pair(server, client, decrypted_private_key, log_extra)
        except Exception as error:
            logger.warning(
                _m(
                    "RoCE link probe failed",
                    extra=get_extra_info(
                        {
                            **log_extra,
                            "server_executor": server.executor_info.uuid,
                            "client_executor": client.executor_info.uuid,
                            "error": str(error),
                        }
                    ),
                ),
            )
            continue
        if gigabits is None:
            continue

        measured_at: str = datetime.now(timezone.utc).isoformat()
        server_host = roce_fabric_host_of(server)
        client_host = roce_fabric_host_of(client)
        if server_host is None or client_host is None:
            continue
        attach_measurement(
            server,
            RoceLinkMeasurement(
                peer_executor_uuid=client.executor_info.uuid,
                peer_address=client_host.roce_address,
                gigabits_per_second=gigabits,
                measured_at=measured_at,
            ),
        )
        attach_measurement(
            client,
            RoceLinkMeasurement(
                peer_executor_uuid=server.executor_info.uuid,
                peer_address=server_host.roce_address,
                gigabits_per_second=gigabits,
                measured_at=measured_at,
            ),
        )
        logger.info(
            _m(
                "RoCE link measured",
                extra=get_extra_info(
                    {
                        **log_extra,
                        "server_executor": server.executor_info.uuid,
                        "client_executor": client.executor_info.uuid,
                        "gigabits_per_second": gigabits,
                    }
                ),
            ),
        )
