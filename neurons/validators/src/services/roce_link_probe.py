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

import asyncio
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
# A write count, not a duration: the probe proves the wire carries RDMA at all, and a validator
# cycle has seconds for a miner's whole segment, not minutes. 1000 writes of 64 KiB finish in well
# under a second on any wire worth selling; the reported rate is a side effect kept for the logs.
PROBE_ITERATIONS = 1000
# Covers the one slow step — the first pull of the 38 MB image on a host that has never seen it.
PROBE_TIMEOUT_SECONDS = 120
# The whole sweep's budget. The caller's timeout cancels the miner's entire job and scores it
# zero, so the probe stops itself well before that: a cycle is 15 minutes and every other check
# has run by now.
PROBE_BUDGET_SECONDS = 180
SPEC_KEY = "roce_link_measurements"

ROCE_LINK_LAYER = "ethernet"
# The prefix length is reported nowhere, so a segment is taken as a /24 — the same guess the
# backend's grouping makes, and it errs towards splitting rather than towards inventing a fabric.
ROCE_SEGMENT_PREFIX_LENGTH = 24
# `ib_write_bw --report_gbits` prints one data row: bytes, iterations, peak, average, message rate.
_BANDWIDTH_ROW = re.compile(r"^\s*\d+\s+\d+\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s*$", re.MULTILINE)


class RoceLinkMeasurement(BaseModel):
    """What one host measured against ONE neighbour, as it travels in that host's specs.

    `gigabits_per_second` is None when the run failed: a pair that could not talk is a fact the
    backend needs, not an absence. Every pair of the segment gets an entry, so the backend can ask
    whether the exact nodes a renter picked have been proven against each other.
    """

    peer_executor_uuid: str
    peer_address: str
    gigabits_per_second: float | None
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
    """EVERY pair of free hosts on one segment, in the order they will be measured.

    Every pair, because the backend sells the whole segment as one cluster — proving a-b and c-d
    says nothing about a-c, and each run costs well under a second. Free hosts only: a rented host
    is carrying someone's job, and the pair could not be sold anyway. Two hosts claiming ONE
    address are not neighbours but two machines behind different NATs, so both are dropped — the
    same rule the backend's grouping applies, and for the same reason.
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
        for index, (server, _) in enumerate(addressed):
            for client, _ in addressed[index + 1 :]:
                pairs.append((server, client))
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
    """Add one peer's result to this host's specs, where the backend's grouping reads them.

    Appended, never assigned: a host is measured against every other host of its segment in one
    cycle, and keeping only the last of those would leave the backend unable to tell whether the
    two nodes a renter picked were ever proven against EACH OTHER. An entry for a peer already
    listed replaces that peer's entry, so the list holds this cycle's answer per peer and no more.
    """
    if result.spec is None:
        result.spec = {}
    kept: list[dict] = [
        entry
        for entry in result.spec.get(SPEC_KEY) or []
        if entry.get("peer_address") != measurement.peer_address
    ]
    result.spec[SPEC_KEY] = kept + [measurement.model_dump(mode="json")]


def _server_command() -> str:
    return (
        f"docker rm -f {PROBE_CONTAINER_NAME} >/dev/null 2>&1; "
        f"docker run --rm -d --name {PROBE_CONTAINER_NAME} --network host --device /dev/infiniband "
        f"{PROBE_IMAGE} server {PROBE_HANDSHAKE_PORT} {PROBE_ITERATIONS}"
    )


def _client_command(server_roce_address: str) -> str:
    return (
        f"docker run --rm --network host --device /dev/infiniband {PROBE_IMAGE} "
        f"client {PROBE_HANDSHAKE_PORT} {PROBE_ITERATIONS} {shlex.quote(server_roce_address)}"
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
    """Measure every free RoCE pair of this miner and record each result in both hosts' specs.

    Bounded by `PROBE_BUDGET_SECONDS` and best effort throughout. The whole sweep sits inside one
    `asyncio.wait_for`, because the caller's own timeout cancels the miner's ENTIRE job and scores
    it zero for the cycle — a probe must never cost a miner its score. Pairs measured before the
    budget ran out keep their entries; the rest are simply not proven this cycle.
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
    try:
        await asyncio.wait_for(
            _measure_every_pair(pairs, decrypted_private_key, log_extra),
            timeout=PROBE_BUDGET_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            _m(
                "RoCE link probe ran out of its budget",
                extra=get_extra_info({**log_extra, "pairs": len(pairs), "budget_seconds": PROBE_BUDGET_SECONDS}),
            ),
        )


async def _measure_every_pair(
    pairs: list[tuple[JobResult, JobResult]], decrypted_private_key: str, log_extra: dict
) -> None:
    """Run each pair in turn, recording the answer — including a failure — on both hosts."""
    for server, client in pairs:
        server_host = roce_fabric_host_of(server)
        client_host = roce_fabric_host_of(client)
        if server_host is None or client_host is None:
            continue

        try:
            gigabits: float | None = await _measure_pair(
                server, client, decrypted_private_key, log_extra
            )
        except Exception as error:
            gigabits = None
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

        measured_at: str = datetime.now(timezone.utc).isoformat()
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
                "RoCE link measured" if gigabits is not None else "RoCE link did not answer",
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
