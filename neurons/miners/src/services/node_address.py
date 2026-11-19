"""HTTP to a node's registered address, never into the miner's own network.

A node's address is set by whoever registered the node. A name may resolve to a loopback, private or
cloud-metadata address (or change between lookups), which would make the miner send its requests to
services next to itself. Every miner request to a node goes through `executor_session()`: a name is
resolved once per connection, refused unless every address it resolves to is public, and the
connection is made to those resolved addresses (no second lookup). Redirects are never followed, so a
node cannot send the request on to another address. Same rule as the validator's
neurons/validators/src/services/node_host.py.

An IP literal never reaches the resolver (aiohttp dials it directly), so a central miner checks every
request's host before it is sent: the addresses it dials come from the portal snapshot, which strangers
register. A self-hosted miner dials the executors its own operator added, so a literal there is dialled
as given and an executor on the operator's private network keeps working.
"""

import ipaddress
import socket

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult
from aiohttp.resolver import DefaultResolver

from core.config import settings


NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")  # RFC 6052 well-known prefix: the IPv4 address in the last 32 bits


class NonPublicNodeAddress(aiohttp.ClientConnectionError):
    """The node's address name resolves to an address that is not public; nothing was dialled."""


def is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # a scoped IPv6 link-local carries %iface
    # judged as the IPv4 address it carries: Python before 3.13 calls ::ffff:127.0.0.1 global
    ip = getattr(ip, "ipv4_mapped", None) or ip
    # a DNS64 answer (what an IPv6-only host gets for a public A record) is judged as the IPv4 address it carries
    if ip.version == 6 and ip in NAT64_PREFIX:
        ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    # Python counts global-scope multicast, IPv6 site-local (fec0::/10) and other reserved ranges as global
    return (
        ip.is_global and not ip.is_multicast and not ip.is_reserved and not getattr(ip, "is_site_local", False)
    )


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    return True


async def _refuse_non_public_literal(_session, _context, params: aiohttp.TraceRequestStartParams) -> None:
    host = params.url.host or ""
    if _is_ip_literal(host) and not is_public_address(host):
        raise NonPublicNodeAddress(f"{host} is not a public address")


class PublicOnlyResolver(AbstractResolver):
    """Resolves like aiohttp's default and refuses the name when any answer is not public."""

    def __init__(self) -> None:
        self._resolver = DefaultResolver()

    async def resolve(self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET) -> list[ResolveResult]:
        answers = await self._resolver.resolve(host, port, family)
        blocked = sorted({answer["host"] for answer in answers if not is_public_address(answer["host"])})
        if blocked:
            raise NonPublicNodeAddress(f"{host} resolves to a non-public address: {', '.join(blocked)}")
        return answers

    async def close(self) -> None:
        await self._resolver.close()


def executor_session(**kwargs) -> aiohttp.ClientSession:
    """A session for requests to nodes; pass allow_redirects=False on each request (aiohttp has no session default).
    In central mode every request's IP-literal host is checked before anything is sent."""
    trace_configs = []
    if settings.CENTRAL_MODE:
        literal_check = aiohttp.TraceConfig()
        literal_check.on_request_start.append(_refuse_non_public_literal)
        trace_configs.append(literal_check)
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(resolver=PublicOnlyResolver()), trace_configs=trace_configs, **kwargs
    )
