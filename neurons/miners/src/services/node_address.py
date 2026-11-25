"""The miner's dial helpers: every outbound HTTP request is made through one of two.

- `trusted_session()`: an address from the miner's own settings (the backend, the portal, a configured hook).
- `executor_session()`: a node's registered address, below.

ruff (TID251, root pyproject.toml) refuses the raw constructors anywhere else, so a new call site picks one.

HTTP to a node's registered address, never into the miner's own network.

A node's address is set by whoever registered the node. A name may resolve to a loopback, private or
cloud-metadata address (or change between lookups), which would make the miner send its requests to
services next to itself. Every miner request to a node goes through `executor_session()`: a name is
resolved once per connection, refused unless every address it resolves to is public, and the
connection is made to those resolved addresses (no second lookup). Redirects are never followed, so a
node cannot send the request on to another address. The public-address rule is datura's
`is_public_address`, the same one the validator applies.

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
from datura.public_address import is_public_address as _is_public_address

from core.config import settings


ALLOWED_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(n.strip()) for n in settings.NODE_ADDRESS_ALLOWED_NETWORKS.split(",") if n.strip()
)


class NonPublicNodeAddress(aiohttp.ClientConnectionError):
    """The node's address name resolves to an address that is not public; nothing was dialled."""


def is_public_address(address: str) -> bool:
    """datura's public-address rule, plus the private network a dev stack names (DAH-7120; unset in production)."""
    try:
        ip = ipaddress.ip_address(address.strip("[]").split("%", 1)[0])
    except ValueError:
        return False
    if any(ip in network for network in ALLOWED_PRIVATE_NETWORKS) and ip.is_private and not (
        ip.is_loopback or ip.is_link_local or ip.is_unspecified
    ):
        return True
    return _is_public_address(address)


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
    """A session for requests to nodes; a redirect ends the request instead of being followed.
    In central mode every request's IP-literal host is checked before anything is sent."""
    trace_configs = []
    if settings.CENTRAL_MODE:
        literal_check = aiohttp.TraceConfig()
        literal_check.on_request_start.append(_refuse_non_public_literal)
        trace_configs.append(literal_check)
    redirect_check = aiohttp.TraceConfig()
    redirect_check.on_request_redirect.append(_refuse_redirect)
    trace_configs.append(redirect_check)
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(resolver=PublicOnlyResolver()), trace_configs=trace_configs, **kwargs
    )


async def _refuse_redirect(_session, _context, params: aiohttp.TraceRequestRedirectParams) -> None:
    raise aiohttp.ClientConnectionError(f"{params.url} answered with a redirect, which is not followed")


def trusted_session(**kwargs) -> aiohttp.ClientSession:
    """A session for an address from the miner's own settings. Never for a node's address: use `executor_session`."""
    return aiohttp.ClientSession(**kwargs)
