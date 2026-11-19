"""HTTP to a node's registered address, never into the miner's own network.

A node's address is set by whoever registered the node. A name may resolve to a loopback, private or
cloud-metadata address (or change between lookups), which would make the miner send its requests to
services next to itself. Every miner request to a node goes through `executor_session()`: a name is
resolved once per connection, refused unless every address it resolves to is public, and the
connection is made to those resolved addresses (no second lookup). Redirects are never followed, so a
node cannot send the request on to another address. Same rule as the validator's
neurons/validators/src/services/node_host.py.

An IP literal is dialled as given, as the validator does: the node registry only accepts public
addresses, and an operator's own executor on a private network keeps working.
"""

import ipaddress
import socket

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult
from aiohttp.resolver import DefaultResolver


class NonPublicNodeAddress(aiohttp.ClientConnectionError):
    """The node's address name resolves to an address that is not public; nothing was dialled."""


def is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # a scoped IPv6 link-local carries %iface
    # judged as the IPv4 address it carries: Python before 3.13 calls ::ffff:127.0.0.1 global
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return ip.is_global and not ip.is_multicast  # Python counts global-scope multicast as global


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
    """A session for requests to nodes; pass allow_redirects=False on each request (aiohttp has no session default)."""
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(resolver=PublicOnlyResolver()), **kwargs)
