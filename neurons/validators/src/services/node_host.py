"""A node registered by hostname (dynamic DNS) is resolved before the validator dials it.

The hostname stays the node's address everywhere (its identity does not change when the IP behind
it does); this check only refuses a name that points anywhere but the public internet.
"""

import asyncio
import ipaddress
import socket


class NodeHostError(Exception):
    """The node's hostname does not resolve to public addresses only."""


def is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


async def require_public_host(host: str, port: int = 22) -> None:
    """Return for an IP literal (unchanged behaviour) or a name whose every address is global; raise NodeHostError otherwise."""
    if is_ip_literal(host):
        return
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise NodeHostError(f"{host} does not resolve: {exc}") from exc
    addresses = {ipaddress.ip_address(info[4][0]) for info in infos}
    if not addresses:
        raise NodeHostError(f"{host} resolves to no address")
    blocked = sorted(str(a) for a in addresses if not a.is_global)
    if blocked:
        raise NodeHostError(f"{host} resolves to a non-public address: {', '.join(blocked)}")
