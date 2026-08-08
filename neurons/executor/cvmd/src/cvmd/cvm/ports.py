"""Forwarded ports: parse them the way dstack does, and refuse a launch that would collide.

A forwarded port is the one piece of a CVM that reaches outside the host, so a collision is not
a startup nuisance — it is one CVM answering on another's address. QEMU's own `hostfwd` failure
arrives after the guest is already being built, inside a child process cvmd is not waiting on,
so the check belongs here, before anything is started.
"""

import errno
import socket
from dataclasses import dataclass

# Mirrors DStackManager._parse_port_mapping: "protocol[:address]:from:to", `from` being the
# host-side port and `to` the guest-side one, with 127.0.0.1 as the implied address.
DEFAULT_ADDRESS = "127.0.0.1"


class PortError(Exception):
    """A mapping cvmd cannot parse, or a host port it cannot claim."""


@dataclass(frozen=True)
class PortMapping:
    protocol: str
    address: str
    host_port: int
    guest_port: int

    def __str__(self) -> str:
        return f"{self.protocol}:{self.address}:{self.host_port}:{self.guest_port}"


def parse(spec: str) -> PortMapping:
    parts = spec.split(":")
    if len(parts) == 3:
        protocol, host_port, guest_port = parts
        address = DEFAULT_ADDRESS
    elif len(parts) == 4:
        protocol, address, host_port, guest_port = parts
    else:
        raise PortError(f"{spec!r} is not 'protocol[:address]:from:to'")

    try:
        host, guest = int(host_port), int(guest_port)
    except ValueError as exc:
        raise PortError(f"{spec!r} has a non-numeric port") from exc
    if not (0 < host < 65536 and 0 < guest < 65536):
        raise PortError(f"{spec!r} has a port outside 1-65535")
    return PortMapping(protocol=protocol.lower(), address=address, host_port=host, guest_port=guest)


def parse_all(specs) -> list[PortMapping]:
    mappings = [parse(spec) for spec in specs]

    claimed: dict[tuple[str, int], str] = {}
    for mapping in mappings:
        key = (mapping.protocol, mapping.host_port)
        if key in claimed:
            raise PortError(
                f"host port {mapping.host_port}/{mapping.protocol} is mapped twice "
                f"({claimed[key]} and {mapping}) — only one of them could ever receive traffic"
            )
        claimed[key] = str(mapping)
    return mappings


def _is_free(mapping: PortMapping) -> bool:
    """Can this host address and port be bound right now?

    SO_REUSEADDR is deliberately **not** set. It would let the bind succeed against a socket in
    TIME_WAIT, and against some sockets already bound to a more specific address — which is the
    exact case this check exists to catch. A port held only by TIME_WAIT reads as busy here,
    and refusing a launch a few seconds early is the safe direction to be wrong in.
    """
    family = socket.AF_INET6 if ":" in mapping.address else socket.AF_INET
    kind = socket.SOCK_DGRAM if mapping.protocol == "udp" else socket.SOCK_STREAM
    with socket.socket(family, kind) as probe:
        try:
            probe.bind((mapping.address, mapping.host_port))
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, errno.EACCES, errno.EADDRNOTAVAIL):
                return False
            raise
    return True


def assert_free(mappings: list[PortMapping]) -> None:
    """Raise naming every port already taken, not just the first.

    All of them, because an operator fixing one port at a time across three launch attempts is
    three chances to leave a CVM half-built.
    """
    taken = [str(mapping) for mapping in mappings if not _is_free(mapping)]
    if taken:
        raise PortError(
            f"these forwarded ports are already in use on this host: {', '.join(taken)}"
        )
