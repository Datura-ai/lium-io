"""One SSH banner read: does `host:port` answer the way a renter's ssh client expects?

Shared by the synthetic rental probe (rental_probe.py, an idle node) and the rented-state check
(rented_machine.py, a renter's live pod). A TCP accept alone proves nothing: docker-proxy accepts on a
published port as soon as the container exists and closes the connection when nothing listens inside.
The identification string (RFC 4253 §4.2) is sshd's first write on every connection, so it is what a
renter's client waits for too.
"""
from __future__ import annotations

import asyncio

# the identification string sshd sends first on every connection (RFC 4253 §4.2)
SSH_BANNER_PREFIX = b"SSH-2.0"

# how the banner read failed; `ssh_failure` on the rented-state event and in the reset evidence
BANNER_REFUSED = "refused"
BANNER_TIMEOUT = "timeout"
BANNER_NO_BANNER = "no_banner"
BANNER_UNREACHABLE = "unreachable"


async def ssh_banner_error(
    host: str, port: int, *, connect_timeout: float, banner_timeout: float
) -> tuple[str, str] | None:
    """None once `host:port` sends sshd's `SSH-2.0` banner, else `(kind, detail)`.

    kind is one of BANNER_REFUSED (nothing listens on the host port), BANNER_TIMEOUT (connect or banner
    read hit its timeout), BANNER_NO_BANNER (the port accepted and sent something else, or closed with no
    bytes: docker-proxy in front of a container with no sshd) or BANNER_UNREACHABLE (any other socket error:
    host unreachable, reset, no route).
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=connect_timeout
        )
    except ConnectionRefusedError as exc:
        return BANNER_REFUSED, repr(exc)
    except TimeoutError:
        return BANNER_TIMEOUT, f"no TCP connection to port {port} within {connect_timeout} s"
    except OSError as exc:
        return BANNER_UNREACHABLE, repr(exc)
    try:
        try:
            banner = await asyncio.wait_for(reader.readline(), timeout=banner_timeout)
        except TimeoutError:
            return BANNER_TIMEOUT, f"port {port} accepted the connection but sent no banner within {banner_timeout} s"
        except (OSError, ValueError) as exc:
            # ValueError: readline's line limit, a server that talks but not SSH
            return BANNER_NO_BANNER, repr(exc)
    finally:
        writer.close()
    if banner.startswith(SSH_BANNER_PREFIX):
        return None
    return (
        BANNER_NO_BANNER,
        f"port {port} accepted the connection but sent no SSH banner "
        f"({banner[:40]!r}; docker-proxy answers before sshd listens)",
    )
