"""One SSH probe of a rented pod's mapped port, from outside the container.

``services/pod_ssh_probe.py`` calls it once per RUNNING rented pod at the start of each cycle. The
result is an observation the validator reports with the node's result, never a verdict: nothing here
changes a score.
"""

from __future__ import annotations

import asyncio
from typing import NamedTuple

from protocol.vc_protocol.validator_requests import PodSshResult

from .ssh_identification import SSH_ID_LINE_MAX, is_ssh2_identification, read_ssh_identification


class ConnectOutcome(NamedTuple):
    result: PodSshResult
    # The OS error behind a failed connect (ECONNREFUSED, EHOSTUNREACH, …); None otherwise.
    errno: int | None = None


async def tcp_connect_fault(host: str, port: int, timeout: float) -> ConnectOutcome:
    """Connect to ``host:port`` and read the server's identification line.

    ``banner`` is a complete ``SSH-2.0-`` line. The line is required because a mapped port is
    answered by docker-proxy on the host: it accepts even when nothing listens inside the container,
    then closes. The whole line is read (up to the LF; the stream buffer is capped at 255 bytes and a
    longer line is refused) before it is judged, so a prefix split across TCP segments is not a fault.

    ``timeout`` is one deadline for the connect and the read together. A connect that does not
    complete by then is ``timeout``; one the host turns away (any ``OSError``) is ``refused`` with its
    errno; a peer that accepts and then closes, sends something else, or keeps the rest of the
    deadline without a line is ``no_banner``.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    try:
        async with asyncio.timeout_at(deadline):
            reader, writer = await asyncio.open_connection(host, port, limit=SSH_ID_LINE_MAX)
    except TimeoutError:
        return ConnectOutcome(PodSshResult.TIMEOUT)
    except OSError as exc:
        return ConnectOutcome(PodSshResult.REFUSED, exc.errno)
    try:
        async with asyncio.timeout_at(deadline):
            line = await read_ssh_identification(reader)
    except TimeoutError:
        line = b""
    finally:
        writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return ConnectOutcome(
        PodSshResult.BANNER if is_ssh2_identification(line) else PodSshResult.NO_BANNER
    )
