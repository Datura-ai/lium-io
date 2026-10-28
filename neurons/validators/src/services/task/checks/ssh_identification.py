"""The one rule for "is sshd answering on this port": the server's RFC 4253 §4.2 identification line.

Used from outside the container on a pod's mapped port, where docker-proxy accepts on the host even
when nothing listens inside, so an accept alone proves nothing: by the rented-pod probe each cycle
(rented_pod_ssh.py) and by the rental probe's wait for sshd (rental_probe.py).
"""

from __future__ import annotations

import asyncio

# RFC 4253 §4.2: `SSH-protoversion-softwareversion SP comments CR LF`, at most 255 bytes including
# CR LF. Only protoversion 2.0 counts: `SSH-1.5-` is a 1.x-only server; `SSH-1.99-` (RFC 4253
# §5.1) marks a server that also speaks 1.x, and both are refused.
SSH_ID_PREFIX = b"SSH-2.0-"
SSH_ID_LINE_MAX = 255
# The same section lets the server send other lines before its identification (each ending in CR LF,
# none starting with `SSH-`) and a client MUST be able to skip them. The scan is bounded: OpenSSH's
# client gives up after 1024 such lines; 64 is more than any pre-banner an sshd is configured to
# print, and a peer that is not sshd at all runs out of lines (or of the caller's deadline) long
# before it can hold the probe. Each skipped line is bounded to SSH_ID_LINE_MAX bytes as well.
SSH_PRE_BANNER_LINES_MAX = 64
SSH_ID_ANY_VERSION_PREFIX = b"SSH-"


def is_ssh2_identification(line: bytes) -> bool:
    """True for a complete RFC 4253 identification line of protocol version 2.0.

    Complete means terminated by LF (sshd sends CR LF) and no longer than 255 bytes; 2.0 means the
    line starts with ``SSH-2.0-`` and names a software version after it. ``SSH-1.99-``, ``SSH-1.5-``,
    a bare ``SSH-2.0-``, a line cut before its LF, or anything else is not the sshd a renter logs in to.
    """
    if not line.endswith(b"\n") or len(line) > SSH_ID_LINE_MAX:
        return False
    body = line.rstrip(b"\r\n")
    return body.startswith(SSH_ID_PREFIX) and len(body) > len(SSH_ID_PREFIX)


async def read_ssh_identification(reader: asyncio.StreamReader) -> bytes:
    """The server's identification line, or b"" when none arrives within the bounds.

    RFC 4253 §4.2 lets a server send other lines before ``SSH-...``; they are skipped, up to
    ``SSH_PRE_BANNER_LINES_MAX`` of them. Open the connection with ``limit=SSH_ID_LINE_MAX``: the
    limit bounds every line, so a peer whose LF sits past byte 255 raises LimitOverrunError instead
    of growing memory (an LF exactly at index 255 returns 256 bytes, which
    ``is_ssh2_identification`` refuses); EOF before the LF raises IncompleteReadError (docker-proxy's
    accept-then-close, or a line cut short). Both are "no identification line". The caller holds
    the deadline.
    """
    for _ in range(SSH_PRE_BANNER_LINES_MAX + 1):
        try:
            line = await reader.readuntil(b"\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, OSError):
            return b""
        if line.startswith(SSH_ID_ANY_VERSION_PREFIX):
            return line
    return b""
