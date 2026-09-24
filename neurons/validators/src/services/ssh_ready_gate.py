"""Before a pod is reported RUNNING, its SSH port must answer with an sshd identification line.

A pod can come up with nothing listening on port 22: an image-managed sshd that is still starting
(the validator skips its own sshd bootstrap for the Lium default image and cached templates) or a host
whose port forwarding is broken. docker-proxy accepts on the host's mapped port even when nothing
listens inside the container, so a TCP accept proves nothing; only the RFC 4253 §4.2 line does.

`SSH_READY_GATE_MODE`: `off` dials nothing; `log` probes after the create returns and writes one
structured line (the measurement mode, never fails and never delays a rent); `enforce` probes before
ContainerCreated and fails the create at `current_step = "ssh_ready"`.

lium-io#1372 (the rented-pod SSH probe) carries its own identification reader; once it merges, one of
the two helpers goes.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

SSH_ID_PREFIX = b"SSH-2.0-"
SSH_ID_ANY_VERSION_PREFIX = b"SSH-"
# RFC 4253 §4.2: the identification line is at most 255 bytes including CR LF, and a server may send
# other lines before it. Both bounds keep a peer that is not sshd from holding the probe or its memory.
SSH_ID_LINE_MAX = 255
SSH_PRE_BANNER_LINES_MAX = 64
SSH_READY_ATTEMPT_TIMEOUT_SECONDS = 5.0


class SshReadyMode(str, enum.Enum):
    OFF = "off"
    LOG = "log"
    ENFORCE = "enforce"


def ssh_ready_gate_mode(raw: str | None) -> SshReadyMode:
    """The configured mode; an unknown value is `off`, so a typo never fails rents."""
    try:
        return SshReadyMode((raw or "").strip().lower())
    except ValueError:
        return SshReadyMode.OFF


class SshReadyOutcome(str, enum.Enum):
    READY = "ready"
    REFUSED = "connection refused"
    TIMED_OUT = "timed out"
    UNREACHABLE = "unreachable"
    NO_BANNER = "no banner"


@dataclass(frozen=True)
class SshReadyResult:
    outcome: SshReadyOutcome
    attempts: int
    elapsed_ms: int

    @property
    def ready(self) -> bool:
        return self.outcome is SshReadyOutcome.READY


class SshNotReady(Exception):
    def __init__(self, port: int, grace_seconds: float, result: SshReadyResult):
        self.port = port
        self.result = result
        super().__init__(
            f"the pod's SSH port {port} did not answer with an SSH banner within "
            f"{grace_seconds:g} s ({result.outcome.value})"
        )


def is_ssh2_identification(line: bytes) -> bool:
    if not line.endswith(b"\n") or len(line) > SSH_ID_LINE_MAX:
        return False
    body = line.rstrip(b"\r\n")
    return body.startswith(SSH_ID_PREFIX) and len(body) > len(SSH_ID_PREFIX)


async def _read_identification(reader: asyncio.StreamReader) -> bytes:
    for _ in range(SSH_PRE_BANNER_LINES_MAX + 1):
        try:
            line = await reader.readuntil(b"\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, OSError):
            return b""
        if line.startswith(SSH_ID_ANY_VERSION_PREFIX):
            return line
    return b""


async def probe_ssh_banner(host: str, port: int, timeout: float) -> SshReadyOutcome:
    """One dial: READY when the peer sends an SSH-2.0 identification line within `timeout`."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, limit=SSH_ID_LINE_MAX), timeout
        )
    except ConnectionRefusedError:
        return SshReadyOutcome.REFUSED
    except asyncio.TimeoutError:
        return SshReadyOutcome.TIMED_OUT
    except OSError:
        return SshReadyOutcome.UNREACHABLE
    try:
        try:
            line = await asyncio.wait_for(_read_identification(reader), timeout)
        except asyncio.TimeoutError:
            return SshReadyOutcome.NO_BANNER
        return SshReadyOutcome.READY if is_ssh2_identification(line) else SshReadyOutcome.NO_BANNER
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def wait_for_ssh_banner(
    host: str,
    port: int,
    *,
    grace_seconds: float,
    poll_seconds: float,
    attempt_timeout_seconds: float = SSH_READY_ATTEMPT_TIMEOUT_SECONDS,
    probe: Callable[[str, int, float], Awaitable[SshReadyOutcome]] = probe_ssh_banner,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> SshReadyResult:
    """Dial until the banner arrives or `grace_seconds` pass; the result carries the last outcome."""
    started = clock()
    attempts = 0
    while True:
        remaining = grace_seconds - (clock() - started)
        outcome = await probe(host, port, max(0.5, min(attempt_timeout_seconds, remaining)))
        attempts += 1
        elapsed = clock() - started
        if outcome is SshReadyOutcome.READY or elapsed >= grace_seconds:
            return SshReadyResult(
                outcome=outcome, attempts=attempts, elapsed_ms=int(elapsed * 1000)
            )
        await sleep(min(poll_seconds, grace_seconds - elapsed))
