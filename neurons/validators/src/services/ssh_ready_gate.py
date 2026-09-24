"""Before a pod is reported RUNNING, its SSH port must answer with an sshd identification line.

A pod can come up with nothing listening on port 22: an image-managed sshd that is still starting
(the validator skips its own sshd bootstrap for the Lium default image and cached templates) or a host
whose port forwarding is broken. docker-proxy accepts on the host's mapped port even when nothing
listens inside the container, so a TCP accept proves nothing; only the RFC 4253 §4.2 line does.

`SSH_READY_GATE_MODE`: `off` dials nothing; `log` probes after the create returns and writes one
structured line (the measurement mode, never fails and never delays a rent); `enforce` probes before
ContainerCreated and fails the create at `current_step = "ssh_ready"`.

The banner rule is the one the rented-pod probe and the rental probe use (task/checks/ssh_identification.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from services.task.checks.ssh_identification import (
    SSH_ID_LINE_MAX,
    is_ssh2_identification,
    read_ssh_identification,
)

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
    CANCELLED = "cancelled by delete"


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


async def probe_ssh_banner(host: str, port: int, timeout: float) -> SshReadyOutcome:
    """One dial: READY when the peer sends an SSH-2.0 identification line. `timeout` covers the
    connect and the banner read together."""
    deadline = time.monotonic() + timeout
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
            line = await asyncio.wait_for(
                read_ssh_identification(reader), max(0.0, deadline - time.monotonic())
            )
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
    stop: Callable[[], bool] | None = None,
    probe: Callable[[str, int, float], Awaitable[SshReadyOutcome]] = probe_ssh_banner,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> SshReadyResult:
    """Dial until the banner arrives, `grace_seconds` pass or `stop()` says the create was cancelled.

    Each dial is capped at what is left of the grace period, so the whole wait never exceeds it. The
    result carries the last outcome, or CANCELLED when `stop()` ended it (checked before every dial
    and every sleep)."""
    started = clock()
    attempts = 0
    outcome = SshReadyOutcome.TIMED_OUT
    while True:
        if stop is not None and stop():
            outcome = SshReadyOutcome.CANCELLED
            break
        remaining = grace_seconds - (clock() - started)
        if remaining <= 0:
            break
        outcome = await probe(host, port, min(attempt_timeout_seconds, remaining))
        attempts += 1
        if outcome is SshReadyOutcome.READY:
            break
        remaining = grace_seconds - (clock() - started)
        if remaining <= 0:
            break
        if stop is not None and stop():
            outcome = SshReadyOutcome.CANCELLED
            break
        await sleep(min(poll_seconds, remaining))
    return SshReadyResult(
        outcome=outcome, attempts=attempts, elapsed_ms=int((clock() - started) * 1000)
    )
