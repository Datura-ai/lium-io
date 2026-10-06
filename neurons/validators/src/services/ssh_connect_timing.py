"""SSH connect phase timing (DAH-2272).

Splits a single ``asyncssh.connect`` into its phases **without changing how the
connection is established** — so we can attribute a slow connect to the network,
the remote sshd, or the validator. It does this by attaching an observation-only
``SSHClient`` (a documented asyncssh extension point) that timestamps the
connection lifecycle callbacks, then — when ``SSH_DEBUG_LOGGING`` is enabled
(off by default; DAH-2272) — emits one structured ``ssh_connect_phase_timing``
log line per connect.

Phases (each the gap between two asyncssh ``SSHClient`` callbacks):

* ``tcp_connect_ms`` — start → ``connection_made`` ("as soon as the TCP
  connection completes"). DNS resolution + TCP handshake → **network /
  remote-host reachability**.
* ``ssh_handshake_ms`` — ``connection_made`` → ``begin_auth`` ("authentication
  is about to begin"). Version-banner exchange + key exchange → the **remote
  sshd** (a ``UseDNS`` reverse-lookup banner stall lands here) + KEX crypto.
* ``ssh_auth_ms`` — ``begin_auth`` → ``auth_completed``. Authentication
  round-trips.
* ``ssh_login_ms`` — ``connection_made`` → ``auth_completed``. The whole
  post-TCP leg; always available even if ``begin_auth`` was skipped, so it is
  the robust fallback for ``ssh_handshake_ms + ssh_auth_ms``.
* ``total_connect_ms`` — start → connect returned.

Plus negotiated metadata read off the connection once it is up:
``server_version`` (the remote sshd's identification banner — e.g.
``SSH-2.0-OpenSSH_8.9p1``, the single most useful clue for a banner stall),
``cipher``/``mac``, any pre-auth ``auth_banner`` the server displayed, and any
server SSH ``DEBUG`` messages.

Reading the split: a large ``tcp_connect_ms`` (with an IP host, so no DNS) →
network path / host uplink; a large ``ssh_handshake_ms`` → remote sshd slow to
emit its banner / KEX; a large value across *all* hosts in the same window →
validator-side (event-loop) rather than remote. For a finer banner-vs-KEX
split, enable ``SSH_DEBUG_LOGGING`` (see ``core.utils``).

Only the *notification* callbacks (all return ``None``) are overridden; the
credential/host-key request callbacks are left to the base ``SSHClient`` so auth
behaviour is untouched.
"""

import inspect
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncssh

from core.config import settings
from core.utils import _m, get_extra_info
from payload_models.payloads import now_ms

logger = logging.getLogger(__name__)

_BANNER_MAX = 200


class _PhaseTimingSSHClient(asyncssh.SSHClient):
    """Observation-only client that records connect-phase marks.

    asyncssh constructs one of these per connection via ``client_factory`` and
    invokes the callbacks on the connection's own task, so the marks line up
    with the exact connect being measured. Every method here is a pure
    notification (returns ``None``); nothing influences the handshake or auth.
    """

    def __init__(self, marks: dict, log_extra: dict, host: object):
        self._marks = marks
        self._log_extra = log_extra
        self._host = host

    def connection_made(self, conn: asyncssh.SSHClientConnection) -> None:
        # Fired as soon as the TCP connection completes (pre-handshake).
        self._marks["tcp_done_ms"] = now_ms()

    def begin_auth(self, username: str) -> None:
        # Fired once version exchange + KEX are done and auth is about to start.
        self._marks["auth_begin_ms"] = now_ms()

    def auth_banner_received(self, msg: str, lang: str) -> None:
        # Optional server-sent pre-auth banner (many hosts send none).
        self._marks["auth_banner_ms"] = now_ms()
        self._marks["auth_banner"] = (msg or "")[:_BANNER_MAX]

    def auth_completed(self) -> None:
        # Fired once authentication has succeeded.
        self._marks["auth_done_ms"] = now_ms()

    def debug_msg_received(self, msg: str, lang: str, always_display: bool) -> None:
        # Rare: an SSH DEBUG message from the server. Capture for diagnostics.
        self._marks.setdefault("server_debug_msgs", []).append((msg or "")[:_BANNER_MAX])

    def connection_lost(self, exc: "Exception | None") -> None:
        # Fires at close — after the wrapper's summary line — so it is logged
        # on its own. Only surface abnormal drops (keepalive timeout, mid-deploy
        # reset); a clean close (exc is None) stays silent to avoid noise.
        if exc is not None:
            logger.warning(
                _m(
                    "ssh_connection_lost",
                    extra=get_extra_info(
                        {**self._log_extra, "host": self._host, "error": str(exc)}
                    ),
                )
            )


def _safe_extra(conn: object, key: str):
    """Read a negotiated value off the connection, tolerating absence/mocks.

    The real ``SSHClientConnection.get_extra_info`` returns a plain value; the
    coroutine guard only matters for mock connections, so an async-shaped result
    is discarded (and closed) rather than logged.
    """
    try:
        value = conn.get_extra_info(key)
    except Exception:
        return None
    if inspect.iscoroutine(value):
        value.close()
        return None
    return value


@asynccontextmanager
async def connect_with_phase_timing(
    *,
    log_extra: dict | None = None,
    **connect_kwargs,
) -> AsyncIterator[asyncssh.SSHClientConnection]:
    """``asyncssh.connect`` wrapper that logs the connect-phase breakdown.

    Drop-in replacement for ``async with asyncssh.connect(...) as ssh_client:``.
    All keyword arguments are forwarded verbatim to ``asyncssh.connect``; an
    internal ``client_factory`` is injected to capture phase marks. On exit the
    connection is closed exactly as the plain async-context-manager form would.

    A caller-supplied ``client_factory`` is not supported (no callsite passes
    one); doing so raises ``ValueError`` rather than silently dropping it.
    """
    if connect_kwargs.get("client_factory") is not None:
        raise ValueError("connect_with_phase_timing does not support a custom client_factory")

    log_extra = log_extra or {}
    marks: dict = {}
    host = connect_kwargs.get("host")
    t0 = now_ms()

    # Use the `async with` form so cleanup (close + wait_closed) is identical to
    # a plain `asyncssh.connect`. By the time `__aenter__` returns, the connect
    # has resolved and the phase callbacks have fired, so the marks are set.
    async with asyncssh.connect(
        client_factory=lambda: _PhaseTimingSSHClient(marks, log_extra, host),
        **connect_kwargs,
    ) as conn:
        # DAH-2272: the per-connect phase-timing line is diagnostic-only and off by
        # default (SSH_DEBUG_LOGGING). Skip building/emitting it when the flag is
        # off so normal operation stays quiet; the abnormal ``ssh_connection_lost``
        # warning in the client callback still fires regardless.
        if settings.SSH_DEBUG_LOGGING:
            end = now_ms()
            tcp_done = marks.get("tcp_done_ms")
            auth_begin = marks.get("auth_begin_ms")
            auth_done = marks.get("auth_done_ms")

            def _delta(a, b):
                return (b - a) if (a is not None and b is not None) else None

            extra = {
                **log_extra,
                "host": host,
                "port": connect_kwargs.get("port"),
                "tcp_connect_ms": _delta(t0, tcp_done),
                "ssh_handshake_ms": _delta(tcp_done, auth_begin),  # banner + KEX
                "ssh_auth_ms": _delta(auth_begin, auth_done),      # authentication
                "ssh_login_ms": _delta(tcp_done, auth_done),       # robust combined
                "total_connect_ms": end - t0,
                "server_version": _safe_extra(conn, "server_version"),
                "cipher": _safe_extra(conn, "recv_cipher"),
                "mac": _safe_extra(conn, "recv_mac"),
            }
            if marks.get("auth_banner"):
                extra["auth_banner"] = marks["auth_banner"]
            if marks.get("server_debug_msgs"):
                extra["server_debug_msgs"] = marks["server_debug_msgs"]

            logger.info(_m("ssh_connect_phase_timing", extra=get_extra_info(extra)))
        yield conn
