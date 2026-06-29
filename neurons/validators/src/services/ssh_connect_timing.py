"""SSH connect phase timing (DAH-2272).

Splits a single ``asyncssh.connect`` into its two coarse legs **without changing
how the connection is established** — so we can attribute a slow connect to the
remote host/network vs. the remote sshd:

* ``tcp_connect_ms`` — DNS resolution + TCP handshake (network / remote-host
  reachability). Measured from just before ``asyncssh.connect`` to asyncssh's
  ``SSHClient.connection_made`` callback, which fires "as soon as the TCP
  connection completes".
* ``ssh_login_ms`` — SSH banner exchange + key exchange + authentication
  (sshd-side + crypto). Measured from ``connection_made`` to
  ``SSHClient.auth_completed`` ("authentication completed successfully").

Reading the split:

* large ``tcp_connect_ms`` → the network path / remote host is slow or lossy
  (the validator's event loop is fine — see the same-window comparison in the
  DAH-2272 RCA).
* large ``ssh_login_ms`` → the remote sshd is slow to respond (classic
  ``UseDNS yes`` reverse-lookup stall, PAM) or KEX/auth is expensive.

For a finer banner-vs-KEX-vs-auth breakdown, enable the asyncssh debug log
(``DEBUG_SSH_DEBUG_LOGGING=true``); see ``core.utils.configure_logs_of_other_modules``.

This is observation-only: it attaches a timing ``client_factory`` (a documented
asyncssh extension point) to the real connect and emits one structured log
line. It never alters the socket, the connect options, or the failure behaviour
of the underlying connect, and on any internal error it degrades to a plain
``asyncssh.connect``.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncssh

from core.utils import _m, get_extra_info
from payload_models.payloads import now_ms

logger = logging.getLogger(__name__)


class _PhaseTimingSSHClient(asyncssh.SSHClient):
    """Records TCP-complete and auth-complete wall-clock marks for one connect.

    asyncssh constructs one of these per connection via ``client_factory`` and
    invokes the callbacks on the connection's own task, so the marks line up
    with the exact connect being measured.
    """

    def __init__(self, marks: dict):
        self._marks = marks

    def connection_made(self, conn: asyncssh.SSHClientConnection) -> None:
        # Fired as soon as the TCP connection completes (pre-handshake).
        self._marks["tcp_done_ms"] = now_ms()

    def auth_completed(self) -> None:
        # Fired once banner + KEX + auth have all succeeded.
        self._marks["auth_done_ms"] = now_ms()


@asynccontextmanager
async def connect_with_phase_timing(
    *,
    log_extra: dict | None = None,
    **connect_kwargs,
) -> AsyncIterator[asyncssh.SSHClientConnection]:
    """``asyncssh.connect`` wrapper that logs the TCP-vs-SSH-login split.

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
    t0 = now_ms()

    # Use the `async with` form so cleanup (close + wait_closed) is identical to
    # a plain `asyncssh.connect`. By the time `__aenter__` returns, the connect
    # has resolved and both phase callbacks have fired, so the marks are set.
    async with asyncssh.connect(
        client_factory=lambda: _PhaseTimingSSHClient(marks),
        **connect_kwargs,
    ) as conn:
        tcp_done = marks.get("tcp_done_ms")
        auth_done = marks.get("auth_done_ms")
        end = now_ms()
        tcp_ms = (tcp_done - t0) if tcp_done is not None else None
        login_ms = (
            (auth_done - tcp_done)
            if (auth_done is not None and tcp_done is not None)
            else None
        )
        logger.info(
            _m(
                "ssh_connect_phase_timing",
                extra=get_extra_info(
                    {
                        **log_extra,
                        "host": connect_kwargs.get("host"),
                        "port": connect_kwargs.get("port"),
                        "tcp_connect_ms": tcp_ms,
                        "ssh_login_ms": login_ms,
                        "total_connect_ms": end - t0,
                    }
                ),
            )
        )
        yield conn
