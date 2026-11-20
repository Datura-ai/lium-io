import asyncio
import logging
import socket
import statistics
import time

import aiohttp

from core.utils import _m, get_extra_info
from services.executor_connectivity.models import PortPair
from services.node_host import public_host_connector

logger = logging.getLogger(__name__)

# per-operation budget instead of a cumulative one: aiohttp's `total` clock starts before the
# connector hands out a socket, so a probe queued behind a full pool burned its budget waiting
# and was recorded as a dead port without ever putting a packet on the wire
PROBE_TIMEOUT = aiohttp.ClientTimeout(sock_connect=3, sock_read=3)
# a node's answer is "<token>:<port>"; reading more than this lets a node grow the validator's memory
PROBE_BODY_LIMIT = 256
# What a probe socket holds before the probe reads it: the kernel takes at most this much of an answer off the wire (a
# node can send no more until the probe reads), and the client parser pauses the socket once a few hundred bytes wait
PROBE_RECEIVE_BUFFER = 4096
# the answer's headers stay small too (aiohttp's defaults let one answer's headers reach about a megabyte)
PROBE_ANSWER_LIMITS = {
    "read_bufsize": PROBE_BODY_LIMIT + 1,
    "max_line_size": 1024,
    "max_field_size": 1024,
    "max_headers": 32,
}


def _probe_socket(addr_info) -> socket.socket:
    family, type_, proto, _, _ = addr_info
    sock = socket.socket(family=family, type=type_, proto=proto)
    # set before connect, so the receive window the node sees is this small from the first packet
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, PROBE_RECEIVE_BUFFER)
    return sock


def probe_connector(**kwargs) -> aiohttp.TCPConnector:
    """The connector for port probes: the node-address rule of public_host_connector, on small receive buffers."""
    return public_host_connector(socket_factory=_probe_socket, **kwargs)


class PortTester:
    """Tests port connectivity via HTTP. Stateless and easily mockable."""

    async def test_one(self, session: aiohttp.ClientSession, host: str, port: PortPair, token: str) -> bool:
        """Test single port connectivity via HTTP."""
        url = f"http://{host}:{port.external}/"
        expected = f"{token}:{port.internal}"

        try:
            # the node answers its own address only: a redirect would point the probe somewhere else. The answer is
            # asked for and read uncompressed, so the cap counts the bytes the node sent
            async with session.get(
                url,
                timeout=PROBE_TIMEOUT,
                allow_redirects=False,
                auto_decompress=False,
                headers={"Accept-Encoding": "identity"},
                **PROBE_ANSWER_LIMITS,
            ) as resp:
                body = b""
                while len(body) <= PROBE_BODY_LIMIT:
                    chunk = await resp.content.read(PROBE_BODY_LIMIT + 1 - len(body))
                    if not chunk:
                        break
                    body += chunk
                if len(body) > PROBE_BODY_LIMIT:
                    return False
                return body.decode(errors="replace").strip() == expected
        except Exception:
            return False

    async def test_many(
        self,
        session: aiohttp.ClientSession,
        host: str,
        ports: list[PortPair],
        token: str,
        log_ctx: dict | None = None,
    ) -> tuple[list[PortPair], list[PortPair]]:
        """Test multiple ports concurrently."""
        durations: list[float] = []

        async def timed(port: PortPair) -> bool:
            started = time.monotonic()
            try:
                return await self.test_one(session, host, port, token)
            finally:
                durations.append(time.monotonic() - started)

        wall_started = time.monotonic()
        results = await asyncio.gather(*[timed(p) for p in ports])
        wall = time.monotonic() - wall_started

        successful = [p for p, ok in zip(ports, results) if ok]
        failed = [p for p, ok in zip(ports, results) if not ok]

        sorted_durations = sorted(durations)
        p95 = sorted_durations[min(int(len(sorted_durations) * 0.95), len(sorted_durations) - 1)]
        logger.info(
            _m(
                f"probed {len(ports)} ports in {wall:.2f}s: {len(successful)} ok, "
                f"per-probe med={statistics.median(sorted_durations):.2f}s p95={p95:.2f}s max={sorted_durations[-1]:.2f}s",
                extra=get_extra_info(log_ctx or {}),
            )
        )
        return successful, failed
