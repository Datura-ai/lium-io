import asyncio
import logging
import statistics
import threading
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
# open probe sockets across every node being verified; a probe waiting for one has not started its budget
PROBE_CONCURRENCY = 256
# a node's answer is "<token>:<port>"; reading more than this lets a node grow the validator's memory
PROBE_BODY_LIMIT = 256


class PortTester:
    """Tests port connectivity via HTTP on its own event loop in a daemon thread.

    Probes never run on the validator's loop: a wave of probe callbacks there delayed every other
    reply on it, so Redis replies sat unread and pool checkouts ran past their wait.
    """

    def __init__(self, concurrency: int = PROBE_CONCURRENCY):
        self._concurrency = concurrency
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: aiohttp.ClientSession | None = None  # touched only on the probe loop
        self._start_lock = threading.Lock()

    def _probe_loop(self) -> asyncio.AbstractEventLoop:
        with self._start_lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="port-probes", daemon=True).start()
                self._loop = loop
            return self._loop

    async def _on_probe_loop(self, coro):
        # cancelling the caller (its deadline) cancels the probes on the probe loop too
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self._probe_loop()))

    async def test_one(self, host: str, port: PortPair, token: str) -> bool:
        """Test single port connectivity via HTTP."""
        return await self._on_probe_loop(self._probe(host, port, token))

    async def _probe(self, host: str, port: PortPair, token: str) -> bool:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                connector=public_host_connector(limit=self._concurrency, force_close=True)
            )
        url = f"http://{host}:{port.external}/"
        expected = f"{token}:{port.internal}"

        try:
            # the node answers its own address only: a redirect would point the probe somewhere else. The answer is
            # asked for and read uncompressed, so the cap counts the bytes the node sent
            async with self._session.get(
                url,
                timeout=PROBE_TIMEOUT,
                allow_redirects=False,
                auto_decompress=False,
                headers={"Accept-Encoding": "identity"},
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
                return await self._probe(host, port, token)
            finally:
                durations.append(time.monotonic() - started)

        async def probe_all() -> list[bool]:
            return await asyncio.gather(*[timed(p) for p in ports])

        wall_started = time.monotonic()
        results = await self._on_probe_loop(probe_all())
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
