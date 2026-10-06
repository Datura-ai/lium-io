import asyncio
import logging
import statistics
import time

import aiohttp

from core.utils import _m, get_extra_info
from services.executor_connectivity.models import PortPair

logger = logging.getLogger(__name__)

# per-operation budget instead of a cumulative one: aiohttp's `total` clock starts before the
# connector hands out a socket, so a probe queued behind a full pool burned its budget waiting
# and was recorded as a dead port without ever putting a packet on the wire
PROBE_TIMEOUT = aiohttp.ClientTimeout(sock_connect=3, sock_read=3)


class PortTester:
    """Tests port connectivity via HTTP. Stateless and easily mockable."""

    async def test_one(self, session: aiohttp.ClientSession, host: str, port: PortPair, token: str) -> bool:
        """Test single port connectivity via HTTP."""
        url = f"http://{host}:{port.external}/"
        expected = f"{token}:{port.internal}"

        try:
            async with session.get(url, timeout=PROBE_TIMEOUT) as resp:
                text = await resp.text()
                return text.strip() == expected
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
