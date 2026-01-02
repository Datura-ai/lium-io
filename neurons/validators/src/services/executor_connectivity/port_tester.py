import asyncio

import aiohttp

from services.executor_connectivity.models import PortPair


class PortTester:
    """Tests port connectivity via HTTP. Stateless and easily mockable."""

    async def test_one(self, session: aiohttp.ClientSession, host: str, port: PortPair, token: str) -> bool:
        """Test single port connectivity via HTTP."""
        url = f"http://{host}:{port.external}/"
        expected = f"{token}:{port.internal}"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
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
    ) -> tuple[list[PortPair], list[PortPair]]:
        """Test multiple ports concurrently."""
        results = await asyncio.gather(*[self.test_one(session, host, p, token) for p in ports])
        successful = [p for p, ok in zip(ports, results) if ok]
        failed = [p for p, ok in zip(ports, results) if not ok]
        return successful, failed
