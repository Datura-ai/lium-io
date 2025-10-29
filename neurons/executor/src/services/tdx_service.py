import asyncio
import hashlib
import logging
from typing import Optional, Tuple

from dstack_sdk import AsyncDstackClient

logger = logging.getLogger(__name__)


class TDXQuoteService:
    """Generates TDX quotes tied to the executor's SSH host key."""

    REPORT_PREFIX = b"SSH_HOST_KEY:"

    def __init__(self) -> None:
        from core.config import settings

        self.enabled: bool = bool(settings.ENABLE_TDX_ATTESTATION)
        self.timeout: int = settings.TDX_QUOTE_TIMEOUT
        self._cached_report_data: Optional[str] = None
        self._cached_quote: Optional[str] = None
        self._client: Optional[AsyncDstackClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> AsyncDstackClient:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = AsyncDstackClient(timeout=self.timeout)
        return self._client

    def _report_data(self, host_key: str) -> Tuple[str, bytes]:
        digest = hashlib.sha256(self.REPORT_PREFIX +
                                host_key.encode("utf-8")).hexdigest()
        report_hex = f"0x{digest}"
        report_bytes = bytes.fromhex(digest)
        return report_hex, report_bytes

    async def get_quote(self, host_key: Optional[str]) -> Optional[str]:
        """Return a cached or freshly generated TDX quote as a JSON string."""
        if not self.enabled:
            return None
        if not host_key:
            logger.warning(
                "TDX attestation enabled but SSH host key is unavailable")
            return None

        report_hex, report_bytes = self._report_data(host_key)
        if self._cached_quote and self._cached_report_data == report_hex:
            return self._cached_quote

        try:
            client = await self._get_client()
            response = await client.get_quote(report_bytes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("TDX quote request failed: %s", exc)
            return None

        quote_json = response.model_dump_json()
        self._cached_report_data = report_hex
        self._cached_quote = quote_json
        return quote_json
