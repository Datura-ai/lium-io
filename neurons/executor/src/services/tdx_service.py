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

    @staticmethod
    def _parse_nonce(nonce: str | None) -> bytes | None:
        """Parse a validator-issued attestation nonce (hex, exactly 32 bytes)."""
        if not nonce:
            return None
        try:
            nonce_bytes = bytes.fromhex(nonce)
        except ValueError as exc:
            raise ValueError("Attestation nonce must be hex-encoded") from exc
        if len(nonce_bytes) != 32:
            raise ValueError("Attestation nonce must be 32 bytes")
        return nonce_bytes

    def _report_data(self, host_key: str, nonce_bytes: bytes | None = None) -> Tuple[str, bytes]:
        """Build report data: sha256("SSH_HOST_KEY:" + host_key) ‖ nonce.

        The host-key hash fills the first 32 bytes (identity — preserves the
        validator's prefix check); the validator-issued nonce fills bytes 32:64
        (freshness). Without a nonce the legacy 32-byte form is kept and dstack
        zero-pads the second half.
        """
        digest = hashlib.sha256(self.REPORT_PREFIX +
                                host_key.encode("utf-8")).hexdigest()
        report_bytes = bytes.fromhex(digest)
        if nonce_bytes is not None:
            report_bytes += nonce_bytes
        report_hex = f"0x{report_bytes.hex()}"
        return report_hex, report_bytes

    async def get_quote(self, host_key: str | None, nonce: str | None = None) -> str | None:
        """Return a TDX quote as a JSON string.

        Quotes are cached only for the legacy (nonce-less) path; a nonce-bound
        quote is freshly generated per attestation event by design — replaying a
        cached quote would fail the validator's report_data[32:64] check.
        """
        if not self.enabled:
            return None
        if not host_key:
            logger.warning(
                "TDX attestation enabled but SSH host key is unavailable")
            return None

        try:
            nonce_bytes = self._parse_nonce(nonce)
        except ValueError as exc:
            logger.error("Invalid attestation nonce received: %s", exc)
            return None

        report_hex, report_bytes = self._report_data(host_key, nonce_bytes)
        if nonce_bytes is None and self._cached_quote and self._cached_report_data == report_hex:
            return self._cached_quote

        try:
            client = await self._get_client()
            response = await client.get_quote(report_bytes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("TDX quote request failed: %s", exc)
            return None

        quote_json = response.model_dump_json()
        if nonce_bytes is None:
            self._cached_report_data = report_hex
            self._cached_quote = quote_json
        return quote_json
