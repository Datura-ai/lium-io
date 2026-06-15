"""DAH-2211 validator -> backend build-log ingestion (best-effort, fire-and-forget).

The backend's `POST /pods/{pod_id}/build-log-chunk` route persists submitted
build-log chunks to `Pod.build_log` so that `GET /pods/{pod_id}/logs` can
replay them after the build completes (success OR failure) without an active
SSE connection — AC-4 / §3.B.2 re-readability.

Auth scheme reused from `latest-set-weights` (utils.auth `_verify_validator_headers`):
    hotkey:    validator SS58 address (must match config.DEFAULT_VALIDATOR_HOTKEY)
    timestamp: unix-seconds string, must be within 5 minutes
    signature: hex-encoded sign of the timestamp string, "0x"-prefixed

The shipper runs as a per-build asyncio background task. Lines flow in via
`append()`; the flusher batches by line count, byte size, or a 1-second tick
and POSTs each batch. On any HTTP / network / signing error we log once and
drop the chunk — live SSE still has the lines, and a failing backend must
not slow or fail the build itself.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID

import aiohttp

from core.utils import _m, get_extra_info

if TYPE_CHECKING:
    import bittensor

logger = logging.getLogger(__name__)


# Batch thresholds — flush whenever ANY of these fires.
_BATCH_MAX_LINES = 50
_BATCH_MAX_BYTES = 8 * 1024
_BATCH_MAX_INTERVAL_SECONDS = 1.0
# Bounded queue — drop overflow rather than memory-spike under a runaway build.
# Live SSE still has the lines via the existing stream_log path.
_QUEUE_MAXSIZE = 4096
# POST timeout — backend latency must never block the build pipeline.
_POST_TIMEOUT_SECONDS = 5.0


class BuildLogShipper:
    """Per-build async log shipper. Construct, `await start()`, `await append(...)`
    for each line, `await stop()` when the build finishes.
    """

    def __init__(
        self,
        pod_id: str | UUID,
        keypair: "bittensor.Keypair | None" = None,
        backend_url: str | None = None,
    ) -> None:
        """`keypair` and `backend_url` default to the validator's configured
        wallet hotkey and `settings.COMPUTE_REST_API_URL`. Any init failure
        (missing wallet, missing setting) leaves the shipper disabled —
        append() and stop() become no-ops; the build is never blocked.
        """
        self._pod_id = str(pod_id)
        if keypair is None or backend_url is None:
            try:
                from core.config import settings as _settings

                if keypair is None:
                    keypair = _settings.get_bittensor_wallet().get_hotkey()
                if backend_url is None:
                    backend_url = getattr(_settings, "COMPUTE_REST_API_URL", None)
            except Exception:
                logger.warning(
                    _m(
                        "BuildLogShipper init failed to resolve wallet/url; ingestion disabled",
                        extra=get_extra_info({"pod_id": self._pod_id}),
                    ),
                    exc_info=True,
                )
                keypair = None
                backend_url = None
        self._keypair = keypair
        self._backend_url = (backend_url or "").rstrip("/") or None
        self._queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._task: asyncio.Task | None = None
        self._seq = 0
        # Suppress noise: log at most one warning per shipper lifecycle.
        self._post_warning_logged = False
        self._dropped_lines = 0

    async def start(self) -> None:
        if self._task is not None:
            return
        if not self._backend_url:
            # No backend URL configured — never ship; append() becomes a no-op
            # via the same put_nowait drop semantics below.
            return
        self._task = asyncio.create_task(self._flush_loop())

    async def append(self, line: str, phase: str = "build") -> None:
        """Enqueue one log line. Best-effort: silently drops under back-pressure."""
        if self._task is None or self._task.done():
            return
        try:
            self._queue.put_nowait((line, phase))
        except asyncio.QueueFull:
            self._dropped_lines += 1

    async def stop(self) -> None:
        """Signal the flusher to drain remaining lines and exit."""
        if self._task is None:
            return
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            # Pull one item to make room for the sentinel — losing one line is
            # fine; otherwise we hang the build on a full queue.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass
        try:
            await asyncio.wait_for(self._task, timeout=_POST_TIMEOUT_SECONDS + 1.0)
        except (TimeoutError, asyncio.TimeoutError):
            self._task.cancel()
        if self._dropped_lines:
            logger.warning(
                _m(
                    "BuildLogShipper dropped lines under back-pressure",
                    extra=get_extra_info({
                        "pod_id": self._pod_id,
                        "dropped": self._dropped_lines,
                    }),
                )
            )

    async def _flush_loop(self) -> None:
        buffer: list[str] = []
        buffer_bytes = 0
        buffer_phase = "build"
        last_flush = time.monotonic()

        async def flush() -> None:
            nonlocal buffer, buffer_bytes, last_flush
            if not buffer:
                return
            chunk = "\n".join(buffer) + "\n"
            await self._post_chunk(chunk, buffer_phase)
            buffer = []
            buffer_bytes = 0
            last_flush = time.monotonic()

        while True:
            timeout = max(0.0, _BATCH_MAX_INTERVAL_SECONDS - (time.monotonic() - last_flush))
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout if buffer else None)
            except (TimeoutError, asyncio.TimeoutError):
                await flush()
                continue

            if item is None:
                await flush()
                return

            line, phase = item
            # Phase shift mid-buffer → flush the prior phase, start fresh.
            if buffer and phase != buffer_phase:
                await flush()
            buffer_phase = phase
            buffer.append(line)
            buffer_bytes += len(line.encode("utf-8")) + 1  # +1 for the joining "\n"
            if len(buffer) >= _BATCH_MAX_LINES or buffer_bytes >= _BATCH_MAX_BYTES:
                await flush()

    async def _post_chunk(self, chunk: str, phase: str) -> None:
        if not self._backend_url:
            return
        self._seq += 1
        url = f"{self._backend_url}/pods/{self._pod_id}/build-log-chunk"
        try:
            timestamp = str(int(time.time()))
            signature = "0x" + self._keypair.sign(timestamp.encode()).hex()
            headers = {
                "hotkey": self._keypair.ss58_address,
                "timestamp": timestamp,
                "signature": signature,
            }
            payload = {"chunk": chunk, "seq": self._seq, "phase": phase}
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_POST_TIMEOUT_SECONDS)
            ) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status >= 400:
                        self._log_post_failure(f"http {resp.status}")
        except Exception as exc:
            self._log_post_failure(str(exc))

    def _log_post_failure(self, reason: str) -> None:
        if self._post_warning_logged:
            return
        self._post_warning_logged = True
        logger.warning(
            _m(
                "BuildLogShipper POST failed (best-effort, dropping further warnings)",
                extra=get_extra_info({
                    "pod_id": self._pod_id,
                    "reason": reason,
                }),
            )
        )
