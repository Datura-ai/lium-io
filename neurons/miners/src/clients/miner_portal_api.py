import asyncio
import logging
import time
from typing import Any

import aiohttp

from core.config import settings
from core.utils import _m, get_extra_info
from protocol.miner_request import AuthenticateRequest

logger = logging.getLogger(__name__)

SNAPSHOT_TTL_SECONDS = 300
FAILED_REFRESH_RETRY_SECONDS = 30
BULK_FETCH_TIMEOUT_SECONDS = 15


class MinerPortalAPI:
    """Executor lookup backed by a portal-wide snapshot instead of per-hotkey requests.

    The validator wave asks for every opted-in miner within the same minute; fetching
    per hotkey turned that into ~40 concurrent portal requests and exhausted the portal
    DB pool (DAH-2469). Instead, one bulk request per TTL fills an in-memory snapshot
    {miner_hotkey: [executors]} that all callers read. A failed refresh keeps serving
    the previous snapshot, so a portal outage no longer looks like "miner has zero
    executors" (which zeroed the miner's whole fleet for the cycle).
    """

    _snapshot: dict[str, list[dict[str, Any]]] = {}
    _snapshot_fetched_at: float = 0.0
    _last_refresh_attempt_at: float = 0.0
    _refresh_lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def fetch_executors(
        cls, miner_hotkey: str, executor_id: str | None
    ) -> list[dict[str, Any]]:
        # serve one miner's executors from the shared snapshot, refreshing it when stale
        snapshot = await cls._get_snapshot()
        executors = snapshot.get(miner_hotkey, [])
        if executor_id:
            executors = [e for e in executors if str(e.get("id")) == str(executor_id)]
        return executors

    @classmethod
    def _is_fresh(cls) -> bool:
        return (
            bool(cls._snapshot)
            and time.monotonic() - cls._snapshot_fetched_at < SNAPSHOT_TTL_SECONDS
        )

    @classmethod
    async def _get_snapshot(cls) -> dict[str, list[dict[str, Any]]]:
        # single-flight refresh: the first caller past TTL fetches, the rest await it
        if cls._is_fresh():
            return cls._snapshot

        async with cls._refresh_lock:
            if cls._is_fresh():
                return cls._snapshot
            # after a failed refresh, don't hammer the portal on every call
            if time.monotonic() - cls._last_refresh_attempt_at < FAILED_REFRESH_RETRY_SECONDS:
                return cls._snapshot
            cls._last_refresh_attempt_at = time.monotonic()

            try:
                snapshot = await cls._fetch_bulk()
            except Exception as e:
                logger.error(
                    _m(
                        "Failed to refresh executor snapshot from portal - serving stale snapshot",
                        extra=get_extra_info(
                            {
                                "error": str(e),
                                "stale_hotkeys": len(cls._snapshot),
                                "stale_age_seconds": int(
                                    time.monotonic() - cls._snapshot_fetched_at
                                )
                                if cls._snapshot
                                else None,
                            }
                        ),
                    )
                )
                return cls._snapshot

            cls._snapshot = snapshot
            cls._snapshot_fetched_at = time.monotonic()
            logger.info(
                _m(
                    "Refreshed executor snapshot from portal",
                    extra=get_extra_info(
                        {
                            "hotkeys": len(snapshot),
                            "executors": sum(len(v) for v in snapshot.values()),
                        }
                    ),
                )
            )
            return cls._snapshot

    @classmethod
    async def _fetch_bulk(cls) -> dict[str, list[dict[str, Any]]]:
        # one portal request for every opted-in miner's executors, grouped by hotkey
        api_url = f"{settings.MINER_PORTAL_API_URL}/miners/executors"

        keypair = settings.get_bittensor_wallet().get_hotkey()
        auth = AuthenticateRequest.from_keypair(keypair)
        headers = {
            "hotkey": auth.payload.miner_hotkey,
            "timestamp": str(auth.payload.timestamp),
            "signature": auth.signature,
        }

        timeout = aiohttp.ClientTimeout(total=BULK_FETCH_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(api_url, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"portal bulk executors returned {resp.status}: {text[:200]}"
                    )
                data = await resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError(
                        f"unexpected portal bulk response shape: {type(data).__name__}"
                    )
                return data
