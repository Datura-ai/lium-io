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
    _snapshot_fetched_at: float | None = None
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
            # case-insensitive: the replaced DB-side filter compared UUIDs, not strings
            wanted_id = str(executor_id).lower()
            executors = [e for e in executors if str(e.get("id")).lower() == wanted_id]
        return executors

    @classmethod
    def _is_fresh(cls) -> bool:
        return (
            cls._snapshot_fetched_at is not None
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
                snapshot = await cls._fetch_bulk_snapshot()
            except asyncio.CancelledError:
                # bypasses `except Exception`; without this log a cancelled refresh
                # (validator dropped the connection mid-fetch) would leave no trace
                logger.warning(
                    _m("Executor snapshot refresh cancelled mid-fetch", extra=get_extra_info({}))
                )
                raise
            except Exception as e:
                had_snapshot = cls._snapshot_fetched_at is not None
                logger.error(
                    _m(
                        "Failed to refresh executor snapshot from portal - serving stale snapshot"
                        if had_snapshot
                        else "No executor snapshot available and portal refresh failed"
                        " - serving EMPTY executor list for ALL miners",
                        extra=get_extra_info(
                            {
                                "url": cls._bulk_snapshot_url(),
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "stale_hotkeys": len(cls._snapshot),
                                "stale_age_seconds": int(
                                    time.monotonic() - cls._snapshot_fetched_at
                                )
                                if cls._snapshot_fetched_at is not None
                                else None,
                            }
                        ),
                    ),
                    exc_info=True,
                )
                return cls._snapshot

            if not snapshot and cls._snapshot:
                # a sudden "no opted-in miners at all" is far more likely a portal bug than
                # a real mass opt-out; replacing the snapshot would zero every fleet
                logger.warning(
                    _m(
                        "Portal returned an empty bulk snapshot while a populated one exists"
                        " - keeping the previous snapshot",
                        extra=get_extra_info(
                            {
                                "previous_hotkeys": len(cls._snapshot),
                                "stale_age_seconds": int(
                                    time.monotonic() - cls._snapshot_fetched_at
                                )
                                if cls._snapshot_fetched_at is not None
                                else None,
                            }
                        ),
                    )
                )
                cls._snapshot_fetched_at = time.monotonic()
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
    def _bulk_snapshot_url(cls) -> str:
        return f"{settings.MINER_PORTAL_API_URL}/miners/executors"

    @classmethod
    async def _fetch_bulk_snapshot(cls) -> dict[str, list[dict[str, Any]]]:
        # one portal request for every opted-in miner's executors, grouped by hotkey
        api_url = cls._bulk_snapshot_url()

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
                # reject anything but {hotkey: [executor, ...]}: caching e.g. an
                # ApiResponse envelope would silently zero every fleet for a full TTL
                if not isinstance(data, dict) or not all(
                    isinstance(executors, list) for executors in data.values()
                ):
                    raise RuntimeError(
                        f"unexpected portal bulk response shape: {type(data).__name__}"
                    )
                return data
