import logging
import time

import aiohttp
import bittensor

from core.config import settings
from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)


class ValidatorPortalAPI:
    """Opted-in miner list, with the last successful portal answer as a fallback.

    Every failure used to return an empty list, which is indistinguishable from
    "nobody opted in": `SubtensorClient.fetch_miners` then skipped the central-miner
    axon override and the opt-in miners silently dropped out of the validated fleet
    (165 -> 137 miners, 372 -> 268 executors during the 2026-07-28 portal outage).
    Keeping the last good answer in memory turns a portal outage into stale-but-correct
    routing instead. Mirrors the miner-side snapshot cache in `MinerPortalAPI`.
    """

    _last_good: list[dict] | None = None
    _last_good_fetched_at: float | None = None

    @classmethod
    async def get_opted_in_miners(cls) -> list[dict] | None:
        """Fetch list of miners that have opted in.

        Returns list of dicts with shape:
            [
                {
                    "miner_hotkey": str,
                    "miner_coldkey": str,
                    "central_miner_ip": str,
                    "central_miner_port": int,
                },
                ...
            ]
        On a failed fetch returns the last successful list instead. Returns None when the
        fetch failed and nothing was ever cached - callers must not read that as "zero
        opted-in miners".
        """
        api_base = settings.MINER_PORTAL_REST_API_URL.rstrip("/") if settings.MINER_PORTAL_REST_API_URL else ""
        if not api_base:
            # a blank portal URL is a deliberate opt-out of portal routing, not an outage
            return []

        url = f"{api_base}/validators/opted-in"

        try:
            miners = await cls._fetch(url)
        except Exception as e:
            cls._log_fetch_failure(url, e)
            return cls._last_good

        if not miners and cls._last_good:
            # a sudden "nobody is opted in" is far more likely a portal bug than a real mass
            # opt-out; accepting it would drop every opt-in miner from the fleet at once.
            # The age deliberately keeps counting from the last real answer.
            logger.warning(
                _m(
                    "Portal returned an empty opted-in list while a populated one is cached"
                    " - keeping the cached list",
                    extra=get_extra_info({
                        "url": url,
                        "cached_miners": len(cls._last_good),
                        "cached_age_seconds": cls._last_good_age_seconds(),
                    }),
                )
            )
            return cls._last_good

        cls._last_good = miners
        cls._last_good_fetched_at = time.monotonic()
        return miners

    @classmethod
    def _log_fetch_failure(cls, url: str, error: Exception) -> None:
        # only a failure with nothing cached can still shrink the validated fleet
        has_cache = cls._last_good is not None
        logger.error(
            _m(
                "Failed to fetch opted-in miners from portal - serving the last good list"
                if has_cache
                else "Failed to fetch opted-in miners from portal and nothing is cached"
                " - opt-in miners will be MISSING this cycle",
                extra=get_extra_info({
                    "url": url,
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "cached_miners": len(cls._last_good) if has_cache else None,
                    "cached_age_seconds": cls._last_good_age_seconds(),
                }),
            ),
            exc_info=True,
        )

    @classmethod
    def _last_good_age_seconds(cls) -> int | None:
        if cls._last_good_fetched_at is None:
            return None
        return int(time.monotonic() - cls._last_good_fetched_at)

    @staticmethod
    async def _fetch(url: str) -> list[dict]:
        # raises on any portal problem, so the caller decides between fresh and cached
        keypair: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()

        timestamp = int(time.time())
        headers = {
            "hotkey": keypair.ss58_address,
            "timestamp": str(timestamp),
            "signature": f"0x{keypair.sign(str(timestamp)).hex()}",
        }

        # Generous total timeout so we survive short event-loop stalls from concurrent
        # sync bittensor/subtensor calls in this process. aiohttp's timer is driven by
        # the event loop, so a 10s cap fires spuriously whenever the loop stays blocked.
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"portal returned status {resp.status}: {body[:500]}")

                data = await resp.json()
                if not isinstance(data, list):
                    raise RuntimeError(f"portal returned {type(data).__name__}, expected a list")
                return data
