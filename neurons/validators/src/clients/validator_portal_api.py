import logging
import time

import aiohttp
import bittensor
from pydantic import BaseModel

from core.config import settings
from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)


class OptedInMiner(BaseModel):
    """The fields the validator needs from one row of the portal's `/validators/opted-in`.

    All three are required in `OptInStatusResponse` (lium-miner-portal `src/dtos/miner.py`),
    so a record missing one means the portal is misbehaving. Nothing enforces that contract
    across the two repos, and a failed row fails the whole fetch - hence only the fields
    that are actually read, and `miner_coldkey` is left to the ignored extras.
    """

    miner_hotkey: str
    central_miner_ip: str
    central_miner_port: int


class ValidatorPortalAPI:
    """Opted-in miner list, with the last successful portal answer as a fallback.

    Every failure used to return an empty list, indistinguishable from "nobody opted
    in": `SubtensorClient.fetch_miners` then skipped the central-miner axon override
    and the opt-in miners dropped out of the validated fleet (DAH-2518).

    Not a cache in the `MinerPortalAPI` sense - every call still hits the portal, and
    the stored answer only serves failures. It has no expiry, so a long outage keeps
    serving an arbitrarily old list; `cached_age_seconds` on the failure log is the
    only signal of how old.
    """

    _last_good_opted_in_miners: list[OptedInMiner] | None = None
    _last_good_fetched_at: float | None = None

    @classmethod
    async def get_opted_in_miners(cls) -> list[OptedInMiner] | None:
        """Fetch the opted-in miners, falling back to the last successful answer.

        Returns None when the fetch failed and no populated answer was ever cached -
        callers must not read that as "zero opted-in miners". An unconfigured portal
        URL still returns an empty list: there is nothing to route to.
        """
        api_base = (settings.MINER_PORTAL_REST_API_URL or "").rstrip("/")
        if not api_base:
            # the setting defaults to the real portal, so an empty one is a broken deploy,
            # and without this line it is the only way to lose the fleet without a trace
            logger.warning(
                _m(
                    "MINER_PORTAL_REST_API_URL is empty - opt-in routing is off this cycle",
                    extra=get_extra_info({}),
                )
            )
            return []

        url = f"{api_base}/validators/opted-in"

        try:
            miners = await cls._fetch_opted_in_miners(url)
        except Exception as error:
            cls._log_fetch_failure(url, error)
            return cls._last_good_opted_in_miners

        if miners:
            cls._last_good_opted_in_miners = miners
            cls._last_good_fetched_at = time.monotonic()
            return miners

        if cls._last_good_opted_in_miners:
            # a sudden "nobody is opted in" is far more likely a portal bug than a real
            # mass opt-out, and accepting it would drop every opt-in miner at once
            logger.warning(
                _m(
                    "Portal returned an empty opted-in list while a populated one is cached"
                    " - keeping the cached list",
                    extra=get_extra_info({
                        "url": url,
                        "cached_miners": len(cls._last_good_opted_in_miners),
                        "cached_age_seconds": cls._seconds_since_last_good_fetch(),
                    }),
                )
            )
            return cls._last_good_opted_in_miners

        # an empty answer is deliberately not stored: it would make every later failure
        # return that empty list instead of None, which is exactly the "an outage looks
        # like nobody opted in" confusion this class exists to remove
        return []

    @classmethod
    def _log_fetch_failure(cls, url: str, error: Exception) -> None:
        # the caller reports the consequence, so this stays neutral - both layers claiming
        # "opt-in miners will be MISSING" double-counts one outage in the ERROR alerts
        has_cached_miners = cls._last_good_opted_in_miners is not None
        if has_cached_miners:
            message = "Failed to fetch opted-in miners from portal - serving the last good list"
        else:
            message = "Failed to fetch opted-in miners from portal and nothing is cached"
        logger.error(
            _m(
                message,
                extra=get_extra_info({
                    "url": url,
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "cached_miners": len(cls._last_good_opted_in_miners) if has_cached_miners else None,
                    "cached_age_seconds": cls._seconds_since_last_good_fetch(),
                }),
            ),
            exc_info=True,
        )

    @classmethod
    def _seconds_since_last_good_fetch(cls) -> int | None:
        if cls._last_good_fetched_at is None:
            return None
        return int(time.monotonic() - cls._last_good_fetched_at)

    @staticmethod
    async def _fetch_opted_in_miners(url: str) -> list[OptedInMiner]:
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
                    error_body = await resp.text()
                    raise RuntimeError(f"portal returned status {resp.status}: {error_body[:500]}")

                records = await resp.json()
                if not isinstance(records, list):
                    raise RuntimeError(f"portal returned {type(records).__name__}, expected a list")
                return [OptedInMiner.model_validate(record) for record in records]
