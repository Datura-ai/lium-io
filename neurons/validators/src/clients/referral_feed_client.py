"""Fail-closed client for the backend's epoch-stable referral-weights feed (DAH-2251).

The backend serves ``<COMPUTE_REST_API_URL>/v1/referral-weights`` shaped like::

    {"epoch_index": <int>, "as_of_block": <int|null>,
     "weights": {"<hotkey_ss58>": "<ema_decimal_string>", ...}}

This client never raises: any transport error, non-2xx response, malformed body, or
stale epoch collapses to an empty result, so a feed problem simply means no referral
emission that cycle rather than blocking or corrupting weight-setting.
"""
from decimal import Decimal, InvalidOperation

import aiohttp

from core.config import settings
from core.utils import _m, get_extra_info, get_logger

logger = get_logger(__name__)

# Generous total timeout, matching ValidatorPortalAPI: aiohttp's timer is driven by the
# event loop, which concurrent sync bittensor/subtensor calls in this process can stall,
# so a tight cap fires spuriously. Waiting is cheaper than the alternative here -- a
# spurious timeout fails closed, costing a whole cycle of referral emission, and the
# caller runs once per weight-set cycle (~72 min), not on a hot path.
_FEED_TIMEOUT_SECONDS = 60


class ReferralFeedClient:
    """Fetches referrer hotkey -> EMA weight from the backend referral feed."""

    async def get_weights(self, current_epoch: int | None = None) -> dict[str, float]:
        """Fetch and parse the referral-weights feed.

        Returns ``{hotkey_ss58: ema_float}`` on success, or ``{}`` on any failure
        (transport error, bad status, malformed JSON, missing/empty weights, or a
        staleness violation against ``current_epoch``). Entries whose EMA value is
        ``<= 0`` or fails to parse are dropped rather than failing the whole fetch.
        """
        # Derived from COMPUTE_REST_API_URL unless REFERRAL_FEED_URL overrides it, so the
        # feed follows the deployment instead of always resolving to prod.
        url = settings.get_referral_feed_url()
        try:
            timeout = aiohttp.ClientTimeout(total=_FEED_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    body = await response.json()

            weights = body.get("weights")
            if not weights:
                logger.warning(
                    _m(
                        "[get_weights] Referral feed returned no weights",
                        extra=get_extra_info({"url": url}),
                    ),
                )
                return {}

            epoch_index = body["epoch_index"]
            if current_epoch is not None:
                staleness = current_epoch - int(epoch_index)
                if staleness > settings.REFERRAL_FEED_MAX_STALENESS_EPOCHS:
                    logger.warning(
                        _m(
                            "[get_weights] Referral feed is stale, ignoring",
                            extra=get_extra_info(
                                {
                                    "epoch_index": epoch_index,
                                    "current_epoch": current_epoch,
                                    "staleness": staleness,
                                    "max_staleness_epochs": settings.REFERRAL_FEED_MAX_STALENESS_EPOCHS,
                                }
                            ),
                        ),
                    )
                    return {}

            result: dict[str, float] = {}
            for hotkey, raw_ema in weights.items():
                try:
                    ema = float(Decimal(str(raw_ema)))
                except (InvalidOperation, ValueError, TypeError):
                    logger.warning(
                        _m(
                            "[get_weights] Dropping hotkey with unparsable ema",
                            extra=get_extra_info({"hotkey": hotkey, "raw_ema": raw_ema}),
                        ),
                    )
                    continue
                if ema <= 0:
                    continue
                result[hotkey] = ema

            return result
        except Exception as e:
            logger.warning(
                _m(
                    "[get_weights] Failed to fetch referral feed, failing closed",
                    extra=get_extra_info({"url": url, "error": str(e)}),
                ),
            )
            return {}
