"""Fail-closed client for the backend's epoch-stable referral-weights feed (DAH-2481).

The backend serves ``<COMPUTE_REST_API_URL>/v1/referral-weights`` shaped like::

    {"epoch_index": <int>, "as_of_block": <int|null>,
     "weights": {"<hotkey_ss58>": "<ema_decimal_string>", ...}}

This client never raises: any transport error, non-2xx response, malformed body, or
stale epoch collapses to an empty result, so a feed problem simply means no referral
emission that cycle rather than blocking or corrupting weight-setting.
"""
from decimal import Decimal, InvalidOperation

import requests

from core.config import settings
from core.utils import _m, get_extra_info, get_logger

logger = get_logger(__name__)


class ReferralFeedClient:
    """Fetches referrer hotkey -> EMA weight from the backend referral feed."""

    def get_weights(self, current_epoch: int | None = None) -> dict[str, float]:
        """Fetch and parse the referral-weights feed.

        Returns ``{hotkey_ss58: ema_float}`` on success, or ``{}`` on any failure
        (transport error, bad status, malformed JSON, missing/empty weights, or a
        staleness violation against ``current_epoch``). Entries whose EMA value is
        ``<= 0`` or fails to parse are dropped rather than failing the whole fetch.
        """
        try:
            response = requests.get(settings.REFERRAL_FEED_URL, timeout=10)
            response.raise_for_status()
            body = response.json()

            weights = body.get("weights")
            if not weights:
                logger.warning(
                    _m(
                        "[get_weights] Referral feed returned no weights",
                        extra=get_extra_info({"url": settings.REFERRAL_FEED_URL}),
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
                    extra=get_extra_info({"url": settings.REFERRAL_FEED_URL, "error": str(e)}),
                ),
            )
            return {}
