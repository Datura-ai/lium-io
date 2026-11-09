"""Idle-pay budget (DAH-6030): total idle pay at most k times realized Default Job revenue.

Every idle payee (Secure bucket pay, Spot pay, Secure floor top-ups) is scaled by one factor
s = min(1, k * revenue / uncapped idle pay), so the share it withholds stays in the burn share like any
other unpaid idle pay. The revenue is the backend's trailing-72h realized figure; the last good reading
is kept in Redis so an unreadable cycle or a restart keeps the budget instead of dropping it.
"""

import json
import time

from core.utils import _m, get_logger

logger = get_logger(__name__)

DJ_REVENUE_KEY = "idle_pay_budget:default_job_revenue_usd_per_hour"
# The k the per-cycle log prices while the guard is off, so the effect is visible before go-live.
SHADOW_K = 0.9


async def store_default_job_revenue(redis_service, usd_per_hour: float | None) -> None:
    """Keep the backend's reading as the last good one; None (unreadable) leaves the previous one in place."""
    if usd_per_hour is None:
        return
    try:
        await redis_service.set(DJ_REVENUE_KEY, json.dumps({"usd_per_hour": usd_per_hour, "at": time.time()}))
    except Exception as exc:
        logger.warning(_m("Idle-pay budget: storing Default Job revenue failed", extra={"error": str(exc)}))


async def load_default_job_revenue(redis_service) -> tuple[float | None, float | None]:
    """(last good USD/hour, its age in hours), or (None, None) when there has never been one (cold start)."""
    try:
        raw = await redis_service.get(DJ_REVENUE_KEY)
        if raw is None:
            return None, None
        stored = json.loads(raw)
        return float(stored["usd_per_hour"]), (time.time() - float(stored["at"])) / 3600
    except Exception as exc:
        logger.warning(_m("Idle-pay budget: Default Job revenue unreadable", extra={"error": str(exc)}))
        return None, None


def idle_pay_scale(k: float | None, revenue_usd_per_hour: float | None, uncapped_usd_per_hour: float) -> float:
    """s in [0, 1]. 1 when the guard is off (k unset), when there is no revenue reading (cold start), or when
    nothing is paid; 0 when revenue is truly zero."""
    if k is None or revenue_usd_per_hour is None or uncapped_usd_per_hour <= 0:
        return 1.0
    return min(1.0, max(0.0, k * revenue_usd_per_hour) / uncapped_usd_per_hour)
