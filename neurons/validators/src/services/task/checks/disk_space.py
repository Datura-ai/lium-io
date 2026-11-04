from __future__ import annotations

import asyncio
import math
from typing import Any

from services.const import MAX_DISK_UTILIZATION_PERCENT

from core.config import settings

from ..messages import DiskSpaceMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node

KB_PER_GB = 1024 * 1024


async def _any_listed_rental_active(ctx: Context, pods: list) -> bool:
    # a leftover running container is not a rental: only the backend's record counts (unknown = live)
    for pod in pods:
        rental_active = await ctx.services.backend.get_pod_rental_active(pod.pod_id)
        if not rental_active or rental_active.active:
            return True
    return False


async def _rental_active(ctx: Context, pods: list, limit_s: float | None) -> bool:
    try:
        return await asyncio.wait_for(_any_listed_rental_active(ctx, pods), timeout=limit_s)
    except TimeoutError:
        return True


def _as_float(value: Any) -> float:
    # a null or unparsable figure is not a reading (NaN), so it fails as NOT_REPORTED
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


class DiskSpaceCheck:
    """DAH-4961: the disk floor the renter listing used to hide nodes for, as a validation failure.

    Pure data on ``specs.hard_disk`` (free in KB, utilization in percent) as VerifyX leaves it, so it
    sits right after VerifyXCheck: the published figure is the one judged. Only idle nodes get here
    (TenantEnforcementCheck halts a rented one); a live rental is never failed for its disk.
    """

    check_id = "executor.validate.disk_space"
    fatal = True

    async def run(self, ctx: Context, rental_lookup_timeout: float | None = None) -> CheckResult:
        """`rental_lookup_timeout` caps the backend's per-pod rental lookup; a slower answer counts as
        a live rental, like an unknown one (VerifyXColdRetryCheck, which runs this near the deadline)."""
        hard_disk = (ctx.state.specs or {}).get("hard_disk")
        what: dict[str, Any] = {
            "max_utilization_percent": MAX_DISK_UTILIZATION_PERCENT,
            "required_free_gb": settings.MIN_FREE_DISK_GB,
        }
        utilization = free_kb = math.nan
        if isinstance(hard_disk, dict) and "utilization" in hard_disk and "free" in hard_disk:
            utilization = _as_float(hard_disk.get("utilization"))
            free_kb = _as_float(hard_disk.get("free"))
        # a missing, null, unparsable, NaN or Infinity figure is not a reading
        if not (math.isfinite(utilization) and math.isfinite(free_kb)):
            template = Msg.NOT_REPORTED
        else:
            # compare unrounded KB like the listing did; round only the displayed figure
            what.update(utilization_percent=utilization, free_gb=round(free_kb / KB_PER_GB, 1))
            if utilization > MAX_DISK_UTILIZATION_PERCENT:
                template = Msg.TOO_FULL
            elif free_kb < settings.MIN_FREE_DISK_GB * KB_PER_GB:
                template = Msg.FREE_TOO_LOW
            else:
                return CheckResult(
                    passed=True,
                    event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what),
                )
        # DAH-5462: a cold retry may replace these readings; VerifyXColdRetryCheck judges the sample it keeps
        if ctx.state.verifyx_cold_retry is not None:
            event = render_message(
                template,
                ctx=ctx,
                check_id=self.check_id,
                what={**what, "judged_after_cold_retry": True},
                impact="Proceed; judged again on the VerifyX sample the cold retry keeps",
            )
            return CheckResult(passed=True, event=event)
        # same ownership rule as TenantEnforcementCheck: another miner's rental on a different
        # endpoint does not exempt this node
        rental = rental_of_this_node(ctx)
        # TenantEnforcementCheck clears ctx.rented at the first ended pod, so a split node can still
        # hold a live rental; ask the backend for each listed pod
        if (
            rental
            and rental.pods
            and (ctx.rented or await _rental_active(ctx, rental.pods, rental_lookup_timeout))
        ):
            event = render_message(
                template,
                ctx=ctx,
                check_id=self.check_id,
                what=what,
                impact="Proceed; a live rental is not failed",
            )
            return CheckResult(passed=True, event=event)
        return CheckResult(
            passed=False, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )
