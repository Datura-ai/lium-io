from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from typing import Any

from services.const import MAX_DISK_UTILIZATION_PERCENT

from core.config import settings

from ..messages import DiskSpaceMessages as Msg
from ..messages import render_message, shown
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


def split_disk(ctx: Context, pods: list, free_kb: float) -> dict[str, Any] | None:
    """Disk verdict for the free splits of a partly rented split node, or None when it does not apply.

    The free GPUs are listed only while the host keeps the whole-host free-disk floor, the same floor an idle node
    is held to. A per-split share would not hold: nothing reserves it, and the rented pods' writable layers grow
    into the free disk up to their storage caps.
    """
    gpu_count = ctx.state.gpu_count
    min_count = ctx.state.gpu_splitting_min_count
    rented_gpus = [pod.gpu_count for pod in pods]
    if not (ctx.state.supports_gpu_splitting and gpu_count and min_count and rented_gpus):
        return None
    # a pod that does not report its GPUs leaves the free count unknown
    if any(not count or count <= 0 for count in rented_gpus):
        return None
    free_gpu_count = gpu_count - sum(rented_gpus)
    free_split_count = free_gpu_count // min_count
    if free_split_count <= 0:
        return None
    fits = free_kb >= settings.MIN_FREE_DISK_GB * KB_PER_GB
    return {
        "free_split_count": free_split_count,
        "listable_split_count": free_split_count if fits else 0,
        # scoring pays this count
        "listable_gpu_count": free_gpu_count if fits else 0,
    }


def specs_after_disk_taken(ctx: Context, taken_bytes: int) -> dict[str, Any] | None:
    """The cycle's specs with `taken_bytes` more of the host disk used since the scrape (DAH-6073: the cleanup pass's
    net change, a grow taking and a removal releasing; negative when more was released), and the split sizing redone
    on that figure, so the listing, the paid GPU count and the disk floor all judge the disk as it is after the pass.
    None when nothing changed or there is no disk reading."""
    specs = ctx.state.specs or {}
    hard_disk = specs.get("hard_disk")
    free_kb = _as_float(hard_disk.get("free")) if isinstance(hard_disk, dict) else math.nan
    if taken_bytes == 0 or not math.isfinite(free_kb):
        return None
    taken_kb = taken_bytes / 1024
    used_kb = _as_float(hard_disk.get("used"))
    updated = {**hard_disk, "free": max(free_kb - taken_kb, 0)}
    if math.isfinite(used_kb) and used_kb + free_kb > 0:
        used_after = min(max(used_kb + taken_kb, 0), used_kb + free_kb)
        updated["free"] = used_kb + free_kb - used_after
        updated["used"] = used_after
        updated["utilization"] = used_after / (used_kb + free_kb) * 100
    rental = rental_of_this_node(ctx)
    per_split = split_disk(ctx, rental.pods, updated["free"]) if rental and rental.pods else None
    return {**specs, "hard_disk": updated, **({"split_disk": per_split} if per_split is not None else {})}


class SplitDiskCheck:
    """DAH-4961: writes ``specs.split_disk`` on a partly rented split node from the scraped disk figure.

    Pure data and never fails. It sits before TenantEnforcementCheck, which halts a rented node's cycle before
    DiskSpaceCheck runs, so the sizing is published for every partly rented node. No rental, an unsplit node or no
    disk reading writes nothing.
    """

    check_id = "executor.validate.split_disk"
    replays_after_session_drop = True
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        rental = rental_of_this_node(ctx)
        hard_disk = (ctx.state.specs or {}).get("hard_disk")
        free_kb = _as_float(hard_disk.get("free")) if isinstance(hard_disk, dict) else math.nan
        per_split = (
            split_disk(ctx, rental.pods, free_kb) if rental and rental.pods and math.isfinite(free_kb) else None
        )
        event = render_message(Msg.SPLIT_DISK, ctx=ctx, check_id=self.check_id, what={"split_disk": per_split})
        if per_split is None:
            return CheckResult(passed=True, event=event)
        state = replace(ctx.state, specs={**ctx.state.specs, "split_disk": per_split})
        return CheckResult(passed=True, event=event, updates={"state": state})


class DiskSpaceCheck:
    """DAH-4961: the disk floor the renter listing used to hide nodes for, as a validation failure.

    Pure data on ``specs.hard_disk`` (free in KB, utilization in percent) as VerifyX leaves it, so it
    sits right after VerifyXCheck: the published figure is the one judged. Only idle nodes get here
    (TenantEnforcementCheck halts a rented one); a live rental is never failed for its disk.
    """

    check_id = "executor.validate.disk_space"
    replays_after_session_drop = True
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
            measured = ""
        else:
            # compare unrounded KB like the listing did; round only the displayed figure
            what.update(utilization_percent=utilization, free_gb=round(free_kb / KB_PER_GB, 1))
            if utilization > MAX_DISK_UTILIZATION_PERCENT:
                template = Msg.TOO_FULL
                measured = f"Disk {shown(utilization, 0, failed_below=False)}% full, at most {MAX_DISK_UTILIZATION_PERCENT}% allowed. "
            elif free_kb < settings.MIN_FREE_DISK_GB * KB_PER_GB:
                template = Msg.FREE_TOO_LOW
                measured = f"{shown(free_kb / KB_PER_GB, 1, failed_below=True)} GB free, {settings.MIN_FREE_DISK_GB} GB required. "
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
            # the free splits are listed only while the host keeps the whole-host floor
            per_split = split_disk(ctx, rental.pods, free_kb) if math.isfinite(free_kb) else None
            if per_split is None:
                return CheckResult(
                    passed=True,
                    event=render_message(
                        template,
                        ctx=ctx,
                        check_id=self.check_id,
                        what=what,
                        impact="Proceed; a live rental is not failed",
                    ),
                )
            listable = per_split["listable_split_count"]
            event = render_message(
                template,
                ctx=ctx,
                check_id=self.check_id,
                what={**what, "split_disk": per_split},
                impact=f"Proceed; a live rental is not failed; {listable} of "
                f"{per_split['free_split_count']} free splits listed",
            )
            state = replace(ctx.state, specs={**ctx.state.specs, "split_disk": per_split})
            return CheckResult(passed=True, event=event, updates={"state": state})
        event = render_message(
            template, ctx=ctx, check_id=self.check_id, what=what, remediation=measured + template.remediation
        )
        return CheckResult(passed=False, event=event)
