"""GPU_THERMAL_THROTTLE: a busy GPU held far below its clock by a thermal or hardware slowdown.

A card whose cooling has failed keeps its utilisation near 100% while NVML drops its graphics clock to a
fraction of the maximum (support ticket-0492: an H100 at 345-650 MHz instead of ~1980 MHz for two days,
86-87 C with SW thermal slowdown set, ~230 W against a 700 W limit). The matmul tolerance is loose on purpose
and the other GPU checks read power, count and VRAM, so nothing else sees it.

Pure data: reads, per GPU, what MachineSpecScrapeCheck left in ``ctx.state.gpu_details`` (``temperature``,
``graphics_speed``, ``max_graphics_speed``, ``clocks_event_reasons``, ``gpu_utilization``). A GPU is flagged
when it is at least ``MIN_UTILIZATION_PCT`` busy and either

- a thermal or hardware slowdown bit of the clocks-event reasons is set, or
- its graphics clock is below ``MIN_CLOCK_RATIO`` of its max clock while no SW power-cap bit explains it
  (a power cap set on purpose also holds the clock down; GpuPowerLimitCheck judges that).

An idle card is not judged: an idle card downclocks by design. A scrape that predates these readings, or a
driver that lacks the calls, leaves the fields missing and the GPU is not judged either.

A GPU whose clock is below ``MIN_CLOCK_RATIO`` of its max WITH a thermal or hardware slowdown bit set has
collapsed (ticket-0492: 345 MHz of 1980 at 95 C): on a node that is not rented the check fails (fatal, score 0,
GPU_THERMAL_COLLAPSE, an availability entry that hides it at once) until the host fixes its cooling, so it is
not listed to the next renter. A rented node
only gets the warning, so this check never touches a running rental. Every other finding (a thermal bit at a
near-full clock, a low clock with no slowdown bit) is the non-fatal GPU_THERMAL_THROTTLE warning, score
unchanged. It runs right after the scrape, before the rented short-circuit (TenantEnforcementCheck), so rented
nodes are judged too.

A collapse is remembered, so the node cannot pass on an idle card (28 Oct, 8x H200 c63f453b: collapsed at 345 MHz
on every rented cycle, read idle and scored 1.0 the cycle after the renter left). Every collapse, rented or not,
writes the collapsed UUIDs to Redis (``gpu_thermal_hold:<hotkey>:<executor uuid>``, THERMAL_HOLD_SECONDS from the
last collapse). An unrented cycle with marked GPUs still unproven fails as a collapse does (held_from_rental). A
marked GPU is cleared by a busy reading that is not collapsed and carries the clocks and reasons judge_gpu needs, or
when a scrape whose every row has a UUID no longer lists it (replaced); the reading is the scrape's, taken before
the matmul, so on an idle node only a filler or the host's own load at scrape time clears it, otherwise the TTL does. A dry run
writes nothing; a Redis that cannot be read is no hold.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from core.config import settings
from core.utils import _m, get_extra_info

from ..availability import ReachSource, ReachTarget
from ..messages import GpuThermalThrottleMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node
from .rented_gpu_drop import GPU_DROP_REDIS_ERRORS

logger = logging.getLogger(__name__)

# nvmlClocksEventReasons bits (nvml.h; the same values as the older nvmlClocksThrottleReasons).
CLOCKS_EVENT_REASON_SW_POWER_CAP = 0x4
CLOCKS_EVENT_REASON_HW_SLOWDOWN = 0x8
CLOCKS_EVENT_REASON_SW_THERMAL_SLOWDOWN = 0x20
CLOCKS_EVENT_REASON_HW_THERMAL_SLOWDOWN = 0x40

THERMAL_REASON_NAMES = {
    CLOCKS_EVENT_REASON_HW_SLOWDOWN: "hw_slowdown",
    CLOCKS_EVENT_REASON_SW_THERMAL_SLOWDOWN: "sw_thermal_slowdown",
    CLOCKS_EVENT_REASON_HW_THERMAL_SLOWDOWN: "hw_thermal_slowdown",
}

# Only a busy card is judged: an idle one drops its clock on purpose.
MIN_UTILIZATION_PCT = 80
# Below half the max clock under load is a collapse, not boost variation (ticket-0492: ~17-33%).
MIN_CLOCK_RATIO = 0.5
THERMAL_HOLD_KEY_PREFIX = "gpu_thermal_hold"
THERMAL_HOLD_SECONDS = 7 * 24 * 3600


def _to_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def judge_gpu(detail: dict[str, Any]) -> dict[str, Any] | None:
    """The throttle finding for one gpu_details row, or None when the GPU is fine or cannot be judged."""
    utilization = _to_number(detail.get("gpu_utilization"))
    if utilization is None or utilization < MIN_UTILIZATION_PCT:
        return None

    reasons_value = _to_number(detail.get("clocks_event_reasons"))
    reasons = int(reasons_value) if reasons_value is not None else None
    clock = _to_number(detail.get("graphics_speed"))
    max_clock = _to_number(detail.get("max_graphics_speed"))

    thermal_reasons = (
        [name for bit, name in THERMAL_REASON_NAMES.items() if reasons & bit]
        if reasons is not None
        else []
    )
    clock_ratio = clock / max_clock if clock is not None and max_clock else None
    power_capped = reasons is not None and bool(reasons & CLOCKS_EVENT_REASON_SW_POWER_CAP)
    low_clock = clock_ratio is not None and clock_ratio < MIN_CLOCK_RATIO and not power_capped

    if not thermal_reasons and not low_clock:
        return None
    return {
        "uuid": detail.get("uuid"),
        "name": detail.get("name"),
        "utilization_pct": utilization,
        "temperature_c": _to_number(detail.get("temperature")),
        "graphics_clock_mhz": clock,
        "max_graphics_clock_mhz": max_clock,
        "clock_ratio": round(clock_ratio, 3) if clock_ratio is not None else None,
        "clocks_event_reasons": hex(reasons) if reasons is not None else None,
        "reasons": thermal_reasons + (["clock_below_half_of_max"] if low_clock else []),
        # Hard evidence of a heat fault: half the clock gone AND the driver says why.
        "collapsed": bool(thermal_reasons)
        and clock_ratio is not None
        and clock_ratio < MIN_CLOCK_RATIO,
    }


def _judgeable_busy(detail: dict[str, Any]) -> bool:
    """A busy reading with every field judge_gpu needs to see a collapse."""
    utilization = _to_number(detail.get("gpu_utilization"))
    return (
        utilization is not None
        and utilization >= MIN_UTILIZATION_PCT
        and _to_number(detail.get("graphics_speed")) is not None
        and bool(_to_number(detail.get("max_graphics_speed")))
        and _to_number(detail.get("clocks_event_reasons")) is not None
    )


class GpuThermalThrottleCheck:
    """Warn about a busy GPU held below its clock by a thermal/hardware slowdown; fail an unrented node on a collapse."""

    check_id = "gpu.validate.thermal_throttle"
    fatal = True  # only the GPU_THERMAL_COLLAPSE result fails; every other result passes

    async def run(self, ctx: Context) -> CheckResult:
        throttled = [
            finding
            for finding in (
                judge_gpu(detail)
                for detail in ctx.state.gpu_details or []
                if isinstance(detail, dict)
            )
            if finding is not None
        ]
        rented_executor = rental_of_this_node(ctx)
        rented = bool(rented_executor and rented_executor.pods)
        collapsed = [finding for finding in throttled if finding["collapsed"]]
        held, since = await self._update_hold(ctx, collapsed)
        if (collapsed or held) and not rented:
            what: dict[str, Any] = {
                # an availability entry, as GpuLossHoldCheck's: the backend hides the node this cycle
                "reach_source": str(ReachSource.VALIDATOR),
                "reach_target": str(ReachTarget.GPU),
                "executor_uuid": ctx.executor.uuid,
                "gpus": throttled,
                "collapsed_count": len(collapsed),
                "min_clock_ratio": MIN_CLOCK_RATIO,
            }
            if not collapsed:
                what.update(held_from_rental=True, held_gpu_uuids=held, held_since=since)
            event = render_message(Msg.COLLAPSED, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(
                passed=False,
                event=event,
                updates={
                    "score": 0.0,
                    "job_score": 0.0,
                    "score_warning": " WARNING: a busy GPU is thermally throttled below half its clock",
                },
            )
        if throttled:
            event = render_message(
                Msg.THROTTLED,
                ctx=ctx,
                check_id=self.check_id,
                what={"gpus": throttled, "throttled_count": len(throttled)},
            )
            # Non-fatal and passed: the warning event is the signal; nothing downstream reads a verdict.
            return CheckResult(passed=True, event=event)

        event = render_message(
            Msg.OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_count": len(ctx.state.gpu_details or [])},
        )
        return CheckResult(passed=True, event=event)

    async def _update_hold(
        self, ctx: Context, collapsed: list[dict[str, Any]]
    ) -> tuple[list[str], str | None]:
        """The marked GPUs still unproven after this reading, and since when; keeps the Redis mark in step."""
        key = f"{THERMAL_HOLD_KEY_PREFIX}:{ctx.miner_hotkey}:{ctx.executor.uuid}"
        mark: dict[str, Any] = {}
        try:
            raw = await ctx.services.redis.get(key)
            mark = json.loads(raw) if raw else {}
            if not isinstance(mark, dict):
                mark = {}
        except (*GPU_DROP_REDIS_ERRORS, TypeError, ValueError):
            self._log_redis_unavailable(ctx, "read")
        marked = {uuid for uuid in mark.get("gpus") or [] if isinstance(uuid, str)}
        collapsed_uuids = {f["uuid"] for f in collapsed if isinstance(f["uuid"], str)}
        details = [detail for detail in ctx.state.gpu_details or [] if isinstance(detail, dict)]
        # only a busy reading judge_gpu can judge proves a card; one missing a clock or the reasons is not clean
        proven = {detail.get("uuid") for detail in details if _judgeable_busy(detail)}
        present = [detail.get("uuid") for detail in details]
        # a marked card the scrape no longer lists was replaced (a lost card is GpuLossHoldCheck's);
        # a row without a uuid could be any marked card, so then none counts as replaced
        if present and all(isinstance(uuid, str) and uuid for uuid in present):
            marked_now = marked & set(present)
        else:
            marked_now = marked
        held = (marked_now - (proven - collapsed_uuids)) | collapsed_uuids
        since = mark.get("since") if marked else datetime.now(UTC).isoformat()
        # A dry run shares Redis with the live validator, so it leaves the live hold alone.
        if settings.DRY_RUN or (held == marked and not collapsed_uuids):
            return sorted(held), since
        try:
            if held:
                # every collapse renews the TTL
                mark = {"gpus": sorted(held), "since": since}
                await ctx.services.redis.set(key, json.dumps(mark), ex=THERMAL_HOLD_SECONDS)
            else:
                await ctx.services.redis.delete(key)
        except GPU_DROP_REDIS_ERRORS:
            self._log_redis_unavailable(ctx, "write")
        return sorted(held), since

    @staticmethod
    def _log_redis_unavailable(ctx: Context, on: str) -> None:
        logger.warning(
            _m(
                "GPU_THERMAL_HOLD_REDIS_UNAVAILABLE",
                extra=get_extra_info({**ctx.default_extra, "on": on}),
            ),
            exc_info=True,
        )
