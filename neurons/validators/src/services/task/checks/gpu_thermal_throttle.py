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
GPU_THERMAL_COLLAPSE) until the host fixes its cooling, so it is not listed to the next renter. A rented node
only gets the warning, so this check never touches a running rental. Every other finding (a thermal bit at a
near-full clock, a low clock with no slowdown bit) is the non-fatal GPU_THERMAL_THROTTLE warning, score
unchanged. It runs right after the scrape, before the rented short-circuit (TenantEnforcementCheck), so rented
nodes are judged too. One reading per cycle; the next clean cycle passes again.
"""

from __future__ import annotations

from typing import Any

from ..messages import GpuThermalThrottleMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

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


class GpuThermalThrottleCheck:
    """Warn about a busy GPU held below its clock by a thermal/hardware slowdown. Never fatal, never scores."""

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
        rented_executor = (
            ctx.state.rented_data.executors.get(ctx.executor.uuid)
            if ctx.state.rented_data
            else None
        )
        rented = bool(rented_executor and rented_executor.pods)
        collapsed = [finding for finding in throttled if finding["collapsed"]]
        if collapsed and not rented:
            event = render_message(
                Msg.COLLAPSED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "gpus": throttled,
                    "collapsed_count": len(collapsed),
                    "min_clock_ratio": MIN_CLOCK_RATIO,
                },
            )
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
