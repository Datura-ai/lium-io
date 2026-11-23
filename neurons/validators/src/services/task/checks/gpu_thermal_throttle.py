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
  (a power cap set on purpose also holds the clock down; GpuPowerLimitCheck judges that). A power-cap bit
  with the power limit still at the card's default explains nothing: that is a host power fault.

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
the matmul, so on an idle node only a filler or the host's own load at scrape time clears it. DAH-6469: an unrented
node held with no new collapse and no filler gets that load from the validator, every such cycle: the kernel-fault
probe runs dense FMA work on every card (``--heat-seconds``), and after LOAD_WARMUP_SECONDS, with the load still
running, a fresh encrypted machine scrape reads the cards (``scrape_gpu_details``); its rows are judged like the
cycle's scrape (collapsed renews). The load is then killed and its exit awaited, within LOAD_KILL_SECONDS, so the
usage checks after this one see neither it nor its rows: ctx.state keeps the pre-load scrape. A card clears only on a
busy judgeable reading without a thermal slowdown bit; a load that ended before the reading came back proves
nothing, and a cycle with less time left than the whole sequence skips it. No reading leaves the hold, and the TTL
stays the fallback. A dry run writes nothing; a Redis that cannot be read is no hold.

A GPU whose clocks a renter locked (DAH-4663, a ``gpu_clock_lock:<uuid>`` record in Redis) is spared only the
low-clock finding: the lock holds its clock down on purpose, but a thermal or hardware slowdown is still judged.
The record is read only when a GPU was flagged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from datetime import UTC, datetime
from typing import Any

from services.gpu_clocks import clock_locked_gpu_uuids

from core.config import settings
from core.utils import _m, get_extra_info

from ..availability import ReachSource, ReachTarget
from ..messages import GpuThermalThrottleMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .capability import get_filler_only_container
from .gpu_fault_probe import PROBE_SECONDS, PROBE_TIMEOUT_SECONDS, probe_source
from .machine_spec_scrape import MachineSpecScrapeCheck, scrape_gpu_details
from .rental_owner import rental_of_this_node
from .rented_gpu_drop import GPU_DROP_REDIS_ERRORS
from .task_time import task_seconds_left

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
# A power limit at least this share of the card's default counts as the default (not lowered on purpose).
POWER_CAP_DEFAULT_RATIO = 0.98
THERMAL_HOLD_KEY_PREFIX = "gpu_thermal_hold"
THERMAL_HOLD_SECONDS = 7 * 24 * 3600
# DAH-6469: the validator's own load on a held node's cards: dense FMA work (the probe's --heat-seconds) after the
# fault rounds. Under real filler load the first collapsed scrape came 3 min after the start at the fastest, so the
# cards are read after that long under it; a card whose cooling still fails heats into its slowdown inside it.
LOAD_PROBE_SECONDS = 180
# the probe's fault rounds plus its workers' fork, JIT and allocation (seconds on a healthy node) before the heat
LOAD_START_SECONDS = PROBE_SECONDS + 30
LOAD_WARMUP_SECONDS = LOAD_START_SECONDS + LOAD_PROBE_SECONDS
# the remote kill and the wait for every load process to exit, then the same again for the SSH call to return
LOAD_KILL_SECONDS = 30
# the load's processes: the probe and its forked workers share the command line; the bracket keeps the pattern from
# matching the shell that runs pkill
LOAD_PROCESS_PATTERN = "-I - --seconds [0-9.]* --heat-second[s] "
LOAD_KILL_COMMAND = (
    f"pkill -9 -f -- {shlex.quote(LOAD_PROCESS_PATTERN)};"
    f" for _ in $(seq {LOAD_KILL_SECONDS}); do pgrep -f -- {shlex.quote(LOAD_PROCESS_PATTERN)} >/dev/null || exit 0;"
    " sleep 1; done; exit 1"
)
# a thermal slowdown bit on a busy reading: the card is still braking for heat, so the reading proves nothing
THERMAL_SLOWDOWN_REASONS = CLOCKS_EVENT_REASON_SW_THERMAL_SLOWDOWN | CLOCKS_EVENT_REASON_HW_THERMAL_SLOWDOWN


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
    # A power cap excuses a low clock only when the limit was set below the card's default (GpuPowerLimitCheck
    # judges that). At the default limit, a cap holding a busy card under half its clock is a host power fault.
    limit = _to_number(detail.get("power_limit"))
    default_limit = _to_number(detail.get("power_default_limit"))
    cap_at_default = (
        power_capped
        and limit is not None
        and bool(default_limit)
        and limit >= default_limit * POWER_CAP_DEFAULT_RATIO
    )
    low_clock = (
        clock_ratio is not None
        and clock_ratio < MIN_CLOCK_RATIO
        and (not power_capped or cap_at_default)
    )

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


def _clears_hold(detail: dict[str, Any]) -> bool:
    """A busy, judgeable reading with no thermal slowdown bit: only that proves a held card's cooling (DAH-6469).
    A ratio at or above MIN_CLOCK_RATIO with the bit set is a card still braking (H100, 7 Nov: 0.682 with the bit,
    collapsed again an hour later at 0.439)."""
    return _judgeable_busy(detail) and not int(_to_number(detail.get("clocks_event_reasons")) or 0) & THERMAL_SLOWDOWN_REASONS


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


def _scrape_timeout(ctx: Context) -> int:
    return ctx.config.machine_scrape_timeout or MachineSpecScrapeCheck.DEFAULT_TIMEOUT


def _load_probe_seconds(ctx: Context) -> int:
    """The whole load sequence: the warm-up, the scrape under the load, then the kill and its exit."""
    return LOAD_WARMUP_SECONDS + _scrape_timeout(ctx) + 2 * LOAD_KILL_SECONDS + 10


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
        if throttled:
            # DAH-4663: a renter's own clock lock holds the clock down on purpose: its low clock alone is not the
            # host's fault, a slowdown bit still is (a record a failed -rgc left behind hides no heat fault)
            locked = await clock_locked_gpu_uuids(ctx.services.redis, [str(f["uuid"]) for f in throttled])
            throttled = [
                finding
                for finding in throttled
                if finding["uuid"] not in locked or finding["reasons"] != ["clock_below_half_of_max"]
            ]
        rented_executor = rental_of_this_node(ctx)
        rented = bool(rented_executor and rented_executor.pods)
        collapsed = [finding for finding in throttled if finding["collapsed"]]
        held, since = await self._update_hold(ctx, collapsed)
        load_probe: dict[str, Any] | None = None
        if (
            held
            and not collapsed
            and not rented
            and not get_filler_only_container(ctx)
            # the load needs its whole sequence; a cycle without it leaves the hold to the next one (or the TTL)
            and task_seconds_left(ctx) >= _load_probe_seconds(ctx)
        ):
            readings = await self._load_probe(ctx)
            probe_findings = [finding for finding in map(judge_gpu, readings) if finding is not None]
            collapsed = [finding for finding in probe_findings if finding["collapsed"]]
            throttled = throttled + probe_findings
            if readings:
                held, since = await self._update_hold(ctx, collapsed, readings)
            load_probe = {"seconds": LOAD_PROBE_SECONDS, "readings": len(readings), "still_held": held}
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
            if load_probe is not None:
                what["load_probe"] = load_probe
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

    async def _load_probe(self, ctx: Context) -> list[dict[str, Any]]:
        """The cards' rows from an encrypted scrape taken under the validator's load; [] when there is no such reading.

        The load is killed, and its exit awaited, before this returns, whatever happened."""
        scrape_timeout = _scrape_timeout(ctx)
        # the load outlasts the warm-up and the whole scrape; it is killed as soon as the reading is back
        heat_seconds = LOAD_PROBE_SECONDS + (LOAD_START_SECONDS - PROBE_SECONDS) + scrape_timeout + LOAD_KILL_SECONDS
        load = asyncio.create_task(
            ctx.runner.run(
                f"{shlex.quote(ctx.executor.python_path)} -I - --seconds {PROBE_SECONDS} --heat-seconds {heat_seconds}",
                timeout=PROBE_TIMEOUT_SECONDS + heat_seconds,
                retryable=False,
                stdin_text=probe_source(),
            )
        )
        try:
            await asyncio.sleep(LOAD_WARMUP_SECONDS)
            if load.done():
                return []
            readings = await scrape_gpu_details(ctx, scrape_timeout)
            # a load that ended (a fault, an error, a killed probe) before the reading came back may not have been on
            # the cards when they were read
            return [] if load.done() else readings
        finally:
            await asyncio.shield(self._stop_load(ctx, load))

    async def _stop_load(self, ctx: Context, load: asyncio.Task) -> None:
        if not load.done():
            kill = await ctx.runner.run(LOAD_KILL_COMMAND, timeout=LOAD_KILL_SECONDS + 10, retryable=False)
            if not kill.success:
                logger.warning(
                    _m(
                        "GPU_THERMAL_LOAD_KILL_FAILED",
                        extra=get_extra_info(
                            {**ctx.default_extra, "exit_code": kill.exit_code, "stderr": kill.stderr[-200:]}
                        ),
                    )
                )
        try:
            await asyncio.wait_for(load, LOAD_KILL_SECONDS)
        except asyncio.TimeoutError:
            pass  # wait_for cancelled the SSH call; the remote processes are already gone or logged above

    async def _update_hold(
        self,
        ctx: Context,
        collapsed: list[dict[str, Any]],
        readings: list[dict[str, Any]] | None = None,
    ) -> tuple[list[str], str | None]:
        """The marked GPUs still unproven after this reading, and since when; keeps the Redis mark in step.

        readings: the rows of the scrape under the validator's load instead of the cycle's scrape (DAH-6469)."""
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
        rows = ctx.state.gpu_details if readings is None else readings
        details = [detail for detail in rows or [] if isinstance(detail, dict)]
        # only a busy reading judge_gpu can judge proves a card; one missing a clock or the reasons is not clean
        proven = {detail.get("uuid") for detail in details if _clears_hold(detail)}
        present = [detail.get("uuid") for detail in details]
        # a marked card the scrape no longer lists was replaced (a lost card is GpuLossHoldCheck's);
        # a row without a uuid could be any marked card, so then none counts as replaced
        # the rows under load only clear a card by its own clean reading: replacement is the cycle's scrape's call
        if readings is None and present and all(isinstance(uuid, str) and uuid for uuid in present):
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
