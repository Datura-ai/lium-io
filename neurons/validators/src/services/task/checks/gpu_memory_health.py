from __future__ import annotations

import json
import logging
import re
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any

import redis.exceptions

from services import executor_payload
from core.config import settings
from core.utils import _m, get_extra_info

from ..availability import ReachSource, ReachTarget
from ..messages import GpuMemoryHealthMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context, MemoryHealthRead
from .banned_provider import has_live_rental
from .gpu_fault_probe import OUTPUT_TAIL_CHARS, _cap, _parse_report, probe_source
from .post_rental_gpu_recheck import RecheckMarker, record_marker

logger = logging.getLogger(__name__)

# one NVML snapshot in a fork with its own 10 s deadline (miner_jobs/gpu_fault_probe.py), plus SSH
TIMEOUT_SECONDS = 30
# a flag names its card as "gpu <index> (<uuid>): ..." (miner_jobs/gpu_fault_probe.py nvml_health)
_FLAG_UUID = re.compile(r"^gpu \S+ \(([^()]+)\): ")
# A recovery action from correctable errors alone is a warning once, but a card whose NVML still
# asks for a reset this many cycles in a row is not being reset, and a renter on it sees faults
# (NVLink crashes on a B300 under GPU_RESET, 27 Oct): from here it is judged like a flagged card.
# shortcut: counted per validator process, so a restart starts the count over; persist if it bites
RESET_REQUIRED_STREAK_LIMIT = 3
_reset_required_streak: dict[tuple[str, str], int] = {}

# DAH-5869: per executor, the kernel time of the last Xid line already judged, the host's boot id and one held Xid fault per
# card: it closes only on proof it healed
CARD_HEALTH_KEY_PREFIX = "gpu_card_health"
CARD_HEALTH_TTL_SECONDS = 86400
_REDIS_ERRORS: tuple[type[BaseException], ...] = (redis.exceptions.RedisError, OSError)
# a state Redis failed to store: the next read judges against it, so a confirmed fault is not lost
# shortcut: kept in this process only, so a validator restart during a Redis outage drops it; persist if it bites
_unsaved_card_health: dict[str, "CardHealthState"] = {}


@dataclass
class CardHealthState:
    xid_after: float | None = None
    boot_id: str | None = None
    held: list[str] = field(default_factory=list)
    # per card UUID, the NVML counters only a GPU reset (or a driver reload) clears that the last read saw set;
    # None: stored before these marks existed, so no read has a baseline yet
    reset_marks: dict[str, list[str]] | None = None


def _boot_id(value: Any) -> str | None:
    """A boot id, or None when it is missing or blank: an empty id is no boot, so it never proves a reboot."""
    return (value.strip() or None) if isinstance(value, str) else None


def _card_health_key(ctx: Context) -> str:
    return f"{CARD_HEALTH_KEY_PREFIX}:{ctx.miner_hotkey}:{ctx.executor.uuid}"


async def _load_card_health(ctx: Context) -> CardHealthState | None:
    """The stored state, or None when Redis could not be read: then nothing is judged or written, so a held
    fault is neither closed nor overwritten."""
    if (unsaved := _unsaved_card_health.get(_card_health_key(ctx))) is not None:
        return unsaved
    try:
        stored = json.loads(await ctx.services.redis.get(_card_health_key(ctx)) or "{}")
    except (*_REDIS_ERRORS, TypeError, ValueError):
        return None
    if not isinstance(stored, dict):
        return None
    # a baseline stored as a line count (before the kernel-time baseline) is dropped: the next read sets one
    xid_after = stored.get("xid_after")

    def strings(key: str) -> list[str]:
        value = stored.get(key)
        return [s for s in value if isinstance(s, str)] if isinstance(value, list) else []

    marks = stored.get("reset_marks")
    return CardHealthState(
        xid_after=xid_after if type(xid_after) in (int, float) else None,
        boot_id=_boot_id(stored.get("boot_id")),
        held=strings("held"),
        reset_marks={
            uuid: [c for c in counters if c in RESET_CLEARED_COUNTERS]
            for uuid, counters in marks.items()
            if isinstance(uuid, str) and isinstance(counters, list)
        }
        if isinstance(marks, dict)
        else None,
    )


# volatile uncorrected ECC and the recovery action: a card reset or a driver reload is the only thing that zeroes them
RESET_CLEARED_COUNTERS = ("ecc_uncorrected", "recovery_action")


def _reset_marks(gpu: dict[str, Any], kept: list[str], conclusive: bool) -> list[str]:
    """The reset-only counters this read saw set. A counter it could not read, or any counter in a read that could
    not see the kernel log (no reset is proven there), keeps the mark the last read left (`kept`): an inconclusive
    read never erases the baseline a later reset is proven against."""
    return [
        c
        for c in RESET_CLEARED_COUNTERS
        if (type(gpu.get(c)) is int and gpu[c] != 0) or ((gpu.get(c) is None or not conclusive) and c in kept)
    ]


# Xids that leave a reset-only counter set (48 DBE, 95 uncontained ECC) or the card unlisted (79, off the bus) until
# the card is reset; any other hardware Xid can leave both counters at 0, so a zeroed read proves nothing for it
RESET_PROVEN_XIDS = {48, 79, 95}
_HELD_XID = re.compile(r"Xid \([^)]*\): (\d+)")


def _card_was_reset(
    fault: str, uuid: str, gpus: dict[str, dict[str, Any]], every_card_named: bool, before: CardHealthState
) -> bool:
    """Every reset-only counter the last read saw set on the card now reads 0: the card was reset since. A held card
    with no marks (a hold stored before the marks existed, or an Xid logged after the last read whose counters the
    reset zeroed before any read saw them) has no such read: there, a read naming every card with both counters
    at 0 proves it only for an Xid in RESET_PROVEN_XIDS."""
    if uuid not in gpus:
        return False
    if marked := (before.reset_marks or {}).get(uuid):
        return all(gpus[uuid].get(c) == 0 for c in marked)
    xid = _HELD_XID.search(fault)
    return (
        every_card_named
        and xid is not None
        and int(xid.group(1)) in RESET_PROVEN_XIDS
        and all(type(gpus[uuid].get(c)) is int and gpus[uuid][c] == 0 for c in RESET_CLEARED_COUNTERS)
    )


def _card_uuid(fault: str) -> str | None:
    match = _FLAG_UUID.match(fault)
    return match.group(1) if match and match.group(1) != "None" else None


def _held_fault_proof(
    fault: str,
    gpus: dict[str, dict[str, Any]],
    every_card_named: bool,
    rebooted: bool,
    before: CardHealthState,
    log_seen: bool,
) -> str | None:
    """What proves a held card fault over: "reboot" (a new host boot) or "reset" (`_card_was_reset`, only in a
    read that also sees the kernel log, so a card logging Xids this read cannot see is not lifted), else None.
    A card missing from the read, or with its counters unread, is no proof: a card that drops out for a cycle
    comes back as faulty as it left (0d0c64ef, 18 Nov)."""
    uuid = _card_uuid(fault)
    if uuid is None:
        return None
    if rebooted:
        return "reboot"
    if log_seen and _card_was_reset(fault, uuid, gpus, every_card_named, before):
        return "reset"
    return None


async def _judge_card_faults(ctx: Context, report: dict[str, Any], before: CardHealthState) -> None:
    """Put this cycle's per-card hardware faults into `report["card_faults"]`: the probe's new Xids and
    every earlier fault not yet proven healed
    (`_held_fault_proof`); then keep the state for the next read. A dry run shares Redis with
    the live validator, so it writes nothing."""
    if report.get("status") not in ("ok", "flagged"):
        # no NVML reading: the state stays as it was, and must not expire while NVML stays unreadable
        if not settings.DRY_RUN:
            try:
                await ctx.services.redis.expire(_card_health_key(ctx), CARD_HEALTH_TTL_SECONDS)
            except _REDIS_ERRORS:
                pass
        return
    xid_faults = [s for s in report.get("xid_faults") or [] if isinstance(s, str)]
    xid = report.get("xid") if isinstance(report.get("xid"), dict) else {}
    boot_id = _boot_id(report.get("boot_id"))
    # the boot id, not the Xid lines: a cleared or wrapped kernel log drops lines on a faulty card
    rebooted = before.boot_id is not None and boot_id is not None and boot_id != before.boot_id
    listed = [gpu for gpu in (report.get("nvml") or {}).get("gpus") or [] if isinstance(gpu, dict)]
    gpus = {gpu["uuid"]: gpu for gpu in listed if isinstance(gpu.get("uuid"), str)}
    every_card_named = bool(listed) and len(gpus) == len(listed)
    # a card read without its UUID or bus id had its new Xid lines skipped: keep them for the next read;
    # with no baseline yet no line was judged, so the last line's time is the baseline
    every_card_tied = every_card_named and all(isinstance(gpu.get("pci_bus_id"), str) for gpu in listed)
    # after a reboot the kernel clock starts over, so the old baseline no longer orders the lines
    last_ts = xid.get("last_ts") if type(xid.get("last_ts")) in (int, float) else None
    uses_this_read_as_xid_baseline = xid.get("available") and (
        every_card_tied or before.xid_after is None or rebooted
    )
    if uses_this_read_as_xid_baseline:
        xid_after = last_ts
    else:
        # an old boot's time would skip this boot's early lines: start the new boot from its first line
        xid_after = 0.0 if rebooted else before.xid_after
    log_seen = bool(xid.get("available"))
    proofs = {f: _held_fault_proof(f, gpus, every_card_named, rebooted, before, log_seen) for f in before.held}
    held = [f for f, proof in proofs.items() if proof is None]
    # one fault per card, this read's first: a card repeating an Xid must not grow the stored state
    fault_by_card: dict[str, str] = {}
    for fault in xid_faults + held:
        uuid = _card_uuid(fault)
        if uuid is not None:
            fault_by_card.setdefault(uuid, fault)
    report["card_faults"] = list(fault_by_card.values())
    if before.held or xid_faults:
        what = {
            "executor_uuid": ctx.executor.uuid,
            "decisions": {_card_uuid(f): proof or "held" for f, proof in proofs.items()},
            "new_xid_faults": xid_faults,
            "card_faults": report["card_faults"],
            "boot_id_before": before.boot_id,
            "boot_id_now": boot_id,
            "rebooted": rebooted,
            "xid_after_before": before.xid_after,
            "xid_after_now": xid_after,
            "cards_listed": sorted(gpus),
            "cards_read": len(listed),
            "every_card_named": every_card_named,
            "reset_marks_before": before.reset_marks,
        }
        logger.info(_m("GPU card hold judged", extra=get_extra_info({**ctx.default_extra, **what})))
    if settings.DRY_RUN:
        return
    key = _card_health_key(ctx)
    # a card this read did not name keeps its marks: its next read still proves a reset
    reset_marks = {u: m for u, m in (before.reset_marks or {}).items() if u not in gpus and not rebooted}
    kept = {} if rebooted else before.reset_marks or {}
    reset_marks.update({u: m for u, gpu in gpus.items() if (m := _reset_marks(gpu, kept.get(u, []), log_seen))})
    state = CardHealthState(
        xid_after=xid_after, boot_id=boot_id or before.boot_id, held=report["card_faults"], reset_marks=reset_marks
    )
    try:
        await ctx.services.redis.set(key, json.dumps(asdict(state)), ex=CARD_HEALTH_TTL_SECONDS)
    except _REDIS_ERRORS:
        _unsaved_card_health[key] = state
    else:
        _unsaved_card_health.pop(key, None)


async def read_memory_health(ctx: Context) -> MemoryHealthRead:
    """The NVML-only probe's run and report, read once per cycle: RentedGpuDropCheck reads it first
    on a rented node (and keeps it in `ctx.state.memory_health`), and GpuMemoryHealthCheck reuses it."""
    if ctx.state.memory_health is not None:
        return ctx.state.memory_health
    before = await _load_card_health(ctx)
    command = f"{shlex.quote(executor_payload.EXECUTOR_PYTHON)} -I - --nvml-only"
    if before is not None and before.xid_after is not None:
        command += f" --xid-after {before.xid_after}"
        if before.boot_id is not None:
            command += f" --xid-boot {shlex.quote(before.boot_id)}"
    run = await ctx.runner.run(
        command, timeout=TIMEOUT_SECONDS, retryable=False, stdin_text=probe_source()
    )
    report = _parse_report(run.stdout) or {}
    if before is not None:
        await _judge_card_faults(ctx, report, before)
    return MemoryHealthRead(run=run, report=report)


def flagged_uuids(report: dict[str, Any]) -> list[str]:
    """The UUIDs of the cards a flagged report names broken."""
    if report.get("status") != "flagged" or not isinstance(report.get("flags"), list):
        return []
    found = (_FLAG_UUID.match(flag) for flag in report["flags"] if isinstance(flag, str))
    return sorted({match.group(1) for match in found if match and match.group(1) != "None"})


def card_fault_uuids(report: dict[str, Any]) -> list[str] | None:
    """The UUIDs of the cards with a hardware fault this cycle (`report["card_faults"]`): a new hardware
    Xid, held until proven healed. None when the card state was not judged (unread)."""
    if "card_faults" not in report:
        return None
    return sorted({u for f in report["card_faults"] if isinstance(f, str) and (u := _card_uuid(f))})


class GpuMemoryHealthCheck:
    """Flag a card whose memory NVML reports broken (DAH-3110): uncorrected ECC errors since the
    driver loaded, a failed row remap, or a pending remap / required recovery action on a card with
    uncorrectable remapped rows or uncorrected ECC (what `nvidia-smi -q -d ECC,ROW_REMAPPER` shows);
    a pending remap or recovery action from correctable errors alone is only a warning, sent as one
    so the provider resets the card, until a recovery action is still required
    RESET_REQUIRED_STREAK_LIMIT cycles in a row: then the card is judged like a flagged one. The
    fault probe sees these only as changes across its own run, and only on an idle node; this is
    one NVML read and no CUDA, so it runs before the rented short-circuit, for rented and idle
    executors alike.

    A flagged card fails the node while GPU_MEMORY_HEALTH_ENFORCEMENT_ENABLED (on by default) like
    a GPU ban: verified job info is cleared, so it is not rented again until the host resets or
    replaces the card, and the failure is fatal only when no renter's pod is running (a live rental
    keeps being verified and billed). DAH-4683: the failure is also a GPU_MEMORY_HEALTH_FAILED
    availability entry, so the backend hides the node and refuses new rentals while leaving a rental
    already on it alone: a split-rented node with one live pod otherwise kept its broken free card
    listed. A node NVML could not read passes. A card flagged under a live rental
    also owes a post-rental every-card recheck (post_rental_gpu_recheck.py).
    """

    # a replay would count this cycle's reset warning twice toward RESET_REQUIRED_STREAK_LIMIT
    replays_after_session_drop = False

    check_id = "gpu.validate.memory_health"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        read = await read_memory_health(ctx)
        run, report = read.run, read.report
        what: dict[str, Any] = {"executor_uuid": ctx.executor.uuid}
        status = report.get("status")
        warnings = _cap(report.get("warnings")) if status == "ok" else None
        warnings = warnings if isinstance(warnings, list) else []
        reset_required = [w for w in warnings if " NVML recovery action " in str(w)]
        key = (ctx.miner_hotkey, ctx.executor.uuid)
        streak = 0
        if reset_required:
            streak = _reset_required_streak[key] = _reset_required_streak.get(key, 0) + 1
        elif status in ("ok", "flagged"):
            # a clean or flagged read ends the run; an unread cycle leaves it as it is
            _reset_required_streak.pop(key, None)
        if status == "flagged" or streak >= RESET_REQUIRED_STREAK_LIMIT:
            flags = _cap(report.get("flags")) if status == "flagged" else reset_required
            if status != "flagged":
                what["reset_required_streak"] = streak
            what["flags"] = flags if isinstance(flags, list) else []
            if has_live_rental(ctx):
                # DAH-4482: the node owes an every-card recheck on its next unrented cycle
                await record_marker(
                    ctx, RecheckMarker(self.check_id, "flagged", flags=what["flags"])
                )
            if settings.GPU_MEMORY_HEALTH_ENFORCEMENT_ENABLED:
                what["reach_source"] = str(ReachSource.VALIDATOR)
                what["reach_target"] = str(ReachTarget.GPU)
                event = render_message(Msg.FAILED, ctx=ctx, check_id=self.check_id, what=what)
                return CheckResult(
                    passed=False,
                    event=event,
                    updates={"clear_verified_job_info": True},
                    fatal=not has_live_rental(ctx),
                )
            message = Msg.FLAGGED
        elif status == "ok":
            # a pending remap from correctable errors only: the card still works, but needs a reset
            if warnings:
                what["warnings"] = warnings
                message = Msg.RESET_ADVISED
            else:
                message = Msg.OK
        else:
            # no NVML binding, NVML that would not start, SSH: not a reading of the cards
            nvml = report.get("nvml") if isinstance(report.get("nvml"), dict) else {}
            what["error"] = _cap(nvml.get("error")) or run.error_message or "no report in output"
            what["stderr_tail"] = run.stderr[-OUTPUT_TAIL_CHARS:]
            message = Msg.UNKNOWN
        event = render_message(message, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event)
