"""POST_RENTAL_GPU_RECHECK (DAH-4482): a node whose GPUs looked broken during a rental proves every card
before it re-lists.

A rented node's cycle halts at TenantEnforcementCheck before any GPU work, and a card that breaks under a
rental can keep passing the single-card matmul once the renter leaves. GpuMemoryHealthCheck, the one NVML
read that runs on rented nodes (DAH-3110), records a per-executor marker when it flags a card under a live
rental; on the node's next unrented cycle this check runs the all-cards matmul and the kernel-fault probe
on every card, whatever the MATMUL_ALLCARDS_* / GPU_FAULT_PROBE_* flags say. A failure fails the node and keeps the marker; a pass
clears it.

A card that falls off the bus (GpuModelValidCheck's GPU_DETAILS_MISMATCH) also sets the marker, and so
does the backend's renter_hardware_fault_executor_ids (hardware-fault reports from two distinct renters):
a GeForce card has no ECC or row-remap counters, so a renter's report is the only signal it gives.
A pass sets a cleared key for MARKER_TTL_SECONDS: the backend list cannot tell a new report from the
old one, so new renter reports on that node go unseen until the cleared key expires.

A cycle with a filler skips the recheck; after FILLER_SKIP_LIMIT such skips the check removes the
node's Lium fillers and probes anyway, so a node that always runs a filler cannot skip it for good.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from services.const import DEFAULT_JOB_OWNER_LIUM
from services.redis_service import POST_RENTAL_GPU_RECHECK_PREFIX

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import GpuFaultProbeMessages, render_message
from ..messages import PostRentalGpuRecheckMessages as Msg
from ..pipeline import CheckResult, Context
from .capability import get_filler_only_container
from .gpu_fault_probe import GpuFaultProbeCheck
from .sshd_bootstrap import remove_lium_fillers

if TYPE_CHECKING:
    from services.matrix_validation_service import AllCardsProbe

logger = logging.getLogger(__name__)

# a few days: long enough to outlast a weekend rental's tail, short enough that a stale flag dies
MARKER_TTL_SECONDS = 3 * 24 * 3600
MAX_MARKED_CARDS = 16
# filler cycles the recheck waits out before it removes the fillers and probes anyway
FILLER_SKIP_LIMIT = 3


@dataclass(frozen=True)
class CardProbes:
    failures: list[str]
    # no all-cards verdict: skipped on a multi-card node, or lost to SSH, a timeout or the wall clock
    all_cards_inconclusive: bool
    fault_probe: CheckResult


@dataclass(frozen=True)
class RecheckMarker:
    source: str
    verdict: str = ""
    flags: list[Any] = field(default_factory=list)
    cards: list[dict[str, str]] = field(default_factory=list)
    filler_skips: int = 0
    at: str = ""  # when the marker was last written; record_marker sets it

    @classmethod
    def parse(cls, raw: Any) -> RecheckMarker:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            data = None
        if not isinstance(data, dict):
            return cls(source="unreadable", verdict=str(raw)[:200])
        known = {f.name for f in fields(cls)}
        return cls(**{"source": "unknown", **{k: v for k, v in data.items() if k in known}})


def marker_key(miner_hotkey: str, executor_uuid: str) -> str:
    return f"{POST_RENTAL_GPU_RECHECK_PREFIX}:{miner_hotkey}:{executor_uuid}"


def cleared_key(miner_hotkey: str, executor_uuid: str) -> str:
    return f"{POST_RENTAL_GPU_RECHECK_PREFIX}_cleared:{miner_hotkey}:{executor_uuid}"


def _renter_reported(ctx: Context) -> bool:
    """Listed by the backend for renter hardware-fault reports, under this hotkey."""
    rented_data = ctx.state.rented_data
    uuid = str(ctx.executor.uuid)
    if not rented_data or uuid not in rented_data.renter_hardware_fault_executor_ids:
        return False
    rented = rented_data.executors.get(uuid)
    owner = rented_data.executor_owner_hotkeys.get(uuid) or (
        rented.miner_hotkey if rented else None
    )
    # an unknown owner applies nothing: a report must not land on a node listed under another hotkey
    return owner == ctx.miner_hotkey


async def record_marker(ctx: Context, marker: RecheckMarker) -> None:
    """Remember what a rented cycle flagged. Never raises: the caller's verdict must not depend on Redis."""
    if settings.DRY_RUN:
        return
    payload = {**asdict(marker), "at": datetime.now(UTC).isoformat()}
    try:
        await ctx.services.redis.set(
            marker_key(ctx.miner_hotkey, ctx.executor.uuid),
            json.dumps(payload),
            ex=MARKER_TTL_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            _m(
                "POST_RENTAL_GPU_RECHECK_MARKER_WRITE_FAILED",
                extra=get_extra_info({**ctx.default_extra, "error": str(exc)}),
            )
        )


async def record_gpu_drop(ctx: Context, missing_uuids: list[str]) -> None:
    """A card fell off the bus: owe a recheck."""
    await record_marker(
        ctx,
        RecheckMarker(
            source="gpu_details_mismatch",
            verdict="gpu_missing",
            cards=[{"uuid": u, "verdict": "missing"} for u in missing_uuids[:MAX_MARKED_CARDS]],
        ),
    )


def _all_cards_no_verdict(probe: AllCardsProbe) -> bool:
    # SSH errors, timeouts and the wall clock say nothing about the cards, as the fault probe treats them
    reasons = [c.get("reason") or "" for c in probe.per_card if not c.get("ok")]
    if not reasons:
        return probe.over_wall_clock
    return all(r == "timeout" or r.startswith("ssh_error") for r in reasons)


class PostRentalGpuRecheckCheck:
    check_id = "gpu.validate.post_rental_recheck"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        key = marker_key(ctx.miner_hotkey, ctx.executor.uuid)
        marker = await self._find_marker(ctx, key)
        if marker is None:
            return CheckResult(
                passed=True, event=render_message(Msg.NO_MARKER, ctx=ctx, check_id=self.check_id)
            )
        what: dict[str, Any] = {"executor_uuid": ctx.executor.uuid, "marker": asdict(marker)}

        if get_filler_only_container(ctx):
            if marker.filler_skips < FILLER_SKIP_LIMIT:
                # both probes would compete with the filler for VRAM; the marker waits for an idle cycle
                await record_marker(ctx, replace(marker, filler_skips=marker.filler_skips + 1))
                return CheckResult(
                    passed=True,
                    event=render_message(
                        Msg.FILLER_SKIPPED, ctx=ctx, check_id=self.check_id, what=what
                    ),
                )
            what["removed_fillers"] = await self._remove_fillers(ctx)
            if not what["removed_fillers"]:
                # the filler still holds VRAM: a probe OOM here would fail a healthy card
                await self._refresh_marker(ctx, key, what)
                self._log(ctx, "POST_RENTAL_GPU_RECHECK_INCONCLUSIVE", what, logging.INFO)
                return CheckResult(
                    passed=True,
                    event=render_message(Msg.INCONCLUSIVE, ctx=ctx, check_id=self.check_id, what=what),
                )

        probes = await self._run_every_card_probes(ctx, what)
        if probes.failures:
            what["error"] = "; ".join(probes.failures)
            await self._refresh_marker(ctx, key, what)
            self._log(ctx, "POST_RENTAL_GPU_RECHECK_FAILED", what, logging.WARNING)
            return CheckResult(
                passed=False,
                event=render_message(Msg.FAILED, ctx=ctx, check_id=self.check_id, what=what),
            )
        if (
            probes.fault_probe.event.reason_code != GpuFaultProbeMessages.PROBE_OK.reason
            or probes.all_cards_inconclusive
        ):
            # a probe that could not run (or found a filler), or no all-cards verdict: not a pass, so
            # the marker stays
            await self._refresh_marker(ctx, key, what)
            self._log(ctx, "POST_RENTAL_GPU_RECHECK_INCONCLUSIVE", what, logging.INFO)
            return CheckResult(
                passed=True,
                event=render_message(Msg.INCONCLUSIVE, ctx=ctx, check_id=self.check_id, what=what),
            )
        await self._clear_marker(ctx, key, what)
        self._log(ctx, "POST_RENTAL_GPU_RECHECK_PASSED", what, logging.INFO)
        return CheckResult(
            passed=True,
            event=render_message(Msg.PASSED, ctx=ctx, check_id=self.check_id, what=what),
        )

    @staticmethod
    async def _recently_cleared(ctx: Context) -> bool:
        try:
            return (
                await ctx.services.redis.get(cleared_key(ctx.miner_hotkey, ctx.executor.uuid))
                is not None
            )
        except Exception:  # noqa: BLE001 — unreadable: do not re-mark a node that may have just passed
            return True

    async def _find_marker(self, ctx: Context, key: str) -> RecheckMarker | None:
        """The stored marker, else a new one for a renter report not already cleared by a pass."""
        marker = await self._read_marker(ctx, key)
        if marker is None and _renter_reported(ctx) and not await self._recently_cleared(ctx):
            marker = await self._record_renter_report_marker(ctx)
        return marker

    @staticmethod
    async def _remove_fillers(ctx: Context) -> list[str]:
        rented_data = ctx.state.rented_data
        if (
            settings.DRY_RUN
            or rented_data.get_default_job_owner(ctx.executor.uuid) != DEFAULT_JOB_OWNER_LIUM
        ):
            return []
        return await remove_lium_fillers(ctx, rented_data.get_filler_containers(ctx.executor.uuid))

    @staticmethod
    async def _record_renter_report_marker(ctx: Context) -> RecheckMarker:
        marker = RecheckMarker(source="renter_report", verdict="renter_reported")
        await record_marker(ctx, marker)
        # DRY_RUN writes nothing: the recheck still sees what it would require this cycle
        return marker

    @staticmethod
    async def _read_marker(ctx: Context, key: str) -> RecheckMarker | None:
        try:
            raw = await ctx.services.redis.get(key)
        except Exception as exc:  # noqa: BLE001 — no marker readable is no recheck, never a failure
            logger.warning(
                _m(
                    "POST_RENTAL_GPU_RECHECK_MARKER_READ_FAILED",
                    extra=get_extra_info({**ctx.default_extra, "error": str(exc)}),
                )
            )
            return None
        return None if raw is None else RecheckMarker.parse(raw)

    @staticmethod
    async def _clear_marker(ctx: Context, key: str, what: dict[str, Any]) -> None:
        if settings.DRY_RUN:
            return
        try:
            await ctx.services.redis.delete(key)
            # the backend keeps listing a node for its whole report window: a pass holds off re-marking
            await ctx.services.redis.set(
                cleared_key(ctx.miner_hotkey, ctx.executor.uuid), "1", ex=MARKER_TTL_SECONDS
            )
        except Exception as exc:  # noqa: BLE001 — the pass stands; the recheck repeats next cycle
            what["marker_clear_error"] = str(exc)

    @staticmethod
    async def _refresh_marker(ctx: Context, key: str, what: dict[str, Any]) -> None:
        # keep the marker alive until a pass clears it, or the node re-lists unchecked
        if settings.DRY_RUN:
            return
        try:
            await ctx.services.redis.expire(key, MARKER_TTL_SECONDS)
        except Exception as exc:  # noqa: BLE001 — the cycle's verdict stands either way
            what["marker_refresh_error"] = str(exc)

    @staticmethod
    async def _run_every_card_probes(ctx: Context, what: dict[str, Any]) -> CardProbes:
        """The all-cards matmul and the kernel-fault probe, whatever their own flags say."""
        failures: list[str] = []
        all_cards_probe = await ctx.services.validation.probe_all_claimed_cards(
            ctx.ssh, ctx.executor, ctx.default_extra, ctx.state.specs or {}
        )
        # None: one card (the capability matmul covers it) or an unsized model (inconclusive)
        what["all_cards_matmul"] = (
            None
            if all_cards_probe is None
            else {
                "passed": all_cards_probe.passed,
                "failure_reason": all_cards_probe.failure_reason,
            }
        )
        gpu_count = ((ctx.state.specs or {}).get("gpu") or {}).get("count") or 0
        inconclusive = all_cards_probe is None and gpu_count > 1
        if all_cards_probe is not None and not all_cards_probe.passed:
            if _all_cards_no_verdict(all_cards_probe):
                inconclusive = True
            else:
                failures.append(f"all-cards matmul: {all_cards_probe.failure_reason}")
        probe = await GpuFaultProbeCheck(enforce=True).run(ctx)
        what["fault_probe"] = {
            "reason": probe.event.reason_code,
            "error": probe.event.what_we_saw.get("error"),
        }
        if not probe.passed:
            failures.append(f"fault probe: {probe.event.what_we_saw.get('error')}")
        return CardProbes(failures, inconclusive, probe)

    @staticmethod
    def _log(ctx: Context, key: str, what: dict[str, Any], level: int) -> None:
        logger.log(level, _m(key, extra=get_extra_info({**ctx.default_extra, **what})))
