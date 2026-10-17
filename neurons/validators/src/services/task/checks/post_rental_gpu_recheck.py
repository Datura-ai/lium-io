"""POST_RENTAL_GPU_RECHECK (DAH-4482): a node whose GPUs looked broken during a rental proves every card
before it re-lists.

A rented node's cycle halts at TenantEnforcementCheck before any GPU work, and a GeForce card that breaks
under a rental keeps passing the single-card matmul once the renter leaves. RentedGpuHealthShadowCheck
records a per-executor marker whenever its read shows a fault or a flag; on the node's next unrented cycle
this check runs the all-cards matmul and the kernel-fault probe on every card, whatever the
MATMUL_ALLCARDS_* / GPU_FAULT_PROBE_* flags say. A failure fails the node and keeps the marker; a pass
clears it. Off by default (POST_RENTAL_GPU_RECHECK_ENABLED): the cycle then only logs
POST_RENTAL_GPU_RECHECK_WOULD_REQUIRE.

A card that falls off the bus (GpuModelValidCheck's GPU_DETAILS_MISMATCH) also sets the marker.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from services.redis_service import POST_RENTAL_GPU_RECHECK_PREFIX

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import GpuFaultProbeMessages, render_message
from ..messages import PostRentalGpuRecheckMessages as Msg
from ..pipeline import CheckResult, Context
from .capability import _get_filler_only_container
from .gpu_fault_probe import GpuFaultProbeCheck

if TYPE_CHECKING:
    from services.matrix_validation_service import AllCardsProbe

logger = logging.getLogger(__name__)

# a few days: long enough to outlast a weekend rental's tail, short enough that a stale flag dies
MARKER_TTL_SECONDS = 3 * 24 * 3600
MAX_MARKED_CARDS = 16


@dataclass(frozen=True)
class CardProbes:
    failures: list[str]
    all_cards_probe: AllCardsProbe | None
    fault_probe: CheckResult


def marker_key(miner_hotkey: str, executor_uuid: str) -> str:
    return f"{POST_RENTAL_GPU_RECHECK_PREFIX}:{miner_hotkey}:{executor_uuid}"


async def record_marker(ctx: Context, source: str, health: dict[str, Any]) -> None:
    """Remember what a rented cycle flagged. Never raises: the caller is a shadow check."""
    if settings.DRY_RUN:
        return
    cards = [
        {
            key: card.get(key)
            for key in ("index", "uuid", "verdict", "reasons", "flags")
            if card.get(key) is not None
        }
        for card in health.get("cards") or []
        if card.get("verdict") != "ok" or card.get("flags")
    ][:MAX_MARKED_CARDS]
    marker = {
        "at": datetime.now(UTC).isoformat(),
        "source": source,
        "verdict": health.get("verdict"),
        "cards": cards,
    }
    try:
        await ctx.services.redis.set(
            marker_key(ctx.miner_hotkey, ctx.executor.uuid),
            json.dumps(marker),
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
    """A card fell off the bus: owe a recheck. Never raises: the caller's verdict must not depend on Redis."""
    await record_marker(
        ctx,
        "gpu_details_mismatch",
        {
            "verdict": "gpu_missing",
            "cards": [{"uuid": u, "verdict": "missing"} for u in missing_uuids[:MAX_MARKED_CARDS]],
        },
    )


class PostRentalGpuRecheckCheck:
    check_id = "gpu.validate.post_rental_recheck"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        key = marker_key(ctx.miner_hotkey, ctx.executor.uuid)
        try:
            raw = await ctx.services.redis.get(key)
        except Exception as exc:  # noqa: BLE001 — no marker readable is no recheck, never a failure
            raw = None
            logger.warning(
                _m(
                    "POST_RENTAL_GPU_RECHECK_MARKER_READ_FAILED",
                    extra=get_extra_info({**ctx.default_extra, "error": str(exc)}),
                )
            )
        if raw is None:
            return CheckResult(
                passed=True, event=render_message(Msg.NO_MARKER, ctx=ctx, check_id=self.check_id)
            )
        try:
            marker = json.loads(raw)
        except (TypeError, ValueError):
            marker = {"raw": str(raw)[:200]}
        what: dict[str, Any] = {"executor_uuid": ctx.executor.uuid, "marker": marker}

        if not settings.POST_RENTAL_GPU_RECHECK_ENABLED:
            self._log(ctx, "POST_RENTAL_GPU_RECHECK_WOULD_REQUIRE", what, logging.WARNING)
            return CheckResult(
                passed=True,
                event=render_message(Msg.WOULD_REQUIRE, ctx=ctx, check_id=self.check_id, what=what),
            )

        if _get_filler_only_container(ctx):
            # both probes would compete with the filler for VRAM; the marker waits for an idle cycle
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.FILLER_SKIPPED, ctx=ctx, check_id=self.check_id, what=what
                ),
            )

        probes = await self._run_every_card_probes(ctx, what)
        if probes.failures:
            what["error"] = "; ".join(probes.failures)
            if not settings.DRY_RUN:
                try:
                    # keep the marker alive while the node keeps failing, or it re-lists unchecked
                    await ctx.services.redis.expire(key, MARKER_TTL_SECONDS)
                except Exception as exc:  # noqa: BLE001 — the failure stands either way
                    what["marker_refresh_error"] = str(exc)
            self._log(ctx, "POST_RENTAL_GPU_RECHECK_FAILED", what, logging.WARNING)
            return CheckResult(
                passed=False,
                event=render_message(Msg.FAILED, ctx=ctx, check_id=self.check_id, what=what),
            )
        gpu_count = ((ctx.state.specs or {}).get("gpu") or {}).get("count") or 0
        if probes.fault_probe.event.reason_code != GpuFaultProbeMessages.PROBE_OK.reason or (
            probes.all_cards_probe is None and gpu_count > 1
        ):
            # a probe that could not run (or found a filler), or a multi-card node the all-cards matmul
            # skipped: not a pass, so the marker stays
            self._log(ctx, "POST_RENTAL_GPU_RECHECK_INCONCLUSIVE", what, logging.INFO)
            return CheckResult(
                passed=True,
                event=render_message(Msg.INCONCLUSIVE, ctx=ctx, check_id=self.check_id, what=what),
            )
        if not settings.DRY_RUN:
            try:
                await ctx.services.redis.delete(key)
            except Exception as exc:  # noqa: BLE001 — the pass stands; the recheck repeats next cycle
                what["marker_clear_error"] = str(exc)
        self._log(ctx, "POST_RENTAL_GPU_RECHECK_PASSED", what, logging.INFO)
        return CheckResult(
            passed=True,
            event=render_message(Msg.PASSED, ctx=ctx, check_id=self.check_id, what=what),
        )

    @staticmethod
    async def _run_every_card_probes(ctx: Context, what: dict[str, Any]) -> CardProbes:
        """The all-cards matmul and the kernel-fault probe, whatever their own flags say."""
        failures: list[str] = []
        all_cards_probe = await ctx.services.validation._probe_all_claimed_cards(
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
        if all_cards_probe is not None and not all_cards_probe.passed:
            failures.append(f"all-cards matmul: {all_cards_probe.failure_reason}")
        probe = await GpuFaultProbeCheck(enforce=True).run(ctx)
        what["fault_probe"] = {
            "reason": probe.event.reason_code,
            "error": probe.event.what_we_saw.get("error"),
        }
        if not probe.passed:
            failures.append(f"fault probe: {probe.event.what_we_saw.get('error')}")
        return CardProbes(failures, all_cards_probe, probe)

    @staticmethod
    def _log(ctx: Context, key: str, what: dict[str, Any], level: int) -> None:
        logger.log(level, _m(key, extra=get_extra_info({**ctx.default_extra, **what})))
