"""POST_RENTAL_GPU_RECHECK (DAH-4482): a node whose GPUs looked broken during a rental proves every card
before it re-lists.

A rented node's cycle halts at TenantEnforcementCheck before any GPU work, and a card that breaks under a
rental can keep passing the single-card matmul once the renter leaves. GpuMemoryHealthCheck, the one NVML
read that runs on rented nodes (DAH-3110), records a per-executor marker when it flags a card under a live
rental; on the node's next unrented cycle this check runs the all-cards matmul and the kernel-fault probe
on every card, whatever the GPU_FAULT_PROBE_* flags say. A failure fails the node and keeps the marker; a pass
clears it. The marker has no expiry (DAH-6355): only a pass ends it.

A card that falls off the bus (GpuModelValidCheck's GPU_DETAILS_MISMATCH) also sets the marker.

A cycle with a filler removes the node's Lium fillers and probes at once: the probes would compete with a filler
for VRAM, and waiting for a cycle without one only delays the passing signal that clears the marker.

GpuLossHoldCheck sets the marker too (source `gpu_loss_hold`) on the first clean read after a GPU dropout. For that
marker only a pass lets the node list: a failure or a cycle with no verdict (a filler that could not be removed) fails the node with a
GPU_LOST availability entry (WAITING_AFTER_GPU_LOSS: the backend hides the node, which a failed cycle alone does not
do while the row is still active) and keeps the marker, where a post-rental recheck lets the node proceed on no verdict.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from services.const import DEFAULT_JOB_OWNER_LIUM
from services.redis_service import POST_RENTAL_GPU_RECHECK_PREFIX

from core.config import settings
from core.utils import _m, get_extra_info

from ..availability import ReachSource, ReachTarget
from ..messages import GpuFaultProbeMessages, render_message
from ..messages import PostRentalGpuRecheckMessages as Msg
from ..pipeline import CheckResult, Context
from .capability import get_filler_only_container, judge_gpu_model, record_all_cards_pass
from .gpu_fault_probe import GpuFaultProbeCheck
from .gpu_model_truth import (
    expected_cc_and_sm,
    gpu_model_mismatches,
    judged_cards,
    reported_gpu_model,
)
from .rental_owner import rental_of_this_node
from .sshd_bootstrap import remove_lium_fillers

if TYPE_CHECKING:
    from services.matrix_validation_service import AllCardsProbe

logger = logging.getLogger(__name__)

MAX_MARKED_CARDS = 16
# GpuLossHoldCheck's marker: a node back from a GPU dropout lists only on a pass, never on no verdict
SOURCE_GPU_LOSS_HOLD = "gpu_loss_hold"


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


async def record_marker(ctx: Context, marker: RecheckMarker) -> None:
    """Remember what a rented cycle flagged. Never raises: the caller's verdict must not depend on Redis."""
    if settings.DRY_RUN:
        return
    payload = {**asdict(marker), "at": datetime.now(UTC).isoformat()}
    try:
        await ctx.services.redis.set(
            marker_key(ctx.miner_hotkey, ctx.executor.uuid),
            # DAH-6355: no TTL; only a pass ends it, however long the rental or the node's absence
            json.dumps(payload),
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
    # Only a lost connection to the whole node (`AllCardsProbe.connection_lost`) and the wall clock with every
    # card answering say nothing about the cards, as CapabilityCheck judges them. A timeout, a timeout again on the
    # probe's retry (`timeout_after_retry`) or a run that ended without an exit status on a live connection is a verdict and fails the
    # recheck (an 8x node's GPU 0 timed out every recheck 10-16 Nov and re-listed each time).
    if not probe.per_card:
        # not probed (no card details or size, our own cipher failed): no card was run, inconclusive as before
        return True
    if all(c.get("ok") for c in probe.per_card):
        return probe.over_wall_clock
    return probe.connection_lost


class PostRentalGpuRecheckCheck:
    check_id = "gpu.validate.post_rental_recheck"
    # a replay would advance this cycle's counters a second time
    replays_after_session_drop = False
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        key = marker_key(ctx.miner_hotkey, ctx.executor.uuid)
        marker = await self._read_marker(ctx, key)
        if marker is None:
            return CheckResult(
                passed=True, event=render_message(Msg.NO_MARKER, ctx=ctx, check_id=self.check_id)
            )
        what: dict[str, Any] = {"executor_uuid": ctx.executor.uuid, "marker": asdict(marker)}

        rental = rental_of_this_node(ctx)
        if rental is not None and rental.pods:
            # the cycle snapshot lists a rental: its workload may still load the cards though it ended mid-cycle,
            # so the cards prove themselves in the first idle cycle (as CapabilityCheck does)
            what["skipped"] = "rented_at_snapshot"
            await self._refresh_marker(ctx, key, what)
            self._log(ctx, "POST_RENTAL_GPU_RECHECK_INCONCLUSIVE", what, logging.INFO)
            return self._not_run(ctx, marker, Msg.INCONCLUSIVE, what)

        if get_filler_only_container(ctx):
            # both probes would compete with the filler for VRAM
            what["removed_fillers"] = await self._remove_fillers(ctx)
            if not what["removed_fillers"]:
                # the filler still holds VRAM: a probe OOM here would fail a healthy card
                await self._refresh_marker(ctx, key, what)
                self._log(ctx, "POST_RENTAL_GPU_RECHECK_INCONCLUSIVE", what, logging.INFO)
                return self._not_run(ctx, marker, Msg.INCONCLUSIVE, what)

        probes = await self._run_every_card_probes(ctx, what)
        if probes.failures:
            what["error"] = "; ".join(probes.failures)
            await self._refresh_marker(ctx, key, what)
            self._log(ctx, "POST_RENTAL_GPU_RECHECK_FAILED", what, logging.WARNING)
            return CheckResult(
                passed=False,
                event=render_message(Msg.FAILED, ctx=ctx, check_id=self.check_id, what=what),
                extra_events=[self._unlisted(ctx, what, Msg.FAILED)] if marker.source == SOURCE_GPU_LOSS_HOLD else [],
            )
        if (
            probes.fault_probe.event.reason_code != GpuFaultProbeMessages.PROBE_OK.reason
            or probes.all_cards_inconclusive
        ):
            # a probe that could not run (or found a filler), or no all-cards verdict: not a pass, so
            # the marker stays
            await self._refresh_marker(ctx, key, what)
            self._log(ctx, "POST_RENTAL_GPU_RECHECK_INCONCLUSIVE", what, logging.INFO)
            return self._not_run(ctx, marker, Msg.INCONCLUSIVE, what)
        await self._clear_marker(ctx, key, what)
        self._log(ctx, "POST_RENTAL_GPU_RECHECK_PASSED", what, logging.INFO)
        return CheckResult(
            passed=True,
            event=render_message(Msg.PASSED, ctx=ctx, check_id=self.check_id, what=what),
        )

    def _not_run(self, ctx: Context, marker: RecheckMarker, msg, what: dict[str, Any]) -> CheckResult:
        """No pass this cycle (no verdict): the marker stays. A post-rental recheck lets the node
        proceed; a node back from a GPU dropout stays unlisted until the test has passed on every card."""
        if marker.source != SOURCE_GPU_LOSS_HOLD:
            return CheckResult(passed=True, event=render_message(msg, ctx=ctx, check_id=self.check_id, what=what))
        return CheckResult(passed=False, event=self._unlisted(ctx, what, msg))

    def _unlisted(self, ctx: Context, what: dict[str, Any], msg):
        """The GPU_LOST availability entry that keeps a node back from a dropout off the market."""
        return render_message(
            Msg.WAITING_AFTER_GPU_LOSS,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "reach_source": str(ReachSource.VALIDATOR),
                "reach_target": str(ReachTarget.GPU),
                "executor_uuid": ctx.executor.uuid,
                "load_test": msg.reason,
            },
        )

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
        except Exception as exc:  # noqa: BLE001 — the pass stands; the recheck repeats next cycle
            what["marker_clear_error"] = str(exc)

    @staticmethod
    async def _refresh_marker(ctx: Context, key: str, what: dict[str, Any]) -> None:
        # keep the marker until a pass clears it, or the node re-lists unchecked; drops a TTL an older validator set
        if settings.DRY_RUN:
            return
        try:
            await ctx.services.redis.persist(key)
        except Exception as exc:  # noqa: BLE001 — the cycle's verdict stands either way
            what["marker_refresh_error"] = str(exc)

    @staticmethod
    async def _run_every_card_probes(ctx: Context, what: dict[str, Any]) -> CardProbes:
        """The all-cards matmul and the kernel-fault probe, whatever their own flags say."""
        failures: list[str] = []
        all_cards_probe = await ctx.services.validation.probe_all_claimed_cards(
            ctx.ssh, ctx.executor, ctx.default_extra, ctx.state.specs or {}
        )
        # None: one card (the capability matmul covers it); an unsized multi-card claim is a failed probe
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
        if all_cards_probe is not None and all_cards_probe.passed:
            await record_all_cards_pass(ctx)
        if all_cards_probe is not None and not all_cards_probe.passed:
            if _all_cards_no_verdict(all_cards_probe):
                inconclusive = True
            else:
                failures.append(f"all-cards matmul: {all_cards_probe.failure_reason}")
        if all_cards_probe is not None:
            # DAH-6241: each card's sealed numbers against the model reported for it
            card_metrics = {c["card_index"]: c["metrics"] for c in all_cards_probe.per_card if c.get("metrics")}
            specs = ctx.state.specs or {}
            measured = judged_cards(None, card_metrics)
            # a card that did the work but gave no numbers is only logged: no prod data yet for this path
            no_numbers = {
                c["card_index"]
                for c in all_cards_probe.per_card
                if c.get("ok")
                and c["card_index"] not in measured
                and expected_cc_and_sm(reported_gpu_model(specs, c["card_index"])) is not None
            }
            if no_numbers:
                what["gpu_model_no_numbers_cards"] = sorted(no_numbers)
            mismatches = await judge_gpu_model(ctx, gpu_model_mismatches(specs, None, card_metrics), measured)
            if mismatches:
                what["gpu_model_mismatch"] = mismatches[0].as_dict()
                failures.append(f"gpu model: card {mismatches[0].card_index} is below {mismatches[0].reported_gpu_model}")
        probe = await GpuFaultProbeCheck(is_post_rental_recheck=True).run(ctx)
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
