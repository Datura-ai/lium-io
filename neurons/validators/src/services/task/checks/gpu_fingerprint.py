from __future__ import annotations

import logging

from core.config import settings
from services.redis_service import GPU_ANCHOR_KEY

from ..messages import GpuFingerprintMessages as Msg, render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# What a differing set that is not a full change is: only cards missing, or other cards among fewer or more.
# Written into the reset evidence as `anchor_hard_would_be` (the backend tells a missing card from the rest by it).
DECISION_GPU_MISSING = "GPU_MISSING"
DECISION_ANCHOR_BROKEN = "ANCHOR_BROKEN"


def full_set_changed(anchor: list[str], current: list[str]) -> bool:
    """The node lists as many distinct cards as it was anchored with, but not the same ones (a VM host can get
    other physical GPUs on a reboot). Not a fault, rented or idle: GpuFingerprintCheck re-anchors the node to the
    new set and verifies it the same cycle, and GpuLossHoldCheck holds nothing for it. Fewer cards is a dropout."""
    return bool(anchor) and len(current) == len(anchor) and set(current) != set(anchor)


def split_uuids(csv: str) -> list[str]:
    """The distinct UUIDs of a comma-joined list, sorted.

    A UUID listed twice is one card read twice (`gpu_uuids` is joined from `gpu_details[*].uuid` by the
    scrape), never a second card, so one copy counts.
    """
    return sorted({u for u in csv.split(",") if u})


class GpuFingerprintCheck:
    """Compare the anchored GPU UUID set with the latest scrape.

    The anchor is the `uuids` value of the executor's verified-job record: written on the first successful
    verification and kept by every later write, including the reset (`clear_verified_job_info` copies it),
    except on a full set change.

    A full set of other cards (`full_set_changed`: as many cards as anchored, other UUIDs) is not a fault in
    either mode: the check passes, `gpu_reanchor` makes the new set the anchor, and nothing is reset. The same
    GPU UUIDs listed by two nodes at once is judged by DuplicateExecutorCheck, not here.

    Any other difference (cards missing, or a card count that changed) is fatal with a verification reset and
    keeps the anchor: GPU_UUID_CHANGED, or under GPU_ANCHOR_HARD_ENABLED (DAH-3457) GPU_MISSING when only
    anchored cards are missing. The node passes again on the first cycle that shows the full set. This check runs
    before SpecChangeCheck (pipeline_factory), so every such change is filed here, and SpecChangeCheck only fires
    when the UUIDs match but the model:count string differs.

    Both sets are compared as distinct UUIDs. A scrape that lists one anchored card twice matches the anchor
    (`duplicate_uuid: true` on the event, one warning in the log). A scrape with zero cards passes this check
    and leaves the anchor unchanged. Neither is read as a set change.
    """

    check_id = "gpu.validate.fingerprint"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        verified = ctx.verified or {}
        prev_uuids = verified.get(GPU_ANCHOR_KEY) or ""
        current_uuids = ctx.state.gpu_uuids or ""
        hard = settings.GPU_ANCHOR_HARD_ENABLED

        anchor = split_uuids(prev_uuids)
        current = split_uuids(current_uuids)
        # A duplicated scrape is not a set change (split_uuids keeps one copy). It is logged and flagged on the
        # event so the rows show how often the scrape does this. A scrape with zero cards never reaches the
        # comparison. Neither one touches the stored anchor.
        listed = [u for u in current_uuids.split(",") if u]
        duplicated = len(listed) != len(current)
        if duplicated:
            logger.warning(
                "GPU UUID listed twice in the scrape of executor %s; one copy counts: %s",
                ctx.executor.uuid,
                current_uuids,
            )
        seen = {"duplicate_uuid": True} if duplicated else {}

        if full_set_changed(anchor, current):
            # not a fault (owner decision, 29 Oct): anchor the new set and verify it now, rented or idle
            logger.info("GPU set of executor %s changed in full; re-anchored: %s", ctx.executor.uuid, current_uuids)
            event = render_message(
                Msg.UUID_OK,
                ctx=ctx,
                check_id=self.check_id,
                what={"gpu_uuids": current_uuids, "previous": prev_uuids, "reanchored": True, **seen},
            )
            return CheckResult(passed=True, event=event, updates={"gpu_reanchor": True})

        if anchor and current and anchor != current:
            missing = sorted(set(anchor) - set(current))
            unexpected = sorted(set(current) - set(anchor))
            decision = DECISION_GPU_MISSING if not unexpected else DECISION_ANCHOR_BROKEN
            what = {
                "previous": prev_uuids,
                "current": current_uuids,
                "missing": missing,
                "unexpected": unexpected,
                **seen,
            }
            if hard and decision == DECISION_GPU_MISSING:
                event = render_message(Msg.GPU_MISSING, ctx=ctx, check_id=self.check_id, what=what)
            else:
                event = render_message(
                    Msg.UUID_CHANGED, ctx=ctx, check_id=self.check_id, what={**what, "anchor_hard_would_be": decision}
                )
            return CheckResult(
                passed=False,
                event=event,
                updates={
                    "clear_verified_job_info": True,
                    # tells a missing card (a renter's GPU reset can cause it) from a swap
                    "clear_verified_job_evidence": {"anchor_hard_would_be": decision},
                },
            )

        event = render_message(
            Msg.UUID_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_uuids": current_uuids, **seen},
        )
        return CheckResult(passed=True, event=event)
