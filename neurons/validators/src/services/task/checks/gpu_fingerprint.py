from __future__ import annotations

from core.config import settings
from services.redis_service import GPU_ANCHOR_BROKEN_KEY, GPU_ANCHOR_KEY

from ..messages import GpuFingerprintMessages as Msg, render_message
from ..pipeline import CheckResult, Context

# The two decisions the hard rule takes on a set that differs from the anchor; written into the legacy
# GPU_UUID_CHANGED event as `anchor_hard_would_be` while GPU_ANCHOR_HARD_ENABLED is off.
DECISION_GPU_MISSING = "GPU_MISSING"
DECISION_ANCHOR_BROKEN = "ANCHOR_BROKEN"


def split_uuids(csv: str) -> list[str]:
    return sorted(u for u in csv.split(",") if u)


class GpuFingerprintCheck:
    """Compare the anchored GPU UUID set with the latest scrape.

    The anchor is the `uuids` value of the executor's verified-job record: written on the first successful
    verification and kept by every later write, including the reset (`clear_verified_job_info` copies it),
    so a node is never re-anchored under the same executor id.

    Legacy (GPU_ANCHOR_HARD_ENABLED off): any difference is GPU_UUID_CHANGED, fatal, verification reset.
    The event's what_we_saw also carries the hard rule's decision so it can be read from the stored rows.
    This check now runs before SpecChangeCheck (pipeline_factory), so every set change reaches it: a missing
    or added card is filed as GPU_UUID_CHANGED (the count change made it SPEC_CHANGED before), and SpecChangeCheck
    only fires when the UUIDs match but the model:count string differs.

    Hard (GPU_ANCHOR_HARD_ENABLED on, DAH-3457):
    - strict subset of the anchor  -> GPU_MISSING, fatal, verification reset; the node passes again on the
      first cycle that shows the full set;
    - any UUID outside the anchor  -> GPU_UUID_CHANGED with `anchor_broken`, fatal, verification reset, and
      the record is marked broken: every later cycle fails the same way until the provider re-registers
      the node (a new executor id gets a new anchor).
    A node whose record is already marked broken fails before the sets are compared. With the flag off again
    the mark is kept in the record but ignored; it applies again on the next flip on.
    """

    check_id = "gpu.validate.fingerprint"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        verified = ctx.verified or {}
        prev_uuids = verified.get(GPU_ANCHOR_KEY) or ""
        current_uuids = ctx.state.gpu_uuids or ""
        hard = settings.GPU_ANCHOR_HARD_ENABLED

        if hard and verified.get(GPU_ANCHOR_BROKEN_KEY):
            event = render_message(
                Msg.ANCHOR_BROKEN,
                ctx=ctx,
                check_id=self.check_id,
                what={"previous": prev_uuids, "current": current_uuids, "anchor_broken": True},
            )
            return CheckResult(passed=False, event=event)

        anchor = split_uuids(prev_uuids)
        current = split_uuids(current_uuids)

        if anchor and current and anchor != current:
            missing = sorted(set(anchor) - set(current))
            unexpected = sorted(set(current) - set(anchor))
            # Sets decide: only cards missing = GPU_MISSING; any card outside the anchor = broken. A scrape that
            # lists every anchored card and one of them twice differs with both lists empty; it is not the
            # anchored set either, so it breaks the anchor rather than passing.
            decision = DECISION_GPU_MISSING if missing and not unexpected else DECISION_ANCHOR_BROKEN
            what = {
                "previous": prev_uuids,
                "current": current_uuids,
                "missing": missing,
                "unexpected": unexpected,
            }
            if not missing and not unexpected:
                what["duplicate_uuid"] = True

            if not hard:
                event = render_message(
                    Msg.UUID_CHANGED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={**what, "anchor_hard_would_be": decision},
                )
                return CheckResult(passed=False, event=event, updates={"clear_verified_job_info": True})

            if decision == DECISION_GPU_MISSING:
                event = render_message(Msg.GPU_MISSING, ctx=ctx, check_id=self.check_id, what=what)
                return CheckResult(passed=False, event=event, updates={"clear_verified_job_info": True})

            event = render_message(
                Msg.ANCHOR_BROKEN,
                ctx=ctx,
                check_id=self.check_id,
                what={**what, "anchor_broken": True},
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"clear_verified_job_info": True, "gpu_anchor_broken": True},
            )

        event = render_message(
            Msg.UUID_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_uuids": current_uuids},
        )
        return CheckResult(passed=True, event=event)
