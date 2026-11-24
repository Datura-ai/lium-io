from __future__ import annotations

from services.redis_service import GPU_ANCHOR_KEY

from ..messages import SpecChangeMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .gpu_fingerprint import as_listed, full_set_changed, listed_model, split_uuids


class SpecChangeCheck:
    """Reset verification when the GPU model:count tuple changes between runs.

    The same cards under another name are reset once, then verified as they are now. A full set of other cards
    of another model never verifies under this executor id: the node's own cards (or machine) must be back.
    """

    check_id = "gpu.validate.spec_change"
    replays_after_session_drop = True
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        prev_spec = (ctx.verified or {}).get("spec") or ""
        current_spec = ctx.state.gpu_model_count or ""

        if prev_spec and current_spec and as_listed(prev_spec) != as_listed(current_spec):
            # a full set of other cards of another model is another machine, not this node: the record keeps the
            # node's model and cards (the fingerprint check never re-anchors them), so it fails every cycle until
            # they are back. Only the same cards under a new name are verified as they are now (DAH-6295).
            other_machine = listed_model(prev_spec) != listed_model(current_spec) and full_set_changed(
                split_uuids((ctx.verified or {}).get(GPU_ANCHOR_KEY) or ""), split_uuids(ctx.state.gpu_uuids or "")
            )
            event = render_message(
                Msg.SPEC_MODEL_CHANGED if other_machine else Msg.SPEC_CHANGED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "previous": prev_spec,
                    "current": current_spec,
                },
            )
            return CheckResult(
                passed=False,
                event=event,
                updates=(
                    {"clear_verified_job_info": True}
                    if other_machine
                    else {"clear_verified_job_info": True, "gpu_respec": True}
                ),
                # a live-rental GPU set change already failed GpuFingerprintCheck non-fatally: let the run reach the pod checks
                fatal=ctx.gpu_fingerprint_passed,
            )

        event = render_message(
            Msg.SPEC_UNCHANGED,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_model_count": current_spec},
        )
        return CheckResult(passed=True, event=event)

