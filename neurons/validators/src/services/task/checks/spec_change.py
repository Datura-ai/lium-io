from __future__ import annotations

from ..messages import SpecChangeMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class SpecChangeCheck:
    """Reset verification when the GPU model:count tuple changes between runs.

    Validators previously cleared Redis when inventory shifted, forcing a fresh audit.
    Keeping that behaviour thwarts miners who hot-swap or reconfigure GPUs between checks
    to game the scoring window.
    """

    check_id = "gpu.validate.spec_change"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        prev_spec = (ctx.verified or {}).get("spec") or ""
        current_spec = ctx.state.gpu_model_count or ""

        if prev_spec and current_spec and prev_spec != current_spec:
            event = render_message(
                Msg.SPEC_CHANGED,
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
                updates={"clear_verified_job_info": True},
            )

        event = render_message(
            Msg.SPEC_UNCHANGED,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_model_count": current_spec},
        )
        return CheckResult(passed=True, event=event)
