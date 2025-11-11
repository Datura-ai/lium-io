from __future__ import annotations

from services.const import UNRENTED_MULTIPLIER

from ..messages import FinalizeMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class FinalizeCheck:
    """Aggregate pipeline results into the structured log + success flag.

    This check produces the user-facing log message and translates score warnings into
    remediation guidance. Housing it in a dedicated check makes the terminal behaviour
    explicit and easy to review.
    """

    check_id = "pipeline.finalize"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        success = ctx.score > 0
        severity = "info" if success else "warning"

        if ctx.score_warning:
            remediation = ("No action needed." + ctx.score_warning) if success else ("Address issues:" + ctx.score_warning)
        else:
            remediation = "No action needed." if success else "Address issues."

        event = render_message(
            Msg.COMPLETED,
            ctx=ctx,
            check_id=self.check_id,
            severity=severity,
            impact=f"Job score={ctx.job_score}, actual score={ctx.score}",
            remediation=remediation,
            what={
                "gpu_model": ctx.state.gpu_model,
                "gpu_count": ctx.state.gpu_count if ctx.state.gpu_count is not None else 0,
                "contract_version": ctx.contract_version,
                "unrented_multiplier": UNRENTED_MULTIPLIER,
                "sysbox_runtime": ctx.state.sysbox_runtime,
            },
        )

        return CheckResult(
            passed=True,
            event=event,
            updates={
                "success": success,
                "log_status": severity,
                "log_text": event.event,
            },
        )
