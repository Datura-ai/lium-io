from __future__ import annotations

from services.const import UNRENTED_MULTIPLIER

from ..models import build_msg
from ..pipeline import CheckResult, Context


class FinalizeCheck:
    """Aggregate pipeline results into the structured log + success flag.

    This mirrors the tail of `create_task_old`, producing the user-facing log message and
    translating score warnings into remediation guidance. Housing it in a dedicated check
    makes the terminal behaviour explicit and easy to review.
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

        event = build_msg(
            event="Validation task completed",
            reason="VALIDATION_COMPLETED",
            severity=severity,
            category="runtime",
            impact=f"Job score={ctx.job_score}, actual score={ctx.score}",
            remediation=remediation,
            what={
                "gpu_model": ctx.gpu_model,
                "gpu_count": ctx.gpu_count,
                "contract_version": ctx.contract_version,
                "unrented_multiplier": UNRENTED_MULTIPLIER,
                "sysbox_runtime": ctx.sysbox_runtime,
            },
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
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
