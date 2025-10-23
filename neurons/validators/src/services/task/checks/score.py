from __future__ import annotations

from ..messages import ScoreMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class ScoreCheck:
    """Convert pipeline context into the same actual/job score pair as the legacy flow.

    By isolating `calc_scores` here we make the business logic transparent and keep the
    final scoring tweaks (collateral, port count, rentals) reviewable without digging
    through `_handle_task_result`.
    """

    check_id = "gpu.validate.score"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        score_calculator = ctx.services.score_calculator
        actual_score, job_score, warning_message = score_calculator(
            ctx.state.gpu_model or "",
            ctx.collateral_deposited,
            ctx.is_rental_succeed,
            ctx.contract_version or "",
            ctx.rented,
            ctx.port_count,
        )

        event = render_message(
            Msg.SCORE_COMPUTED,
            ctx=ctx,
            check_id=self.check_id,
            impact=f"Job score={job_score}, actual score={actual_score}",
            what={
                "actual_score": actual_score,
                "job_score": job_score,
                "warning_message": warning_message,
            },
        )

        return CheckResult(
            passed=True,
            event=event,
            updates={
                "score": actual_score,
                "job_score": job_score,
                "score_warning": warning_message or None,
            },
        )
