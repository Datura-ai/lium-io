from __future__ import annotations

from typing import Callable, Tuple

from ..models import build_msg
from ..pipeline import CheckResult, Context


class ScoreCheck:
    """Convert pipeline context into the same actual/job score pair as the legacy flow.

    By isolating `calc_scores` here we make the business logic transparent and keep the
    final scoring tweaks (collateral, port count, rentals) reviewable without digging
    through `_handle_task_result`.
    """

    check_id = "gpu.validate.score"
    fatal = False

    def __init__(self, *, score_calculator: Callable[[str, bool, bool, str, bool, int], Tuple[float, float, str]]):
        self.score_calculator = score_calculator

    async def run(self, ctx: Context) -> CheckResult:
        actual_score, job_score, warning_message = self.score_calculator(
            ctx.state.gpu_model or "",
            ctx.collateral_deposited,
            ctx.is_rental_succeed,
            ctx.contract_version or "",
            ctx.rented,
            ctx.port_count,
        )

        event = build_msg(
            event="Scores computed",
            reason="SCORE_COMPUTED",
            severity="info",
            category="policy",
            impact=f"Job score={job_score}, actual score={actual_score}",
            what={
                "actual_score": actual_score,
                "job_score": job_score,
                "warning_message": warning_message,
            },
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
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
