from __future__ import annotations

from ..models import build_msg
from ..pipeline import CheckResult, Context


class SpecChangeCheck:
    check_id = "gpu.validate.spec_change"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        prev_spec = (ctx.verified or {}).get("spec") or ""
        current_spec = ctx.gpu_model_count or ""

        if prev_spec and current_spec and prev_spec != current_spec:
            event = build_msg(
                event="GPU inventory changed",
                reason="SPEC_CHANGED",
                severity="warning",
                category="env",
                impact="Verification reset; score set to 0",
                remediation=(
                    "Keep GPU configuration stable between checks. Avoid hot-plugging GPUs or "
                    "changing MIG profiles mid-validation."
                ),
                what={
                    "previous": prev_spec,
                    "current": current_spec,
                },
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"clear_verified_job_info": True},
            )

        event = build_msg(
            event="GPU inventory stable",
            reason="SPEC_UNCHANGED",
            severity="info",
            category="env",
            impact="Proceed",
            what={"gpu_model_count": current_spec},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=True, event=event)
