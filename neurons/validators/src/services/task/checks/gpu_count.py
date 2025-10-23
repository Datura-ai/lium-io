from __future__ import annotations

from ..models import build_msg
from ..pipeline import CheckResult, Context


class GpuCountCheck:
    """Enforce the subnet's maximum visible GPU policy before scoring work.

    The old task rejected miners reporting more than `MAX_GPU_COUNT`; keeping this early
    avoids downstream scoring on hardware that would be disqualified regardless of
    collateral or performance.
    """

    check_id = "gpu.validate.count"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs
        gpu_count = ctx.state.gpu_count
        if gpu_count is None:
            gpu_count = specs.get("gpu", {}).get("count", 0)
        max_gpu_count = ctx.config.max_gpu_count

        if max_gpu_count is None:
            event = build_msg(
                event="GPU count policy missing",
                reason="GPU_COUNT_POLICY_MISSING",
                severity="error",
                category="policy",
                impact="Validation halted",
                remediation="Validator bug: max GPU count not configured in context",
                what={"observed_count": gpu_count},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        if gpu_count > max_gpu_count:
            event = build_msg(
                event="GPU count exceeds policy",
                reason="GPU_COUNT_EXCEEDS_MAX",
                severity="error",
                category="policy",
                impact="Score set to 0",
                remediation=(
                    f"Reduce visible GPU count to {max_gpu_count} or less (e.g., use CUDA_VISIBLE_DEVICES environment variable)"
                ),
                what={"count": gpu_count, "max_allowed": max_gpu_count},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        event = build_msg(
            event="GPU count within limits",
            reason="GPU_COUNT_OK",
            severity="info",
            category="policy",
            impact="Proceed",
            what={"count": gpu_count, "max_allowed": max_gpu_count},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=True, event=event)
