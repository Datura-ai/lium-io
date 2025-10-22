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

    def __init__(self, *, max_gpu_count: int):
        self.max_gpu_count = max_gpu_count

    async def run(self, ctx: Context) -> CheckResult:
        gpu_count = ctx.specs.get("gpu", {}).get("count", 0)

        if gpu_count > self.max_gpu_count:
            event = build_msg(
                event="GPU count exceeds policy",
                reason="GPU_COUNT_EXCEEDS_MAX",
                severity="error",
                category="policy",
                impact="Score set to 0",
                remediation=(
                    f"Reduce visible GPU count to {self.max_gpu_count} or less (e.g., use CUDA_VISIBLE_DEVICES environment variable)"
                ),
                what={"count": gpu_count, "max_allowed": self.max_gpu_count},
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
            what={"count": gpu_count, "max_allowed": self.max_gpu_count},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=True, event=event)
