from __future__ import annotations

from ..messages import GpuCountMessages as Msg, render_message
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
            event = render_message(
                Msg.POLICY_MISSING,
                ctx=ctx,
                check_id=self.check_id,
                what={"observed_count": gpu_count},
            )
            return CheckResult(passed=False, event=event)

        if gpu_count > max_gpu_count:
            event = render_message(
                Msg.COUNT_EXCEEDS,
                ctx=ctx,
                check_id=self.check_id,
                what={"count": gpu_count, "max_allowed": max_gpu_count},
            )
            return CheckResult(passed=False, event=event)

        event = render_message(
            Msg.COUNT_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"count": gpu_count, "max_allowed": max_gpu_count},
        )
        return CheckResult(passed=True, event=event)
