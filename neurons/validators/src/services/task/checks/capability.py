from __future__ import annotations

from typing import Awaitable, Callable

from ..models import build_msg
from ..pipeline import CheckResult, Context


class CapabilityCheck:
    """Run the containerised GPU capability probe (nvidia-smi in Docker).

    This executes the same `validate_gpu_model_and_process_job` command used before, which
    verifies that containers can see the GPUs. Failing it previously zeroed the score, so
    keeping it prevents miners from hiding driver issues behind a good scrape.
    """

    check_id = "gpu.validate.capability"
    fatal = True

    def __init__(self, *, capability_runner: Callable[[Context], Awaitable[bool]]):
        self.capability_runner = capability_runner

    async def run(self, ctx: Context) -> CheckResult:
        ok = await self.capability_runner(ctx)

        if ok:
            event = build_msg(
                event="GPU capability validated",
                reason="GPU_VERIFY_OK",
                severity="info",
                category="env",
                impact="Proceed",
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=True, event=event)

        event = build_msg(
            event="GPU capability verification failed",
            reason="GPU_VERIFY_FAILED",
            severity="error",
            category="env",
            impact="Score set to 0",
            remediation="Run Docker GPU diagnostics (nvidia-smi) and ensure containers can access GPUs.",
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=False, event=event)
