from __future__ import annotations

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

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs
        if not specs:
            event = build_msg(
                event="GPU capability skipped (no specs)",
                reason="GPU_VERIFY_SKIPPED",
                severity="error",
                category="env",
                impact="Validation halted",
                remediation="Run machine scrape before capability validation.",
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        validation_service = ctx.services.validation

        try:
            ok = await validation_service.validate_gpu_model_and_process_job(
                ssh_client=ctx.ssh,
                executor_info=ctx.executor,
                default_extra=ctx.default_extra,
                machine_spec=specs,
            )
        except Exception as exc:
            ok = False
            failure_reason = str(exc)
        else:
            failure_reason = None

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
            what={"error": failure_reason} if failure_reason else {},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=False, event=event)
