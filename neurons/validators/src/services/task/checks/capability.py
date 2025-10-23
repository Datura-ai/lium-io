from __future__ import annotations

from ..messages import CapabilityMessages as Msg, render_message
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
            event = render_message(
                Msg.NO_SPECS,
                ctx=ctx,
                check_id=self.check_id,
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
            event = render_message(
                Msg.VERIFY_OK,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=True, event=event)

        event = render_message(
            Msg.VERIFY_FAILED,
            ctx=ctx,
            check_id=self.check_id,
            what={"error": failure_reason} if failure_reason else {},
        )
        return CheckResult(passed=False, event=event)
