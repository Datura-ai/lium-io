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

        result = None
        failure_reason = None
        try:
            result = await validation_service.validate_gpu_model_and_process_job(
                ssh_client=ctx.ssh,
                executor_info=ctx.executor,
                default_extra=ctx.default_extra,
                machine_spec=specs,
            )
        except Exception as exc:
            failure_reason = str(exc)

        if result and result.success:
            event = render_message(
                Msg.VERIFY_OK,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=True, event=event)

        # Build detailed failure information
        failure_details = {}
        if result is not None:
            failure_details = {
                "error": result.error_message,
                "expected_uuid": result.expected_uuid,
                "returned_uuid": result.returned_uuid,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        elif failure_reason:
            failure_details = {"error": failure_reason}

        event = render_message(
            Msg.VERIFY_FAILED,
            ctx=ctx,
            check_id=self.check_id,
            what=failure_details,
        )
        return CheckResult(passed=False, event=event)
