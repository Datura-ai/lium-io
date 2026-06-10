from __future__ import annotations

from dataclasses import replace

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

        filler_container = _get_filler_only_container(ctx)
        if filler_container:
            event = render_message(
                Msg.FILLER_SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={"filler_container": filler_container},
            )
            return CheckResult(passed=True, event=event)

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
                what={"metrics": result.metrics},
            )
            # Carry FP32 TFLOPS metrics into pipeline state so ResultHandler can nest them
            # in the published specs. Only on success and only when present (fail-safe:
            # None leaves state untouched and no gpu_metrics key is ever published).
            updates: dict = {}
            if result.metrics is not None:
                updates["state"] = replace(ctx.state, gpu_metrics=result.metrics)
            return CheckResult(passed=True, event=event, updates=updates)

        # Build detailed failure information
        failure_details = {}
        if result is not None:
            failure_details = {
                "error": result.error_message,
                "expected_uuid": result.expected_uuid,
                "returned_uuid": result.returned_uuid,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "metrics": result.metrics,
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


def _get_filler_only_container(ctx: Context) -> str | None:
    rented_data = ctx.state.rented_data
    if not rented_data:
        return None

    filler_container = rented_data.get_filler_container(ctx.executor.uuid)
    rented_executor = rented_data.executors.get(ctx.executor.uuid)
    has_customer_rental = bool(rented_executor and rented_executor.pods)
    return filler_container if filler_container and not has_customer_rental else None
