from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from core.config import settings
from services.matrix_validation_service import UUID_MISMATCH_ERROR_PREFIX

from ..messages import CapabilityMessages as Msg, MessageTemplate, render_message
from ..pipeline import CheckResult, Context

if TYPE_CHECKING:
    from services.matrix_validation_service import ValidationResult


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

        # DAH-3011: a first, unscored verification sizes the matmul from a VRAM budget instead of
        # the whole card (same challenge/seal/UUID check). The keyword is only passed on that path
        # so the scored call is byte-for-byte today's.
        sizing = {"vram_budget_mb": settings.FIRST_PASS_MATMUL_VRAM_MB} if ctx.config.first_pass else {}

        result = None
        failure_reason = None
        try:
            result = await validation_service.validate_gpu_model_and_process_job(
                ssh_client=ctx.ssh,
                executor_info=ctx.executor,
                default_extra=ctx.default_extra,
                machine_spec=specs,
                **sizing,
            )
        except Exception as exc:
            failure_reason = str(exc)

        if result and result.success:
            what: dict = {"metrics": result.metrics}
            if sizing:
                what["first_pass_vram_budget_mb"] = sizing["vram_budget_mb"]
            event = render_message(
                Msg.VERIFY_OK,
                ctx=ctx,
                check_id=self.check_id,
                what=what,
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
                "stderr_tail": _tail(result.stderr),
                "metrics": result.metrics,
            }
        elif failure_reason:
            failure_details = {"error": failure_reason}

        template = _failure_template(result)
        event = render_message(
            template,
            ctx=ctx,
            check_id=self.check_id,
            what=failure_details,
        )
        return CheckResult(passed=False, event=event)


STDERR_TAIL_CHARS = 300

# The native verifier prints "Failed to allocate d_A: <cudaGetErrorString>" for any cudaMalloc
# failure, so the error string after the colon is what tells the cases apart. The executor then
# answers no uuid at all, which the service reports as "UUID mismatch: expected '<uuid>', got
# 'None'" — the wrong story for the provider. Each case is its own reason code (#1335 and #1352
# settled this on 11 Sep: nothing downstream enumerates the code, and GPU_VERIFY_TIMEOUT already
# works this way). Order matters: the specific CUDA errors are matched before the out-of-memory
# markers.
#
# cudaErrorSystemNotReady (802): CUDA cannot initialise although nvidia-smi works — on HGX
# H100/H200 boards the NVLink fabric is not ready (Fabric Manager / fabric partition / driver).
# ticket-0318 (DAH-3362): a 1x H200 node failed every cycle with exactly this string.
# Message forms only: the verifier formats every error with cudaGetErrorString, never the enum
# name, and the executor's wrapper writes nothing of its own to stderr.
_CUDA_NOT_READY_MARKERS = ("system not yet initialized",)
# cudaErrorNoDevice (100), or cudaErrorInsufficientDriver (35) — what the statically linked CUDA
# runtime in the native verifier reports when the toolkit mounted no libcuda into the container (or
# the host driver is older than that runtime): the probe runs inside the executor container and
# finds no usable GPU there.
_CONTAINER_GPU_ACCESS_MARKERS = (
    "no CUDA-capable device",
    "CUDA driver version is insufficient for CUDA runtime version",
)
# cudaMalloc failing for memory (DAH-3264: 49 such verdicts on 31 executors in 26 h).
_VRAM_UNAVAILABLE_MARKERS = (
    "out of memory",
    "cudaErrorMemoryAllocation",
)
_NO_UUID = {"", "none", "null"}


def _tail(text: str | None, limit: int = STDERR_TAIL_CHARS) -> str:
    text = (text or "").strip()
    return text[-limit:] if len(text) > limit else text


def _failure_template(result: ValidationResult | None) -> MessageTemplate:
    """Pick the reason for a failed capability probe.

    A timeout keeps its own reason. An answer with no uuid is classified by what stderr/stdout
    says: CUDA not ready (fabric) → `VERIFY_FAILED_CUDA_NOT_READY`, no CUDA device →
    `VERIFY_FAILED_CONTAINER_GPU_ACCESS`, out of memory → `VERIFY_FAILED_VRAM_UNAVAILABLE`.
    Each is its own reason code, as `GPU_VERIFY_TIMEOUT` is. A returned uuid that does not match
    — the anti-spoof case; the expected uuid is a per-call nonce, so a genuine executor never
    answers a wrong one — stays the generic `VERIFY_FAILED`, whatever stderr says. So does any
    failure the service reported for another reason (a sealed result that failed authentication)
    and anything unrecognised.
    """
    if result is None:
        return Msg.VERIFY_FAILED
    if result.timed_out:
        return Msg.VERIFY_TIMEOUT
    # Only the service's "UUID mismatch" failure is the probe's own answer. Its other empty-uuid
    # results (a sealed blob that failed authentication, a stdout that could not be parsed) are
    # not allocation failures, and stderr is miner-controlled: they keep the generic reason.
    if not (result.error_message or "").startswith(UUID_MISMATCH_ERROR_PREFIX):
        return Msg.VERIFY_FAILED
    returned = (result.returned_uuid or "").strip().lower()
    if returned not in _NO_UUID:
        return Msg.VERIFY_FAILED
    output = f"{result.stderr or ''}\n{result.stdout or ''}".lower()
    for markers, template in (
        (_CUDA_NOT_READY_MARKERS, Msg.VERIFY_FAILED_CUDA_NOT_READY),
        (_CONTAINER_GPU_ACCESS_MARKERS, Msg.VERIFY_FAILED_CONTAINER_GPU_ACCESS),
        (_VRAM_UNAVAILABLE_MARKERS, Msg.VERIFY_FAILED_VRAM_UNAVAILABLE),
    ):
        if any(m.lower() in output for m in markers):
            return template
    return Msg.VERIFY_FAILED


def _get_filler_only_container(ctx: Context) -> str | None:
    rented_data = ctx.state.rented_data
    if not rented_data:
        return None

    filler_container = rented_data.get_filler_container(ctx.executor.uuid)
    rented_executor = rented_data.executors.get(ctx.executor.uuid)
    has_customer_rental = bool(rented_executor and rented_executor.pods)
    return filler_container if filler_container and not has_customer_rental else None
