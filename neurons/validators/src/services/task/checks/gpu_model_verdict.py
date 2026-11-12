from __future__ import annotations

from ..messages import CapabilityMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .banned_provider import has_live_rental
from .capability import gpu_model_mismatch_result, stored_gpu_model_verdict


class GpuModelVerdictCheck:
    """DAH-6241: a rented node halts at TenantEnforcementCheck, before the capability matmul, so its
    GPU model is judged by the last verdict the matmul or a filler sample stored for it. No matmul
    runs beside the renter's pod. Idle and filler nodes are judged in CapabilityCheck."""

    check_id = "gpu.validate.model_verdict"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        rented = has_live_rental(ctx)
        mismatches = (await stored_gpu_model_verdict(ctx))[1] if rented else []
        if mismatches:
            # the cycle goes on, so the rental is still verified; score 0 and hidden from new rentals
            return gpu_model_mismatch_result(
                ctx, self.check_id, {**mismatches[0].as_dict(), "transport": "stored", "rented": True}
            )
        event = render_message(
            Msg.GPU_MODEL_VERDICT_OK, ctx=ctx, check_id=self.check_id, what={"rented": rented}
        )
        return CheckResult(passed=True, event=event)
