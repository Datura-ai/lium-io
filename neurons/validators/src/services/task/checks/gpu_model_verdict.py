from __future__ import annotations

from ..messages import CapabilityMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .capability import stored_gpu_model_verdict
from .rental_owner import rental_of_this_node


class GpuModelVerdictCheck:
    """DAH-6241: a rented node halts at TenantEnforcementCheck, before the capability matmul, so its
    GPU model is judged by the last verdict the matmul or a filler sample stored for it. No matmul
    runs beside the renter's pod. Idle and filler nodes are judged in CapabilityCheck."""

    check_id = "gpu.validate.model_verdict"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        rental = rental_of_this_node(ctx)
        mismatch = None
        if rental is not None and rental.pods:
            _, mismatch = await stored_gpu_model_verdict(ctx)
        if mismatch is not None:
            event = render_message(
                Msg.GPU_MODEL_MISMATCH,
                ctx=ctx,
                check_id=self.check_id,
                what={**mismatch, "transport": "stored", "rented": True},
            )
            return CheckResult(passed=False, event=event)
        event = render_message(
            Msg.GPU_MODEL_VERDICT_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"rented": bool(rental is not None and rental.pods)},
        )
        return CheckResult(passed=True, event=event)
