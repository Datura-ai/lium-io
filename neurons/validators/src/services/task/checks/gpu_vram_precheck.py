from __future__ import annotations

from services.gpu_precheck import GpuPrecheckError, precheck_gpu_spec

from ..messages import GpuVramMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class GpuVramPrecheck:
    """Gate validation on GPU model <-> VRAM consistency.

    This is a pure-data check: it only needs ``gpu_model`` and the first GPU's
    reported ``capacity`` (MB), both populated by MachineSpecScrapeCheck. It has
    no SSH/GPU/native dependency, so it runs early — right after
    GpuModelValidCheck and BEFORE the rented short-circuit
    (TenantEnforcementCheck). Previously the same precheck only ran inside
    CapabilityCheck (the matmul), which sits after the short-circuit, so rented
    executors bypassed it entirely. Running it here gates rented and idle
    executors alike. The native matmul in CapabilityCheck remains authoritative
    for everything the precheck does not cover.
    """

    check_id = "gpu.validate.vram_precheck"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        gpu_model = ctx.state.gpu_model
        gpu_details = ctx.state.gpu_details
        capacity_mb = 0
        if gpu_details:
            capacity_mb = gpu_details[0].get("capacity", 0) or 0

        try:
            precheck_gpu_spec(gpu_model=gpu_model, gpu_capacity_mb=int(capacity_mb))
        except GpuPrecheckError as exc:
            event = render_message(
                Msg.VRAM_REJECTED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "gpu_model": gpu_model,
                    "gpu_capacity_mb": capacity_mb,
                    "error": str(exc),
                },
            )
            return CheckResult(passed=False, event=event)

        event = render_message(
            Msg.VRAM_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_model": gpu_model, "gpu_capacity_mb": capacity_mb},
        )
        return CheckResult(passed=True, event=event)
