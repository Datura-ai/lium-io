from __future__ import annotations

from services.gpu_spec_table import H100_PCIE, H100_SXM

from ..messages import SpecChangeMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class SpecChangeCheck:
    """Reset verification when the GPU model:count tuple changes between runs.

    Validators previously cleared Redis when inventory shifted, forcing a fresh audit.
    Keeping that behaviour thwarts miners who hot-swap or reconfigure GPUs between checks
    to game the scoring window.
    """

    check_id = "gpu.validate.spec_change"
    replays_after_session_drop = True
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        prev_spec = (ctx.verified or {}).get("spec") or ""
        current_spec = ctx.state.gpu_model_count or ""

        if prev_spec and current_spec and _as_listed(prev_spec) != _as_listed(current_spec):
            event = render_message(
                Msg.SPEC_CHANGED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "previous": prev_spec,
                    "current": current_spec,
                },
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"clear_verified_job_info": True, "gpu_respec": True},
                # a live-rental GPU set change already failed GpuFingerprintCheck non-fatally: let the run reach the pod checks
                fatal=ctx.gpu_fingerprint_passed,
            )

        event = render_message(
            Msg.SPEC_UNCHANGED,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_model_count": current_spec},
        )
        return CheckResult(passed=True, event=event)


# DAH-6546: the same GPU under two datacenter names, one explicit pair as the backend's GPU-set move treats it
# (executor_gpu.py _swap_model); never a broad grouping, which could let a node swap to a weaker GPU without a reset
_SAME_MODEL = {H100_SXM: H100_PCIE, "NVIDIA B300 SXM6 PC": "NVIDIA B300 SXM6 AC"}


def _as_listed(spec: str) -> str:
    # DAH-5677: an H100 SXM-on-PCIe-adapter node verified before as "H100 80GB HBM3:N" is listed as "H100 PCIe:N" now
    model, sep, count = spec.rpartition(":")
    return f"{_SAME_MODEL[model]}{sep}{count}" if model in _SAME_MODEL else spec
