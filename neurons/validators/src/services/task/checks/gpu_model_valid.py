from __future__ import annotations

from ..messages import GpuModelMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class GpuModelValidCheck:
    """Gate validation on supported GPU SKUs and healthy scrape output.

    This mirrors the legacy guard that rejected unknown models, zero counts, or mismatched
    detail lists. It prevents us from handing out scores when the scrape clearly failed or
    when a miner advertises off-policy hardware.
    """

    check_id = "gpu.validate.model"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        gpu_model_rates = ctx.config.gpu_model_rates
        if not gpu_model_rates:
            event = render_message(
                Msg.POLICY_MISSING,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=False, event=event)

        gpu_model_rates_map = gpu_model_rates
        specs = ctx.state.specs
        gpu_count = ctx.state.gpu_count
        if gpu_count is None:
            gpu_count = specs.get("gpu", {}).get("count", 0)
        gpu_details = ctx.state.gpu_details
        if not gpu_details:
            gpu_details = specs.get("gpu", {}).get("details", [])

        gpu_model = None
        if gpu_count > 0 and len(gpu_details) > 0:
            gpu_model = gpu_details[0].get("name", None)

        if not gpu_model_rates_map.get(gpu_model):
            supported_models = list(gpu_model_rates_map.keys())
            event = render_message(
                Msg.MODEL_UNSUPPORTED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "gpu_model": gpu_model,
                    "gpu_count": gpu_count,
                    "supported_models": supported_models,
                },
                remediation=(
                    "Use a supported GPU model. Supported models: "
                    f"{', '.join(supported_models[:5])}{'...' if len(supported_models) > 5 else ''}"
                ),
            )
            return CheckResult(passed=False, event=event)

        if gpu_count == 0:
            event = render_message(
                Msg.COUNT_ZERO,
                ctx=ctx,
                check_id=self.check_id,
                what={"gpu_count": gpu_count},
            )
            return CheckResult(passed=False, event=event)

        if len(gpu_details) != gpu_count:
            event = render_message(
                Msg.DETAILS_MISMATCH,
                ctx=ctx,
                check_id=self.check_id,
                what={"gpu_count": gpu_count, "details_len": len(gpu_details)},
            )
            return CheckResult(passed=False, event=event)

        event = render_message(
            Msg.MODEL_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_model": gpu_model, "gpu_count": gpu_count},
        )
        return CheckResult(passed=True, event=event)
