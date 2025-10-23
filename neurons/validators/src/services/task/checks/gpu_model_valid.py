from __future__ import annotations

from ..models import build_msg
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
            event = build_msg(
                event="GPU model policy missing",
                reason="GPU_MODEL_POLICY_MISSING",
                severity="error",
                category="policy",
                impact="Validation halted",
                remediation="Validator bug: GPU model rates not configured in context",
                what={},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        gpu_model_rates_map = gpu_model_rates
        gpu_count = ctx.specs.get("gpu", {}).get("count", 0)
        gpu_details = ctx.specs.get("gpu", {}).get("details", [])

        gpu_model = None
        if gpu_count > 0 and len(gpu_details) > 0:
            gpu_model = gpu_details[0].get("name", None)

        if not gpu_model_rates_map.get(gpu_model):
            supported_models = list(gpu_model_rates_map.keys())
            event = build_msg(
                event="GPU model not supported",
                reason="GPU_MODEL_UNSUPPORTED",
                severity="warning",
                category="policy",
                impact="Job skipped; score set to 0",
                remediation=(
                    "Use a supported GPU model. Supported models: "
                    f"{', '.join(supported_models[:5])}{'...' if len(supported_models) > 5 else ''}"
                ),
                what={
                    "gpu_model": gpu_model,
                    "gpu_count": gpu_count,
                    "supported_models": supported_models,
                },
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        if gpu_count == 0:
            event = build_msg(
                event="No GPUs detected",
                reason="GPU_COUNT_ZERO",
                severity="warning",
                category="env",
                impact="Job skipped; score set to 0",
                remediation="Ensure GPUs are properly installed and visible to the system",
                what={"gpu_count": gpu_count},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        if len(gpu_details) != gpu_count:
            event = build_msg(
                event="GPU count mismatch",
                reason="GPU_DETAILS_MISMATCH",
                severity="warning",
                category="env",
                impact="Job skipped; score set to 0",
                remediation="GPU count and details length don't match. Check GPU detection",
                what={"gpu_count": gpu_count, "details_len": len(gpu_details)},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        event = build_msg(
            event="GPU model validated",
            reason="GPU_MODEL_OK",
            severity="info",
            category="env",
            impact="Proceed",
            what={"gpu_model": gpu_model, "gpu_count": gpu_count},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=True, event=event)
