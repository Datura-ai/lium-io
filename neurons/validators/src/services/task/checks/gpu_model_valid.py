from __future__ import annotations

from collections import Counter

from services.redis_service import GPU_ANCHOR_KEY

from ..messages import GpuModelMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .banned_provider import has_live_rental
from .gpu_fingerprint import split_uuids
from .post_rental_gpu_recheck import record_gpu_drop
from .rented_gpu_drop import _listed_uuids


class GpuModelValidCheck:
    """Gate validation on supported GPU SKUs and healthy scrape output.

    This mirrors the legacy guard that rejected unknown models, zero counts, or mismatched
    detail lists. It prevents us from handing out scores when the scrape clearly failed or
    when a miner advertises off-policy hardware.

    A detail list whose length differs from the count (GPU_DETAILS_MISMATCH) also clears the
    verified job (DAH-3519): in production this is a scrape that enumerated fewer cards than the
    node reports, which is what a card that fell off the bus looks like. The check is fatal and
    runs before SpecChangeCheck, so the reset has to happen here; the node then leaves the market
    through the same path banned_gpu, gpu_fingerprint, duplicate_executor and spec_change use,
    rented or idle, and is verified again on the first cycle whose scrape lists every card. The drop also
    owes a post-rental recheck (post_rental_gpu_recheck.py).

    Every card is read, not only the first (DAH-4304): each must be a supported model, and a node
    whose cards are more than one model fails with GPU_MODEL_MIXED.
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

        card_models = []
        if gpu_count > 0:
            card_models = [detail.get("name", None) for detail in gpu_details]
        gpu_model = card_models[0] if card_models else None

        if gpu_model not in gpu_model_rates_map:
            return self._unsupported_model_result(ctx, gpu_model, gpu_count, fatal=True)

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
            anchor = split_uuids((ctx.verified or {}).get(GPU_ANCHOR_KEY) or "")
            await record_gpu_drop(ctx, sorted(set(anchor) - set(_listed_uuids(gpu_details))))
            return CheckResult(passed=False, event=event, updates={"clear_verified_job_info": True})

        card_mismatch = self._mixed_or_unsupported_card_result(ctx, card_models, gpu_count)
        if card_mismatch is not None:
            return card_mismatch

        event = render_message(
            Msg.MODEL_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_model": gpu_model, "gpu_count": gpu_count},
        )
        return CheckResult(passed=True, event=event)

    def _mixed_or_unsupported_card_result(
        self, ctx: Context, card_models: list[str | None], gpu_count: int
    ) -> CheckResult | None:
        """DAH-4304: a node is listed, priced and rented as one model, so every GPU on it must be
        the same supported model. A live rental is not cut, as with a GPU ban."""
        fatal = not has_live_rental(ctx)
        unsupported = [model for model in card_models if model not in ctx.config.gpu_model_rates]
        if unsupported:
            return self._unsupported_model_result(ctx, unsupported[0], gpu_count, fatal=fatal)

        model_counts = dict(Counter(card_models))
        if len(model_counts) == 1:
            return None
        models_found = ", ".join(f"{count}x {model}" for model, count in model_counts.items())
        event = render_message(
            Msg.MODEL_MIXED,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_models": model_counts, "gpu_count": gpu_count},
            remediation=(
                "Every GPU on a node must be the same model. Models found on this node: "
                f"{models_found}."
            ),
        )
        return CheckResult(passed=False, event=event, fatal=fatal)

    def _unsupported_model_result(
        self, ctx: Context, gpu_model: str | None, gpu_count: int, *, fatal: bool
    ) -> CheckResult:
        supported_models = list(ctx.config.gpu_model_rates.keys())
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
        return CheckResult(passed=False, event=event, fatal=fatal)
