"""Validation fast path: start the collateral read early, decide it where it is decided today.

`CollateralCheck` reads the collateral contract (p50 3.5 s over the fleet) and sits behind a dozen
pure-data GPU checks that take a few seconds together. This check, placed right after the scrape,
starts the same contract read with the same arguments as a background task; `CollateralCheck`
awaits it at its own place in the pipeline. Verdict, event and gates are `CollateralCheck`'s and
unchanged — only the seconds the read spent alone are gone. A prefetch whose arguments would differ
from what `CollateralCheck` computes is discarded there and the read runs again.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from ..messages import CollateralPrefetchMessages as Msg
from ..messages import render_message
from ..models import CollateralPrefetch, CollateralReadArgs
from ..pipeline import CheckResult, Context


def collateral_read_args(ctx: Context) -> CollateralReadArgs:
    """The arguments `CollateralCheck` hands the contract, computed the one way it computes them."""
    specs = ctx.state.specs
    gpu_count = ctx.state.gpu_count
    if gpu_count is None:
        gpu_count = specs.get("gpu", {}).get("count", 0)
    gpu_details = ctx.state.gpu_details
    if not gpu_details:
        gpu_details = specs.get("gpu", {}).get("details", [])
    gpu_model = gpu_details[0].get("name") if gpu_details else None
    return CollateralReadArgs(
        miner_hotkey=ctx.miner_hotkey,
        executor_uuid=ctx.executor.uuid,
        gpu_model=gpu_model,
        gpu_count=gpu_count,
    )


class CollateralPrefetchCheck:
    check_id = "gpu.validate.collateral_prefetch"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        args = collateral_read_args(ctx)
        if not args.gpu_count:
            # Nothing the contract can be asked about yet; GpuCountCheck fails such a run anyway.
            event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={"gpu_count": args.gpu_count})
            return CheckResult(passed=True, event=event)
        task = asyncio.ensure_future(
            ctx.services.collateral.is_eligible_executor(
                miner_hotkey=args.miner_hotkey,
                executor_uuid=args.executor_uuid,
                gpu_model=args.gpu_model,
                gpu_count=args.gpu_count,
            )
        )
        event = render_message(
            Msg.STARTED,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_model": args.gpu_model, "gpu_count": args.gpu_count},
        )
        return CheckResult(
            passed=True,
            event=event,
            updates={"state": replace(ctx.state, collateral_prefetch=CollateralPrefetch(args=args, task=task))},
        )
