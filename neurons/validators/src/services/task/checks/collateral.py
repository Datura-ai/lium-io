from __future__ import annotations

from ..messages import CollateralMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .collateral_prefetch import collateral_read_args


class CollateralCheck:
    """Confirm collateral eligibility so scores respect marketplace policy.

    Legacy logic zeroed out jobs when miners lacked the required bond. Keeping the
    decision close to the top of the pipeline ensures we do not waste time probing hosts
    that will ultimately be rejected by staking rules.
    """

    check_id = "gpu.validate.collateral"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        collateral_service = ctx.services.collateral
        enable_no_collateral = ctx.config.enable_no_collateral
        self.fatal = not enable_no_collateral

        args = collateral_read_args(ctx)
        gpu_count = args.gpu_count
        gpu_model = args.gpu_model

        # Validation fast path: CollateralPrefetchCheck started this same read earlier; its answer
        # is used only when it was asked the same question this check would ask now.
        prefetch = ctx.state.collateral_prefetch
        prefetched = prefetch is not None and prefetch.args == args
        if prefetched:
            collateral_deposited, error_message, contract_version = await prefetch.task
        else:
            if prefetch is not None:
                prefetch.task.cancel()
            collateral_deposited, error_message, contract_version = await collateral_service.is_eligible_executor(
                miner_hotkey=ctx.miner_hotkey,
                executor_uuid=ctx.executor.uuid,
                gpu_model=gpu_model,
                gpu_count=gpu_count,
            )

        if collateral_deposited:
            event = render_message(
                Msg.VERIFIED,
                ctx=ctx,
                check_id=self.check_id,
                what={"collateral_deposited": True, "contract_version": contract_version},
            )
        else:
            remediation = (
                f"Deposit collateral for this executor. Error: {error_message}"
                if error_message
                else Msg.MISSING.remediation
            )
            event = render_message(
                Msg.MISSING,
                ctx=ctx,
                check_id=self.check_id,
                what={"collateral_deposited": False, "contract_version": contract_version, "error_message": error_message},
                remediation=remediation,
            )

        passed = collateral_deposited or not self.fatal
        if prefetched:
            event.what_we_saw["prefetched"] = True

        return CheckResult(
            passed=passed,
            event=event,
            updates={
                "collateral_deposited": collateral_deposited,
                "collateral_error_message": error_message,
                "contract_version": contract_version,
            },
        )
