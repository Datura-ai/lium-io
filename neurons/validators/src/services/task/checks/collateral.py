from __future__ import annotations

from ..messages import CollateralMessages as Msg, render_message
from ..pipeline import CheckResult, Context


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

        specs = ctx.state.specs
        gpu_count = ctx.state.gpu_count
        if gpu_count is None:
            gpu_count = specs.get("gpu", {}).get("count", 0)
        gpu_details = ctx.state.gpu_details
        if not gpu_details:
            gpu_details = specs.get("gpu", {}).get("details", [])
        gpu_model = gpu_details[0].get("name") if gpu_details else None

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

        return CheckResult(
            passed=passed,
            event=event,
            updates={
                "collateral_deposited": collateral_deposited,
                "contract_version": contract_version,
            },
        )
