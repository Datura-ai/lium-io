from __future__ import annotations

from ..models import build_msg
from ..pipeline import CheckResult, Context


class CollateralCheck:
    """Confirm collateral eligibility so scores respect marketplace policy.

    Legacy logic zeroed out jobs when miners lacked the required bond. Keeping the
    decision close to the top of the pipeline ensures we do not waste time probing hosts
    that will ultimately be rejected by staking rules.
    """

    check_id = "gpu.validate.collateral"

    def __init__(self, *, collateral_service, enable_no_collateral: bool):
        self.collateral_service = collateral_service
        self.fatal = not enable_no_collateral

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs
        gpu_count = ctx.state.gpu_count
        if gpu_count is None:
            gpu_count = specs.get("gpu", {}).get("count", 0)
        gpu_details = ctx.state.gpu_details
        if not gpu_details:
            gpu_details = specs.get("gpu", {}).get("details", [])
        gpu_model = gpu_details[0].get("name") if gpu_details else None

        collateral_deposited, error_message, contract_version = await self.collateral_service.is_eligible_executor(
            miner_hotkey=ctx.miner_hotkey,
            executor_uuid=ctx.executor.uuid,
            gpu_model=gpu_model,
            gpu_count=gpu_count,
        )

        if collateral_deposited:
            event = build_msg(
                event="Collateral verified",
                reason="COLLATERAL_OK",
                severity="info",
                category="policy",
                impact="Proceed",
                what={"collateral_deposited": True, "contract_version": contract_version},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
        else:
            event = build_msg(
                event="No collateral deposited",
                reason="COLLATERAL_MISSING",
                severity="warning",
                category="policy",
                impact="Score may be reduced or set to 0 based on policy",
                remediation=(
                    f"Deposit collateral for this executor. Error: {error_message}"
                    if error_message
                    else "Deposit collateral for this executor"
                ),
                what={"collateral_deposited": False, "error_message": error_message},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
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
