from __future__ import annotations

from services.collateral_status import get_collateral_status_reader

from ..messages import CollateralStatusMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class CollateralStatusCheck:
    """Report collateral_deposited with the meaning its readers expect; never gates the run or the score."""

    check_id = "gpu.validate.collateral"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        gpu_count = ctx.state.gpu_count
        if gpu_count is None:
            gpu_count = ctx.state.specs.get("gpu", {}).get("count", 0)
        gpu_details = ctx.state.gpu_details or ctx.state.specs.get("gpu", {}).get("details", [])
        gpu_model = gpu_details[0].get("name") if gpu_details else None

        status, cached = await get_collateral_status_reader().status(
            miner_hotkey=ctx.miner_hotkey,
            executor_uuid=ctx.executor.uuid,
            gpu_model=gpu_model,
            gpu_count=gpu_count,
        )
        if status.read_failed:
            template = Msg.READ_FAILED
        elif status.deposited:
            template = Msg.DEPOSITED
        else:
            template = Msg.NOT_DEPOSITED
        what = {
            "collateral_deposited": status.deposited,
            "contract_version": status.contract_version,
            "cached": cached,
        }
        if status.collateral_tao is not None:
            what["collateral_tao"] = str(status.collateral_tao)
        if status.required_tao is not None:
            what["required_tao"] = str(status.required_tao)
        if status.error_message:
            what["error_message"] = status.error_message
        event = render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(
            passed=True,
            event=event,
            updates={
                "collateral_deposited": status.deposited,
                "contract_version": status.contract_version,
            },
        )
