from __future__ import annotations

from typing import Awaitable, Callable

from ..models import build_msg
from ..pipeline import CheckResult, Context


class VerifyXCheck:
    check_id = "gpu.validate.verifyx"
    fatal = True

    def __init__(
        self,
        *,
        verifyx_runner: Callable[[Context], Awaitable[object]],
        enabled: bool,
    ):
        self.verifyx_runner = verifyx_runner
        self.enabled = enabled

    async def run(self, ctx: Context) -> CheckResult:
        if not self.enabled:
            event = build_msg(
                event="VerifyX validation skipped",
                reason="VERIFYX_DISABLED",
                severity="info",
                category="env",
                impact="Proceed",
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=True, event=event)

        result = await self.verifyx_runner(ctx)

        if result.data and result.data.get("success"):
            updated_specs = dict(ctx.specs)
            updated_specs.update(
                {
                    "ram": result.data.get("ram", updated_specs.get("ram")),
                    "hard_disk": result.data.get("hard_disk", updated_specs.get("hard_disk")),
                    "network": result.data.get("network", updated_specs.get("network")),
                }
            )

            event = build_msg(
                event="VerifyX validation passed",
                reason="VERIFYX_OK",
                severity="info",
                category="env",
                impact="Proceed",
                what={"verifyx_success": True},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={
                    "specs": updated_specs,
                },
            )

        errors = None
        if result.data:
            errors = result.data.get("errors")
        errors = errors or result.error or "Unknown errors"

        event = build_msg(
            event="VerifyX validation failed",
            reason="VERIFYX_FAILED",
            severity="error",
            category="env",
            impact="Score set to 0",
            remediation="Run VerifyX locally to debug network, disk, and RAM probes.",
            what={"errors": errors},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=False, event=event)
