from __future__ import annotations

from typing import Awaitable, Callable

from ..models import build_msg
from ..pipeline import CheckResult, Context


class DuplicateExecutorCheck:
    check_id = "executor.validate.duplicate"
    fatal = True

    def __init__(self, *, duplicate_checker: Callable[[str, str], Awaitable[bool]]):
        self.duplicate_checker = duplicate_checker

    async def run(self, ctx: Context) -> CheckResult:
        is_duplicate = await self.duplicate_checker(ctx.miner_hotkey, ctx.executor.uuid)

        if is_duplicate:
            event = build_msg(
                event="Duplicate executor registration",
                reason="EXECUTOR_DUPLICATE",
                severity="warning",
                category="policy",
                impact="Score set to 0; verification cleared",
                remediation="Ensure every executor has a unique UUID and deregister duplicates before retrying.",
                what={
                    "executor_uuid": ctx.executor.uuid,
                    "miner_hotkey": ctx.miner_hotkey,
                },
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"clear_verified_job_info": True},
            )

        event = build_msg(
            event="Executor registration unique",
            reason="EXECUTOR_NOT_DUPLICATE",
            severity="info",
            category="policy",
            impact="Proceed",
            what={"executor_uuid": ctx.executor.uuid},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=True, event=event)
