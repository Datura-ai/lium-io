from __future__ import annotations

from ..models import build_msg
from ..pipeline import CheckResult, Context

DUPLICATED_MACHINE_SET = "duplicated_machine:set"

class DuplicateExecutorCheck:
    """Ensure a miner is not registering the same executor UUID multiple times.

    The original pipeline cleared verification when Redis flagged duplicates. Keeping the
    guard avoids wasted scoring cycles and enforces one-to-one miner/executor mappings.
    """

    check_id = "executor.validate.duplicate"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        redis_service = ctx.services.redis
        is_duplicate = await redis_service.is_elem_exists_in_set(
            DUPLICATED_MACHINE_SET,
            f"{ctx.miner_hotkey}:{ctx.executor.uuid}",
        )

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
