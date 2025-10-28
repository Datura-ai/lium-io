from __future__ import annotations

from ..messages import DuplicateExecutorMessages as Msg, render_message
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
            event = render_message(
                Msg.DUPLICATE,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "executor_uuid": ctx.executor.uuid,
                    "miner_hotkey": ctx.miner_hotkey,
                },
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"clear_verified_job_info": True},
            )

        event = render_message(
            Msg.UNIQUE,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid},
        )
        return CheckResult(passed=True, event=event)
