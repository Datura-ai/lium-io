from __future__ import annotations

from ..messages import BannedGpuMessages as Msg, render_message
from ..pipeline = CheckResult, Context


class BannedGpuCheck:
    """Block miners whose GPU UUIDs appear on the banlist maintained in Redis.

    Legacy validation refused to score temporarily ineligible GPUs (e.g., due to fraud or
    hardware defects). Keeping this explicit check lets policy updates propagate without
    editing the core pipeline.
    """

    check_id = "gpu.validate.banned"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        current_uuids = ctx.state.gpu_uuids or ""

        if not current_uuids:
            event = render_message(
                Msg.UUID_EMPTY,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=True, event=event)

        uuids = [u for u in current_uuids.split(",") if u]

        redis_service = ctx.services.redis
        banned_guids = await redis_service.get_banned_guids()
        is_banned = any(guid in banned_guids for guid in uuids)

        if is_banned:
            event = render_message(
                Msg.GPU_BANNED,
                ctx=ctx,
                check_id=self.check_id,
                what={"gpu_uuids": current_uuids},
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"clear_verified_job_info": True},
            )

        event = render_message(
            Msg.GPU_ALLOWED,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_uuids": current_uuids},
        )
        return CheckResult(passed=True, event=event)
