from __future__ import annotations

from ..models import build_msg
from ..pipeline import CheckResult, Context


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
            event = build_msg(
                event="No GPU fingerprints captured",
                reason="GPU_UUID_EMPTY",
                severity="info",
                category="env",
                impact="Proceed",
                remediation="Ensure the scrape script emits GPU UUIDs via nvidia-smi",
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=True, event=event)

        uuids = [u for u in current_uuids.split(",") if u]

        redis_service = ctx.services.redis
        banned_guids = await redis_service.get_banned_guids()
        is_banned = any(guid in banned_guids for guid in uuids)

        if is_banned:
            event = build_msg(
                event="GPU model temporarily ineligible",
                reason="GPU_BANNED",
                severity="warning",
                category="policy",
                impact="Score set to 0; verification cleared",
                remediation="Swap to eligible GPUs or wait for policy updates before retrying validation.",
                what={"gpu_uuids": current_uuids},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"clear_verified_job_info": True},
            )

        event = build_msg(
            event="GPU fingerprints allowed",
            reason="GPU_BANNED_CHECK_OK",
            severity="info",
            category="policy",
            impact="Proceed",
            what={"gpu_uuids": current_uuids},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=True, event=event)
