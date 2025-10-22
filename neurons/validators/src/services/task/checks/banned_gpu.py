from __future__ import annotations

from typing import Awaitable, Callable

from ..models import build_msg
from ..pipeline import CheckResult, Context


class BannedGpuCheck:
    check_id = "gpu.validate.banned"
    fatal = True

    def __init__(self, *, banned_checker: Callable[[list[str]], Awaitable[bool]]):
        self.banned_checker = banned_checker

    async def run(self, ctx: Context) -> CheckResult:
        current_uuids = ctx.gpu_uuids or ""

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
        is_banned = await self.banned_checker(uuids)

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
