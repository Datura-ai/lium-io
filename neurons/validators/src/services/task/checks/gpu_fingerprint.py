from __future__ import annotations

from typing import Callable

from ..models import build_msg
from ..pipeline import CheckResult, Context


class GpuFingerprintCheck:
    check_id = "gpu.validate.fingerprint"
    fatal = True

    def __init__(self, *, fingerprint_checker: Callable[[str, str], bool]):
        self.fingerprint_checker = fingerprint_checker

    async def run(self, ctx: Context) -> CheckResult:
        prev_uuids = (ctx.verified or {}).get("uuids") or ""
        current_uuids = ctx.gpu_uuids or ""

        if prev_uuids and current_uuids and self.fingerprint_checker(prev_uuids, current_uuids):
            event = build_msg(
                event="GPU fingerprints changed",
                reason="GPU_UUID_CHANGED",
                severity="warning",
                category="env",
                impact="Verification reset; score set to 0",
                remediation=(
                    "Ensure the same physical GPUs remain attached and stable. Power-cycling or PCIe "
                    "reordering can change UUID order; keep the mapping consistent."
                ),
                what={
                    "previous": prev_uuids,
                    "current": current_uuids,
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
            event="GPU fingerprints stable",
            reason="GPU_UUID_OK",
            severity="info",
            category="env",
            impact="Proceed",
            what={"gpu_uuids": current_uuids},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=True, event=event)
