from __future__ import annotations

from ..models import build_msg
from ..pipeline import CheckResult, Context


class GpuFingerprintCheck:
    """Compare stored GPU UUIDs with the latest scrape to detect hardware swaps.

    This retains the old `check_fingerprints_changed` guard, catching cases where miners
    cycle hardware or reorder devices to dodge bans. Deviations trigger a verification
    reset just like the legacy flow.
    """

    check_id = "gpu.validate.fingerprint"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        prev_uuids = (ctx.verified or {}).get("uuids") or ""
        current_uuids = ctx.gpu_uuids or ""

        if prev_uuids and current_uuids:
            prev_sorted = sorted(u for u in prev_uuids.split(",") if u)
            current_sorted = sorted(u for u in current_uuids.split(",") if u)

            if prev_sorted != current_sorted:
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
