from __future__ import annotations

from ..messages import GpuFingerprintMessages as Msg, render_message
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
        current_uuids = ctx.state.gpu_uuids or ""

        if prev_uuids and current_uuids:
            prev_sorted = sorted(u for u in prev_uuids.split(",") if u)
            current_sorted = sorted(u for u in current_uuids.split(",") if u)

            if prev_sorted != current_sorted:
                event = render_message(
                    Msg.UUID_CHANGED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "previous": prev_uuids,
                        "current": current_uuids,
                    },
                )
                return CheckResult(
                    passed=False,
                    event=event,
                    updates={"clear_verified_job_info": True},
                )

        event = render_message(
            Msg.UUID_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"gpu_uuids": current_uuids},
        )
        return CheckResult(passed=True, event=event)
