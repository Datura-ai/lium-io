"""STORAGE_LIMIT: a node whose Docker cannot cap a container's disk.

Without a per-container storage limit (``docker run --storage-opt size=...``, overlay2 on xfs with ``pquota``) a
renter who fills the container's disk fills the host's ``/var/lib/docker``: the host and every pod on it go down
(18 Oct 2026, serdkey node 4837fec9). The scrape already records the probe's verdict as
``specs.storage_limit_supported``; nothing acted on it.

Pure data, no SSH. A node without the limit that is not rented fails (score 0, STORAGE_LIMIT_REQUIRED) until
the host passes the probe; a rented node only gets the non-fatal STORAGE_LIMIT_MISSING warning, so this check
never touches a running rental.
It runs right after the scrape, before the rented short-circuit, like GpuThermalThrottleCheck.
"""

from __future__ import annotations

from ..messages import StorageLimitMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context


class StorageLimitCheck:
    """Fail an unrented node, and warn about a rented one, whose Docker cannot cap a container's disk."""

    check_id = "storage.validate.limit_supported"
    fatal = True  # only the STORAGE_LIMIT_REQUIRED result fails; every other result passes

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs or {}
        if specs.get("storage_limit_supported") is True:
            return CheckResult(
                passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id)
            )

        rented_executor = (
            ctx.state.rented_data.executors.get(ctx.executor.uuid)
            if ctx.state.rented_data
            else None
        )
        rented = bool(rented_executor and rented_executor.pods)
        what = {
            "storage_limit_supported": False,
            "detail": str(specs.get("storage_limit_scrape_error") or "")[:300] or None,
            "rented": rented,
        }
        if not rented:
            return CheckResult(
                passed=False,
                event=render_message(Msg.REQUIRED, ctx=ctx, check_id=self.check_id, what=what),
                updates={
                    "score": 0.0,
                    "job_score": 0.0,
                    "score_warning": " WARNING: Docker cannot limit a container's disk on this node",
                },
            )
        return CheckResult(
            passed=True,
            event=render_message(Msg.MISSING, ctx=ctx, check_id=self.check_id, what=what),
        )
