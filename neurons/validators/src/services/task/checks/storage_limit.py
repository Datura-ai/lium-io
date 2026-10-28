"""STORAGE_LIMIT: a node whose Docker cannot cap a container's disk.

Without a per-container storage limit (``docker run --storage-opt size=...``, overlay2 on xfs with ``pquota``) a
renter who fills the container's disk fills the host's ``/var/lib/docker``: the host and every pod on it go down
(18 Oct 2026, serdkey node 4837fec9). The scrape already records the probe's verdict as
``specs.storage_limit_supported``; nothing acted on it.

Pure data, no SSH. While ``REQUIRE_STORAGE_LIMIT_SUPPORTED`` is off every node without the limit gets the
non-fatal STORAGE_LIMIT_MISSING warning, score unchanged, so providers see it in their logs before enforcement.
With the setting on, a node without the limit that is not rented fails (score 0, STORAGE_LIMIT_REQUIRED) until
the host passes the probe; a rented node only gets the warning, so this check never touches a running rental.
It runs right after the scrape, before the rented short-circuit, like GpuThermalThrottleCheck.
"""

from __future__ import annotations

from core.config import settings

from ..messages import StorageLimitMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context


class StorageLimitCheck:
    """Warn about, and with REQUIRE_STORAGE_LIMIT_SUPPORTED fail, a node that cannot cap a container's disk."""

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
            "enforced": settings.REQUIRE_STORAGE_LIMIT_SUPPORTED,
            "rented": rented,
        }
        if settings.REQUIRE_STORAGE_LIMIT_SUPPORTED and not rented:
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
