"""STORAGE_LIMIT: a node whose Docker cannot cap a container's disk.

Without a per-container storage limit (``docker run --storage-opt size=...``, overlay2 on xfs with ``pquota``) a
renter who fills the container's disk fills the host's ``/var/lib/docker``: the host and every pod on it go down
(18 Oct 2026, serdkey node 4837fec9). The scrape already records the probe's verdict as
``specs.storage_limit_supported``; nothing acted on it.

Pure data, no SSH. While ``REQUIRE_STORAGE_LIMIT_SUPPORTED`` is off every node without the limit gets the
non-fatal STORAGE_LIMIT_MISSING warning, score unchanged, so providers see it in their logs before enforcement.
With the setting on, a node without the limit that is not rented and whose host could build it (the scrape's
``storage_limit_capable``: docker's root already on xfs, or a blank whole disk of 100 GiB+ that passes `lium gpu-splitting setup`'s target checks; setup needs docker's default
/var/lib/docker, otherwise the manual steps at docs.lium.io/providers/nodes/docker-storage) fails (score 0,
STORAGE_LIMIT_REQUIRED) until the host passes the probe; a single-disk host keeps the warning only; a rented node only gets the warning, so this check never touches a running rental.
It runs right after the scrape, before the rented short-circuit, like GpuThermalThrottleCheck.
"""

from __future__ import annotations

from core.config import settings

from ..availability import ReachSource, ReachTarget
from ..messages import StorageLimitMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context


# check_storage_limit_ability's (machine_scrape.py) answers when the probe gave no verdict
INCONCLUSIVE_PROBE_ERRORS = ("Test command timed out", "An unexpected error occurred")


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
        # DAH-1997: a host the scrape saw no way to build the limit on (one disk, all in use) keeps the
        # warning; unknown (an older scrape, a failed reading) counts as able
        capable = specs.get("storage_limit_capable") is not False
        detail = str(specs.get("storage_limit_scrape_error") or "")
        # a probe that timed out or crashed (slow image pull, busy host) says nothing about the limit: warn only
        conclusive = not detail.startswith(INCONCLUSIVE_PROBE_ERRORS)
        what = {
            "storage_limit_supported": False,
            "capable": capable,
            "conclusive": conclusive,
            "detail": detail[:300] or None,
            "enforced": settings.REQUIRE_STORAGE_LIMIT_SUPPORTED,
            "rented": rented,
        }
        if settings.REQUIRE_STORAGE_LIMIT_SUPPORTED and not rented and capable and conclusive:
            what["reach_source"] = str(ReachSource.VALIDATOR)
            what["reach_target"] = str(ReachTarget.DOCKER_STORAGE)
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
