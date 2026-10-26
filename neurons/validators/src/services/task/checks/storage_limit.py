"""STORAGE_LIMIT: a node whose Docker cannot cap a container's disk.

Without a per-container storage limit (``docker run --storage-opt size=...``, overlay2 on xfs with ``pquota``) a
renter who fills the container's disk fills the host's ``/var/lib/docker``: the host and every pod on it go down
(18 Oct 2026, serdkey node 4837fec9). The scrape already records the probe's verdict as
``specs.storage_limit_supported``; nothing acted on it.

Pure data, no SSH. While ``REQUIRE_STORAGE_LIMIT_SUPPORTED`` is off every node without the limit gets the
non-fatal STORAGE_LIMIT_MISSING warning, score unchanged, so providers see it in their logs before enforcement.
The limit counts only for overlay2 on xfs mounted with pquota whose ``--storage-opt size=`` container runs (the scrape).
With the setting on, a node without the limit that is not rented and whose host could build it (the scrape's
``storage_limit_capable``: a local filesystem other than docker's own mount with 100 GiB left after the system's
reserve, the room `lium gpu-splitting setup` needs for its xfs image file) fails (score 0, STORAGE_LIMIT_REQUIRED) until the host passes the probe; a host
with less free keeps the warning only; a rented node only gets the warning, so this check never touches a running rental.
It runs right after the scrape, before the rented short-circuit, like GpuThermalThrottleCheck.
"""

from __future__ import annotations

import logging

from core.config import settings

from ..availability import ReachSource, ReachTarget
from ..messages import StorageLimitMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# check_storage_limit_ability's (machine_scrape.py) answers when the probe gave no verdict
INCONCLUSIVE_PROBE_ERRORS = ("Test command timed out", "An unexpected error occurred")
# a probe without a verdict reuses the node's last pass only while it is this fresh: the timeout string is
# provider-reported, so a host that dropped the limit and stalls the probe gets at most this long on the old pass
STORAGE_LIMIT_PASS_REUSE_SECONDS = 3600


def _pass_key(ctx: Context) -> str:
    return f"storage_limit_verified:{ctx.miner_hotkey}:{ctx.executor.uuid}"


class StorageLimitCheck:
    """Warn about, and with REQUIRE_STORAGE_LIMIT_SUPPORTED fail, a node that cannot cap a container's disk."""

    check_id = "storage.validate.limit_supported"
    fatal = True  # only STORAGE_LIMIT_REQUIRED and STORAGE_LIMIT_INCONCLUSIVE fail; every other result passes

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs or {}
        if specs.get("storage_limit_supported") is True:
            await self._remember(ctx, passed=True)
            return CheckResult(
                passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id)
            )

        rented_executor = (
            ctx.state.rented_data.executors.get(ctx.executor.uuid)
            if ctx.state.rented_data
            else None
        )
        rented = bool(rented_executor and rented_executor.pods)
        # DAH-1997: a host without room for setup's image file keeps the warning; unknown (an older
        # scrape, a failed reading) counts as able
        capable = specs.get("storage_limit_capable") is not False
        detail = str(specs.get("storage_limit_scrape_error") or "")
        # a probe that timed out or crashed (slow image pull, busy host) says nothing about the limit: the node keeps
        # its last pass if that is fresh, and the published specs keep saying so, so a whole-host rental stays capped
        conclusive = not detail.startswith(INCONCLUSIVE_PROBE_ERRORS)
        if conclusive:
            await self._remember(ctx, passed=False)
        elif await self._recent_pass(ctx):
            ctx.state.specs["storage_limit_supported"] = True
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.OK,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={"reused_last_pass": True, "detail": detail[:300]},
                ),
            )
        else:
            # no verdict and no recent pass: the published specs say unknown, never "no limit"
            ctx.state.specs["storage_limit_supported"] = None
        what = {
            "storage_limit_supported": False if conclusive else None,
            "capable": capable,
            "conclusive": conclusive,
            "detail": detail[:300] or None,
            "enforced": settings.REQUIRE_STORAGE_LIMIT_SUPPORTED,
            "rented": rented,
        }
        if settings.REQUIRE_STORAGE_LIMIT_SUPPORTED and not rented and capable:
            what["reach_source"] = str(ReachSource.VALIDATOR)
            what["reach_target"] = str(ReachTarget.DOCKER_STORAGE)
            return CheckResult(
                passed=False,
                event=render_message(
                    Msg.REQUIRED if conclusive else Msg.INCONCLUSIVE,
                    ctx=ctx,
                    check_id=self.check_id,
                    what=what,
                ),
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

    @staticmethod
    async def _remember(ctx: Context, *, passed: bool) -> None:
        try:
            if passed:
                await ctx.services.redis.set(
                    _pass_key(ctx), "1", ex=STORAGE_LIMIT_PASS_REUSE_SECONDS
                )
            else:
                await ctx.services.redis.delete(_pass_key(ctx))
        except Exception as e:  # noqa: BLE001 — a Redis blip must not fail the check; no reuse follows from it
            logger.warning("storage limit pass not recorded for %s: %s", ctx.executor.uuid, e)

    @staticmethod
    async def _recent_pass(ctx: Context) -> bool:
        try:
            return await ctx.services.redis.get(_pass_key(ctx)) is not None
        except Exception as e:  # noqa: BLE001 — without the memory the probe counts as a failed one (hold)
            logger.warning("storage limit pass not read for %s: %s", ctx.executor.uuid, e)
            return False
