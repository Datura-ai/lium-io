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
reserve, the room `lium gpu-splitting setup` needs for its xfs image file) fails (score 0, STORAGE_LIMIT_REQUIRED) after
two conclusive failed probes in a row, until the host passes the probe; a host with less free keeps the warning only;
a rented node only gets the warning, so this check never touches a running rental.
It runs right after the scrape, before the rented short-circuit, like GpuThermalThrottleCheck.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

from core.config import settings

from ..availability import ReachSource, ReachTarget
from ..messages import StorageLimitMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# check_storage_limit_ability's (machine_scrape.py) answers when the probe gave no verdict
INCONCLUSIVE_PROBE_ERRORS = ("Test command timed out", "An unexpected error occurred")
# a node's last conclusive verdict stands until the next one (Mikhail, 26 Oct): the host's limit does not come or
# go between cycles, and a timeout is almost always our side. Refreshed every cycle; only a node unseen this long
# starts over
STORAGE_LIMIT_VERDICT_TTL_SECONDS = 7 * 24 * 3600
# conclusive "unsupported" probes in a row before STORAGE_LIMIT_REQUIRED
STORAGE_LIMIT_REQUIRED_AFTER_FAILURES = 2
# inconclusive this long in a row: a warning to the provider, score unchanged
STORAGE_LIMIT_INCONCLUSIVE_WARN_AFTER = timedelta(hours=24)


def _verdict_key(ctx: Context) -> str:
    return f"storage_limit_verdict:{ctx.miner_hotkey}:{ctx.executor.uuid}"


class StorageLimitCheck:
    """Warn about, and with REQUIRE_STORAGE_LIMIT_SUPPORTED fail, a node that cannot cap a container's disk."""

    check_id = "storage.validate.limit_supported"
    fatal = True  # only STORAGE_LIMIT_REQUIRED and STORAGE_LIMIT_INCONCLUSIVE fail; every other result passes

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs or {}
        if specs.get("storage_limit_supported") is True:
            await self._save(ctx, {"verdict": "pass", "failures": 0})
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
        # its last conclusive verdict, and the published specs keep saying so, so a whole-host rental stays capped
        conclusive = not detail.startswith(INCONCLUSIVE_PROBE_ERRORS)
        last = await self._load(ctx)
        now = datetime.now(UTC)
        if conclusive:
            state = {"verdict": "fail", "failures": int(last.get("failures") or 0) + 1}
        else:
            state = {
                **last,
                "inconclusive_since": last.get("inconclusive_since") or now.isoformat(),
            }
        await self._save(ctx, state)
        verdict = state.get("verdict")
        inconclusive_since = state.get("inconclusive_since")
        long_inconclusive = bool(
            inconclusive_since
            and now - datetime.fromisoformat(inconclusive_since)
            > STORAGE_LIMIT_INCONCLUSIVE_WARN_AFTER
        )
        if not conclusive:
            # publish the last verdict; a node that never had one is unknown, never "no limit"
            ctx.state.specs["storage_limit_supported"] = {"pass": True, "fail": False}.get(verdict)
        if verdict == "pass":
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.INCONCLUSIVE_LONG if long_inconclusive else Msg.OK,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "reused_last_pass": True,
                        "inconclusive_since": inconclusive_since,
                        "detail": detail[:300],
                    },
                ),
            )
        failures = int(state.get("failures") or 0)
        what = {
            "storage_limit_supported": False if verdict == "fail" else None,
            "capable": capable,
            "conclusive": conclusive,
            "consecutive_failures": failures,
            "inconclusive_since": inconclusive_since,
            "detail": detail[:300] or None,
            "enforced": settings.REQUIRE_STORAGE_LIMIT_SUPPORTED,
            "rented": rented,
        }
        required = failures >= STORAGE_LIMIT_REQUIRED_AFTER_FAILURES
        if (
            settings.REQUIRE_STORAGE_LIMIT_SUPPORTED
            and not rented
            and capable
            and (required or verdict is None)
        ):
            what["reach_source"] = str(ReachSource.VALIDATOR)
            what["reach_target"] = str(ReachTarget.DOCKER_STORAGE)
            return CheckResult(
                passed=False,
                event=render_message(
                    Msg.REQUIRED if required else Msg.INCONCLUSIVE,
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
            event=render_message(
                Msg.INCONCLUSIVE_LONG if long_inconclusive else Msg.MISSING,
                ctx=ctx,
                check_id=self.check_id,
                what=what,
            ),
        )

    @staticmethod
    async def _load(ctx: Context) -> dict:
        try:
            raw = await ctx.services.redis.get(_verdict_key(ctx))
            return json.loads(raw) if raw else {}
        except Exception as e:  # noqa: BLE001 — without the memory the node counts as never verdicted (hold)
            logger.warning("storage limit verdict not read for %s: %s", ctx.executor.uuid, e)
            return {}

    @staticmethod
    async def _save(ctx: Context, state: dict) -> None:
        try:
            await ctx.services.redis.set(
                _verdict_key(ctx), json.dumps(state), ex=STORAGE_LIMIT_VERDICT_TTL_SECONDS
            )
        except Exception as e:  # noqa: BLE001 — a Redis blip must not fail the check
            logger.warning("storage limit verdict not recorded for %s: %s", ctx.executor.uuid, e)
