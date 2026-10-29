"""STORAGE_LIMIT: a node whose Docker cannot cap a container's disk.

Without a per-container storage limit (``docker run --storage-opt size=...``, overlay2 on xfs with ``pquota``) a
renter who fills the container's disk fills the host's ``/var/lib/docker``: the host and every pod on it go down
(18 Oct 2026, serdkey node 4837fec9). The scrape already records the probe's verdict as
``specs.storage_limit_supported``; nothing acted on it.

Pure data, no SSH. While ``REQUIRE_STORAGE_LIMIT_SUPPORTED`` is off every node without the limit gets the
non-fatal STORAGE_LIMIT_MISSING warning, score unchanged, so providers see it in their logs before enforcement.
The limit counts only for overlay2 on xfs mounted with pquota whose ``--storage-opt size=`` container runs (the scrape).
With the setting on, a node that is not rented and whose host could build the limit (the scrape's
``storage_limit_capable``: a local filesystem other than docker's own mount with 100 GiB left after the system's
reserve, the room `lium gpu-splitting setup` needs for its xfs image file) is held from new rentals, still paid, after
one conclusive failed probe, and fails (score 0, STORAGE_LIMIT_REQUIRED) after two in a row, until the host passes the
probe. A probe without a verdict neither counts nor resets: the last verdict stands, and over 24 h without one holds.
No verdict on record (new, expired or unreadable memory) holds, never zeros. A host with less free keeps the warning
only; an unknown capability adds no consequence and clears none; a rented node only gets the warning, so this check
never touches a running rental.
It runs right after the scrape, before the rented short-circuit, like GpuThermalThrottleCheck.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

from core.config import settings

from ..availability import ReachSource, ReachTarget
from ..messages import AVAILABILITY_CATEGORY
from ..messages import StorageLimitMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# a node's last conclusive verdict stands until the next one (Mikhail, 26 Oct): the host's limit does not come or
# go between cycles, and a timeout is almost always our side. Refreshed every cycle; only a node unseen this long
# starts over
STORAGE_LIMIT_VERDICT_TTL_SECONDS = 7 * 24 * 3600
# conclusive "unsupported" probes in a row before STORAGE_LIMIT_REQUIRED
STORAGE_LIMIT_REQUIRED_AFTER_FAILURES = 2
# no verdict this long in a row: the node is held from new rentals, score unchanged
STORAGE_LIMIT_INCONCLUSIVE_WARN_AFTER = timedelta(hours=24)
# what the check does to an unrented node with enforcement on (Mikhail, 29 Oct)
WARN, HOLD, ZERO = 0, 1, 2  # warning only; no new rentals, still paid; score 0 and no new rentals
HOLD_IMPACT = "Score not changed; the node is hidden from new rentals until a disk-limit test passes"


def _verdict_key(ctx: Context) -> str:
    return f"storage_limit_verdict:{ctx.miner_hotkey}:{ctx.executor.uuid}"


def _long_inconclusive(state: dict, now: datetime) -> bool:
    since = state.get("inconclusive_since")
    return bool(since and now - datetime.fromisoformat(since) > STORAGE_LIMIT_INCONCLUSIVE_WARN_AFTER)


def _consequence(state: dict, now: datetime) -> int:
    """What a stored verdict causes: a failure holds, two in a row zero; no verdict for over a day holds."""
    if state.get("verdict") == "fail":
        failures = int(state.get("failures") or 0)
        return ZERO if failures >= STORAGE_LIMIT_REQUIRED_AFTER_FAILURES else HOLD
    return HOLD if _long_inconclusive(state, now) else WARN


class StorageLimitCheck:
    """Warn about, and with REQUIRE_STORAGE_LIMIT_SUPPORTED hold or fail, a node that cannot cap a container's disk."""

    check_id = "storage.validate.limit_supported"
    fatal = True  # only STORAGE_LIMIT_REQUIRED fails; a hold passes with an availability event

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs or {}
        # the scrape's verdict: True / False, None (or missing) when the probe gave none
        supported = specs.get("storage_limit_supported")
        if supported is True:
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
        # DAH-1997: a host without room for setup's image file keeps the warning; None (an older scrape, a failed
        # reading) neither adds a consequence nor clears one
        capable = specs.get("storage_limit_capable")
        detail = str(specs.get("storage_limit_scrape_error") or "")
        conclusive = supported is False
        last = await self._load(ctx)
        now = datetime.now(UTC)
        if last is None:
            state = None  # verdict memory unreadable: hold, never a zero
        elif conclusive:
            previous = int(last.get("failures") or 0) if last.get("verdict") == "fail" else 0
            state = {"verdict": "fail", "failures": previous + 1}
        else:
            # no verdict neither counts nor resets: the last one stands
            state = {**last, "inconclusive_since": last.get("inconclusive_since") or now.isoformat()}
        verdict = (state or {}).get("verdict")
        # no verdict on record (never probed, expired, unreadable): hold until a conclusive one
        consequence = _consequence(state, now) if verdict else HOLD
        if capable is None:
            # stored before the rented / enforcement gates, so a rental or the flag clears nothing here
            consequence = min(consequence, int((last or {}).get("applied") or WARN))
        elif capable is False:
            consequence = WARN
        saved = state is not None and await self._save(ctx, {**state, "applied": consequence})
        if not saved and capable:
            consequence = max(consequence, HOLD)
        if not settings.REQUIRE_STORAGE_LIMIT_SUPPORTED or rented:
            consequence = WARN
        state = state or {}
        long_inconclusive = _long_inconclusive(state, now)
        inconclusive_since = state.get("inconclusive_since")
        if not conclusive:
            # publish the last verdict; a node that never had one is unknown, never "no limit"
            ctx.state.specs["storage_limit_supported"] = {"pass": True, "fail": False}.get(verdict)
        what = {
            "storage_limit_supported": {"pass": True, "fail": False}.get(verdict),
            "capable": capable,
            "conclusive": conclusive,
            "consecutive_failures": int(state.get("failures") or 0),
            "inconclusive_since": inconclusive_since,
            "detail": detail[:300] or None,
            "enforced": settings.REQUIRE_STORAGE_LIMIT_SUPPORTED,
            "rented": rented,
        }
        if verdict == "pass":
            what["reused_last_pass"] = True
        if consequence == WARN:
            template = Msg.INCONCLUSIVE_LONG if long_inconclusive else Msg.OK if verdict == "pass" else Msg.MISSING
            return CheckResult(
                passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
            )
        what["reach_source"] = str(ReachSource.VALIDATOR)
        what["reach_target"] = str(ReachTarget.DOCKER_STORAGE)
        if consequence == ZERO:
            return CheckResult(
                passed=False,
                event=render_message(Msg.REQUIRED, ctx=ctx, check_id=self.check_id, what=what),
                updates={
                    "score": 0.0,
                    "job_score": 0.0,
                    "score_warning": " WARNING: Docker cannot limit a container's disk on this node",
                },
            )
        template = Msg.INCONCLUSIVE_LONG if long_inconclusive else Msg.MISSING if verdict else Msg.INCONCLUSIVE
        event = render_message(
            template,
            ctx=ctx,
            check_id=self.check_id,
            what=what,
            category=AVAILABILITY_CATEGORY,
            impact=HOLD_IMPACT,
        )
        return CheckResult(passed=True, event=event)

    @staticmethod
    async def _load(ctx: Context) -> dict | None:
        """The node's stored verdict ({} when none); None when Redis could not be read."""
        try:
            raw = await ctx.services.redis.get(_verdict_key(ctx))
            return json.loads(raw) if raw else {}
        except Exception as e:  # noqa: BLE001 — the caller holds the node
            logger.warning("storage limit verdict not read for %s: %s", ctx.executor.uuid, e)
            return None

    @staticmethod
    async def _save(ctx: Context, state: dict) -> bool:
        try:
            await ctx.services.redis.set(
                _verdict_key(ctx), json.dumps(state), ex=STORAGE_LIMIT_VERDICT_TTL_SECONDS
            )
            return True
        except Exception as e:  # noqa: BLE001 — the caller holds the node (a pass needs no memory)
            logger.warning("storage limit verdict not recorded for %s: %s", ctx.executor.uuid, e)
            return False
