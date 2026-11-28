"""A node whose lium-rentals network the rental create refuses is not listed.

The rental create runs a pod only on a lium-rentals network that is a bridge with inter-container traffic off
(rental_docker_sdk `_require_icc_off`) and refuses every rental on one that is not. No check read that network's
options, so such a node passed every cycle, was listed, failed its renter's create, was paused and listed again by the
next clean cycle. This inspects the same network and judges it with the create's own rule (`rental_network_isolated`).

A host without the network passes: the create makes it with the right options. Absence counts only on Docker's own
not-found answer; a daemon failure can also print `[]` and exit 1. Only a read decides: a refused network is recorded,
and every cycle whose read times out or fails keeps the node hidden until a read finds the network isolated or absent.
A failure is an availability error, never fatal: the node is hidden and new rentals on it are refused, while the
cycle runs on to ScoreCheck, so the node keeps its score and a rental already on it runs on. The rental probe reads
`rental_network_usable` and skips its create, which the create would refuse on this network.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace

from services.rental_docker_sdk import RENTAL_NETWORK_NAME, rental_network_isolated

from ..availability import ReachSource, ReachTarget
from ..messages import RentalNetworkMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

READ_TIMEOUT_SECONDS = 15
INSPECT_COMMAND = f"/usr/bin/docker network inspect {RENTAL_NETWORK_NAME}"
UNUSABLE_KEY_PREFIX = "rental_network_unusable"
UNUSABLE_TTL_SECONDS = 30 * 24 * 3600
# the daemon's answer for a missing network: "network <name> not found" (current) or "No such network: <name>" (older)
NOT_FOUND = re.compile(
    rf"network {re.escape(RENTAL_NETWORK_NAME)} not found|no such network: {re.escape(RENTAL_NETWORK_NAME)}", re.I
)

logger = logging.getLogger(__name__)


def inspected_network(exit_code: int, stdout: str, stderr: str = "") -> tuple[dict | None, bool]:
    """(the network, read): no network with read=True when the host has none; read=False when the answer says nothing."""
    if exit_code != 0:
        return None, bool(NOT_FOUND.search(stderr or ""))
    try:
        answer = json.loads(stdout or "null")
    except ValueError:
        return None, False
    if isinstance(answer, list) and len(answer) == 1 and isinstance(answer[0], dict):
        return answer[0], True
    return None, False


class RentalNetworkCheck:
    check_id = "executor.validate.rental_network"
    replays_after_session_drop = True
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        what: dict = {"executor_uuid": ctx.executor.uuid}
        if ctx.runner is None:
            return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what))
        key = f"{UNUSABLE_KEY_PREFIX}:{ctx.executor.uuid}"
        result = await ctx.runner.run(INSPECT_COMMAND, timeout=READ_TIMEOUT_SECONDS, retryable=True)
        network, read = (None, False) if result.error_type is not None else inspected_network(
            result.exit_code, result.stdout, result.stderr
        )
        if not read:
            what["read_error"] = result.error_type or f"exit {result.exit_code}"
            try:
                held = bool(await ctx.services.redis.get(key))
            except Exception:
                # shortcut: with Redis down the last refusal is unknown, so this cycle proceeds; the create's own
                # inspect still refuses the rental
                logger.warning("rental network refusal record unavailable", exc_info=True, extra=ctx.default_extra)
                held = False
            if not held:
                return CheckResult(
                    passed=True, event=render_message(Msg.NOT_READ, ctx=ctx, check_id=self.check_id, what=what)
                )
            what["held_from_last_read"] = True
            return self._unusable(ctx, what)
        if network is None or rental_network_isolated(network):
            try:
                await ctx.services.redis.delete(key)
            except Exception:
                logger.warning("rental network refusal record not cleared", exc_info=True, extra=ctx.default_extra)
            what["present"] = network is not None
            return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what))
        try:
            await ctx.services.redis.set(key, "1", ex=UNUSABLE_TTL_SECONDS)
        except Exception:
            logger.warning("rental network refusal not recorded", exc_info=True, extra=ctx.default_extra)
        what.update({"driver": network.get("Driver"), "options": network.get("Options") or {}})
        return self._unusable(ctx, what)

    def _unusable(self, ctx: Context, what: dict) -> CheckResult:
        what.update({"reach_source": str(ReachSource.VALIDATOR), "reach_target": str(ReachTarget.NODE)})
        event = render_message(Msg.UNUSABLE, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=False, event=event, updates={"state": replace(ctx.state, rental_network_usable=False)})
