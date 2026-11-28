"""A node whose lium-rentals network the rental create refuses is not listed.

The rental create runs a pod only on a lium-rentals network that is a bridge with inter-container traffic off
(rental_docker_sdk `_require_icc_off`) and refuses every rental on one that is not. No check read that network's
options, so such a node passed every cycle, was listed, failed its renter's create, was paused and listed again by the
next clean cycle. This inspects the same network and judges it with the create's own rule (`rental_network_isolated`).

A host without the network passes: the create makes it with the right options. Only an answer the host gave decides:
a read that timed out or failed for another reason passes, and the create's own inspect still refuses that rental.
A failure is an availability error, never fatal: the node is hidden and new rentals on it are refused, while the
cycle runs on to ScoreCheck, so the node keeps its score and a rental already on it runs on. The rental probe reads
`rental_network_usable` and skips its create, which the create would refuse on this network.
"""

from __future__ import annotations

import json
from dataclasses import replace

from services.rental_docker_sdk import RENTAL_NETWORK_NAME, rental_network_isolated

from ..availability import ReachSource, ReachTarget
from ..messages import RentalNetworkMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

READ_TIMEOUT_SECONDS = 15
INSPECT_COMMAND = f"/usr/bin/docker network inspect {RENTAL_NETWORK_NAME}"


def inspected_network(exit_code: int, stdout: str) -> tuple[dict | None, bool]:
    """(the network, read): no network with read=True when the host has none; read=False when the answer says nothing."""
    try:
        answer = json.loads(stdout or "null")
    except ValueError:
        return None, False
    if exit_code != 0:
        # `docker network inspect` of a missing name prints `[]` and exits 1
        return None, answer == []
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
        result = await ctx.runner.run(INSPECT_COMMAND, timeout=READ_TIMEOUT_SECONDS, retryable=True)
        network, read = (None, False) if result.error_type is not None else inspected_network(
            result.exit_code, result.stdout
        )
        if not read:
            what["read_error"] = result.error_type or f"exit {result.exit_code}"
            return CheckResult(passed=True, event=render_message(Msg.NOT_READ, ctx=ctx, check_id=self.check_id, what=what))
        if network is None or rental_network_isolated(network):
            what["present"] = network is not None
            return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what))
        what.update(
            {
                "driver": network.get("Driver"),
                "options": network.get("Options") or {},
                "reach_source": str(ReachSource.VALIDATOR),
                "reach_target": str(ReachTarget.NODE),
            }
        )
        event = render_message(Msg.UNUSABLE, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=False, event=event, updates={"state": replace(ctx.state, rental_network_usable=False)})
