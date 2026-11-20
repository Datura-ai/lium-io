"""Running rental containers get the link-local egress drop that custom builds already have.

The rules, their scripts and the host inventory they follow are one source shared with the executor
(``datura.rental_egress``). An executor that advertises ``RENTAL_EGRESS_CAPABILITY`` on ``/version`` applies them
itself, at start, when the rental network is created and when a rental container starts, so they hold after a
reboot and whatever this validator can reach. For it this check only reads them back (it never writes, so it also
runs while host writes wait), and a host that runs a rental network or a legacy rental without the drop gets the
``RENTAL_EGRESS_RULES_MISSING`` availability error: no new rental is placed there, a running one is kept. An
executor that does not advertise it (an older release, or ``/version`` unreachable) gets the rules applied over SSH
as before, so no node is ever hidden for running an older executor.

- Never before a scored step: on a rented node TenantEnforcementCheck runs it in its rented halt
  branch, so only the halt waits for it; on an idle node it runs after ScoreCheck, where a read that finds the
  rules missing is one of the run's events, and again when the pipeline ends (an older executor's rules then
  never wait behind a fatal check). A first helper run is bounded by ``HELPER_TIMEOUT_SECONDS``, so it is skipped
  (and retried next cycle) when less than that is left before the executor task is cancelled.
- Periodic: at most once per ``APPLY_INTERVAL_SECONDS`` per node, except that rules found missing are read again
  every cycle until they are back, so the availability error stands only while they are missing.
- Fail-open: a host without a DOCKER-USER chain or without iptables logs a warning and
  validation proceeds; a hardening rule must not take a provider's node out of rotation.
"""

from __future__ import annotations

import asyncio
import logging
import time

from datura.rental_egress import (  # noqa: F401 — re-exported: the check's tests reach the shared helpers here
    APPLY_INTERVAL_SECONDS,
    CHAINS,
    HELPER_TIMEOUT_SECONDS,
    HOST_LOCK,
    HOST_NS,
    LINK_LOCAL_CIDR,
    RENTAL_EGRESS_CAPABILITY,
    InventoryUnknown,
    _helper,
    apply_rules,
    helper_command,
    legacy_bridge_inventory,
    legacy_network_ids,
    link_local_cleanup_script,
    link_local_drop_script,
    locked_script,
    rental_bridge,
    verify_rules,
)
from services.local_verify_client import LocalVerifyClient

from core.config import settings
from core.utils import _m, get_extra_info

from ..availability import ReachSource, ReachTarget
from ..messages import RentalEgressGuardMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node
from .task_time import task_seconds_left

logger = logging.getLogger(__name__)

# the inspect read and FinalizeCheck after the helper
BUDGET_MARGIN_SECONDS = 30


async def advertised_capabilities(ctx: Context) -> set[str]:
    """What the executor's ``/version`` lists; empty on any error, which keeps the SSH write path."""
    client = LocalVerifyClient(
        None,
        timeout_s=settings.LOCAL_VERIFY_TIMEOUT_SECONDS,
        connect_timeout_s=settings.LOCAL_VERIFY_CONNECT_TIMEOUT_SECONDS,
    )
    return (await client.advertised(ctx.executor)).capabilities


class RentalEgressGuardCheck:
    """Keeps the rental network's link-local egress drop in place, at most once per interval per executor: reads it
    back on an executor that applies it itself, applies it over SSH on one that does not."""

    check_id = "executor.network.rental_egress_guard"
    fatal = False

    def __init__(
        self,
        interval_seconds: int = APPLY_INTERVAL_SECONDS,
        *,
        rented_only: bool = False,
        on_completion: bool | None = None,
        last_run: dict[tuple, float] | None = None,
    ) -> None:
        self._interval = interval_seconds
        # rented_only: the instance before TenantEnforcementCheck, rented nodes only. The completion instance runs
        # when the pipeline ends, whichever way, for every node: neither a running rental after a reboot nor the
        # cleanup after the last legacy rental may wait behind a fatal check that ends every cycle early. It only
        # writes: a read there is no event of the run, so the idle read is the in-run instance after ScoreCheck
        # (on_completion=False), which only reads.
        self._rented_only = rented_only
        self.on_completion = (not rented_only) if on_completion is None else on_completion
        # shared by every instance so a node is guarded once per interval whichever one it reaches; keyed by
        # the miner and the endpoint too, since two miners can report one UUID for different hosts
        self._last_run: dict[tuple, float] = {} if last_run is None else last_run

    def _skip(self, ctx: Context, reason: str) -> CheckResult:
        return CheckResult(
            passed=True,
            event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={"reason": reason}),
        )

    def _not_a_run(self, key: tuple, last: float | None) -> None:
        """Undo this cycle's cadence stamp, so the next cycle tries again."""
        if last is None:
            self._last_run.pop(key, None)
        else:
            self._last_run[key] = last

    async def run(self, ctx: Context) -> CheckResult:
        executor_uuid = ctx.executor.uuid
        key = (ctx.miner_hotkey, executor_uuid, ctx.executor.address, ctx.executor.ssh_port)
        rental = rental_of_this_node(ctx)
        if self._rented_only and not (rental and rental.pods):
            return self._skip(ctx, "guarded at the other position")
        if task_seconds_left(ctx) < HELPER_TIMEOUT_SECONDS + BUDGET_MARGIN_SECONDS:
            return self._skip(ctx, "not enough task time left")
        now = time.monotonic()
        last = self._last_run.get(key)
        if last is not None and now - last < self._interval:
            return self._skip(ctx, "within cadence window")
        # Retry on the next interval whatever happens below, never every cycle.
        self._last_run[key] = now

        reads_only = False
        try:
            # Every remote call shares one deadline that ends BUDGET_MARGIN_SECONDS before the task is
            # cancelled, so a stalled SSH call can never cost the cycle its validation result.
            async with asyncio.timeout(task_seconds_left(ctx) - BUDGET_MARGIN_SECONDS):
                reads_only = RENTAL_EGRESS_CAPABILITY in await advertised_capabilities(ctx)
                if reads_only and self.on_completion:
                    self._not_a_run(key, last)
                    return self._skip(ctx, "applied by the executor; read in the run")
                if not reads_only and not (self.on_completion or self._rented_only):
                    # an older executor's idle rules are written when the run ends, as before
                    self._not_a_run(key, last)
                    return self._skip(ctx, "written when the run ends")
                run = self._runner(ctx)
                guarded, error = await (verify_rules(run) if reads_only else apply_rules(run))
        except InventoryUnknown as exc:
            # Rules stay as they are: an unread host is never taken for an empty one. Not a run.
            self._not_a_run(key, last)
            logger.warning(
                _m(
                    "Rental egress guard: host inventory unknown, rules left as they are",
                    extra=get_extra_info({"executor_uuid": executor_uuid, "error": str(exc)}),
                )
            )
            return self._skip(ctx, "inventory unknown")
        except TimeoutError:
            # Not a run: the next cycle tries again.
            self._not_a_run(key, last)
            return self._skip(ctx, "out of task time")
        except Exception as exc:  # noqa: BLE001 — fail-open, see module docstring
            guarded, error = {}, str(exc)
        if reads_only:
            return self._read_result(ctx, key, last, guarded, error)
        if guarded is None:
            # No rental network and no legacy rental: nothing to guard; the next interval checks again.
            if error is None:
                return self._skip(ctx, "no rental network")
            # A failed cleanup may have left stale rules: not a run, so the next cycle retries.
            self._not_a_run(key, last)
            logger.warning(
                _m(
                    "Rental egress guard: stale rule cleanup failed",
                    extra=get_extra_info({"executor_uuid": executor_uuid, "error": error}),
                )
            )
            return self._skip(ctx, "cleanup failed")

        if error is not None:
            logger.warning(
                _m(
                    "Rental egress guard: link-local drop not applied",
                    extra=get_extra_info({"executor_uuid": executor_uuid, "error": error}),
                )
            )
            return CheckResult(
                passed=True,
                event=render_message(Msg.NOT_APPLIED, ctx=ctx, check_id=self.check_id, what={"error": error}),
            )
        return CheckResult(
            passed=True,
            event=render_message(Msg.APPLIED, ctx=ctx, check_id=self.check_id, what=guarded),
        )

    def _read_result(
        self, ctx: Context, key: tuple, last: float | None, read: dict | None, error: str | None
    ) -> CheckResult:
        if error is not None:
            # fail-open like a failed apply: the rules could not be read (no DOCKER-USER chain included)
            logger.warning(
                _m(
                    "Rental egress guard: rules not read",
                    extra=get_extra_info({"executor_uuid": ctx.executor.uuid, "error": error}),
                )
            )
            return CheckResult(
                passed=True,
                event=render_message(Msg.NOT_READ, ctx=ctx, check_id=self.check_id, what={"error": error}),
            )
        if read is None:
            return self._skip(ctx, "no rental network")
        if read["missing"]:
            # read again every cycle until they are back: the error must stand on each cycle they are missing
            self._not_a_run(key, last)
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.MISSING,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        **read,
                        "reach_source": str(ReachSource.VALIDATOR),
                        "reach_target": str(ReachTarget.NODE),
                    },
                ),
            )
        return CheckResult(
            passed=True,
            event=render_message(Msg.PRESENT, ctx=ctx, check_id=self.check_id, what=read),
        )

    @staticmethod
    def _runner(ctx: Context):
        async def run(command: str):
            return await ctx.ssh.run(command, check=False)

        return run
