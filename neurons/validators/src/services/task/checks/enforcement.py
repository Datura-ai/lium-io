"""Report first, enforce after the notice (DAH-6539 follow-up, Mikhail's 19 Nov incident review).

The 17 Nov account check hid nodes from the moment it merged, so a provider whose notice was held heard nothing before
losing idle pay. A check whose failure is new to the fleet therefore only reports it at first: the run passes, and the
reason code goes into the final event (`reported_only`). The portal mails that node's provider once and then sends
`check_enforce_after[reason_code]`; from that time the same failure refuses the node. Direct evidence of an attack (a
machine set up for another owner's account) never waits for this.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..pipeline import CheckResult, Context


def fresh(until: str | None) -> bool:
    """Whether the portal read the miner sent this node from still vouches for it (`snapshot_valid_until`): False once
    the time passed or when it does not parse (a time without a zone included)."""
    try:
        return datetime.now(UTC) < datetime.fromisoformat(str(until))
    except (TypeError, ValueError):
        return False


def enforced(ctx: Context, reason_code: str) -> bool:
    """True once the portal says this node's provider was notified of `reason_code` and its time has come. A miner or
    portal that sends no map at all predates this rule: its nodes are enforced as before, so shipping the validator
    first never lets a stalled read pass. A map from a read that no longer vouches for the node (a snapshot the miner
    kept serving through a portal outage) may be an earlier registration's of the same uuid: the finding only reports."""
    if ctx.executor.check_enforce_after is None:
        return True
    if not fresh(ctx.executor.snapshot_valid_until):
        return False
    after = ctx.executor.check_enforce_after.get(reason_code)
    if not after:
        return False
    try:
        at = datetime.fromisoformat(after.replace("Z", "+00:00"))
    except ValueError:
        return False
    return datetime.now(UTC) >= (at if at.tzinfo else at.replace(tzinfo=UTC))


def reported_only(ctx: Context, event) -> CheckResult:
    """The run passes; the failure is kept for the final event, from which the portal sends the notice."""
    # shortcut: a plain field update, so two parallel checks reporting in one run would keep only one code; only one
    # check reports today, make it a merged update when a second one does
    return CheckResult(
        passed=True, event=event, updates={"reported_only": (*ctx.reported_only, event.reason_code)}
    )
