from __future__ import annotations

from ..messages import StaleContainerCleanupMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context


class StaleContainerCleanupCheck:
    """Reap orphaned rental containers *before* port verification.

    DAH-2164 follow-up. When a pod container outlives its rental — most clearly when a pod
    is marked ``BROKEN_BY_PROVIDER`` and the platform intentionally does NOT tear the
    container down — the orphan keeps binding the rental port range (e.g. 40000-40009).
    The backend then reports the executor as NOT rented, so:

      * ``PortConnectivityCheck`` deploys its DinD probe and Docker rejects the bind
        ("port is already allocated") -> 0 working ports, and
      * ``PortCountCheck`` (``fatal=True``) fails on ``not is_rented and
        port_count < MIN_PORT_COUNT`` and the pipeline runner halts there.

    The only place that used to run the stale-container cleanup was
    ``TenantEnforcementCheck`` (``executor.validate.rented_state``), which sits *after*
    those port checks. So the cleanup was never reached, the orphan was never removed, and
    the executor was stuck at score 0 every cycle with no path to self-heal.

    Running the cleanup here — ahead of the port checks — breaks that deadlock: orphaned
    (non-rented) containers older than the grace window are force-removed, freeing the
    ports so the very next port probe in the same cycle can bind them. Containers that are
    present in ``rented_data`` (real tenants + the filler) are never touched. The cleanup is
    best-effort and never fatal: a failure here must not change the executor's verdict.
    """

    check_id = "executor.cleanup.stale_containers"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        removed_count, removed_names = await ctx.services.container_cleanup.cleanup(
            ssh_client=ctx.ssh,
            rented_data=ctx.state.rented_data,
            executor_uuid=ctx.executor.uuid,
        )

        event = render_message(
            Msg.CLEANED,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "removed_count": removed_count,
                "removed_containers": removed_names,
            },
        )
        return CheckResult(passed=True, event=event)
