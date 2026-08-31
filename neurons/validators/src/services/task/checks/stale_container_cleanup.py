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

        # DAH-2475: give the DPHN filler cache back when the node can no longer afford it. This is
        # the ONLY caller — the create-time sweep deliberately never reclaims (it would raise free
        # disk, the backend would grant the cache again next cycle, and the node would re-download
        # ~37 GB forever), so a node that has fallen under the rental listing floor is rescued from
        # here, where the decision is made on real free space and outside any launch.
        reclaimed_cache_volumes = await ctx.services.container_cleanup.reclaim_dphn_cache_when_disk_is_tight(
            ssh_client=ctx.ssh,
            executor_uuid=ctx.executor.uuid,
        )

        # DAH-2805: a filler weight download that is killed mid-flight leaves a `*.incomplete` file
        # nothing will ever read again, and one prod node reached 741 GB of them. It is swept from
        # here rather than from inside the filler image because this loop reaches the node on every
        # cycle whatever image is on it, and it cleans what is already on the fleet.
        swept_download_temporaries = await ctx.services.container_cleanup.sweep_abandoned_download_temporaries(
            ssh_client=ctx.ssh,
            executor_uuid=ctx.executor.uuid,
        )

        event = render_message(
            Msg.CLEANED,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "removed_count": removed_count,
                "removed_containers": removed_names,
                "reclaimed_cache_volumes": reclaimed_cache_volumes,
                "swept_download_temporaries": swept_download_temporaries,
            },
        )
        return CheckResult(passed=True, event=event)
