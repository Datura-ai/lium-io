from __future__ import annotations

from dataclasses import replace

from services.const import DEFAULT_JOB_OWNER_LIUM, MIN_PORT_COUNT

from ..messages import PortCountMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class PortCountCheck:
    """Verify minimum port availability and record port count for scoring.

    Reads the verified port count from ctx.state (set by PortConnectivityCheck)
    instead of querying the database.
    """

    check_id = "executor.validate.port_count"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        port_count = ctx.state.verified_port_count

        # Check if executor is rented (same pattern as rented_machine.py)
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        is_rented = rented_executor is not None and len(rented_executor.pods) > 0
        # Only the pass/fail floor reads this; `available_port_count` below stays the answered count.
        background_job_port_count = (
            0 if is_rented else self._ports_held_by_platform_background_jobs(ctx)
        )

        updated_state = replace(
            ctx.state,
            specs={
                **ctx.state.specs,
                "available_port_count": port_count,
                "port_range": ctx.executor.port_range,
                "port_mappings": ctx.executor.port_mappings,
            },
        )

        if not is_rented and port_count + background_job_port_count < MIN_PORT_COUNT:
            # DAH-2991: when the shortfall is our own leftover — an orphaned rental container the
            # cleanup could not remove — say so, instead of a bare count (ticket-0287 diagnosed it by hand).
            orphaned = ctx.state.orphaned_containers
            event = render_message(
                Msg.INSUFFICIENT_PORTS,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "available_port_count": port_count,
                    "required": MIN_PORT_COUNT,
                    "held_by_orphaned_containers": orphaned,
                    "held_by_preemptible_background_jobs": background_job_port_count,
                },
                remediation=(
                    f"Ports are held by orphaned rental container(s) {', '.join(orphaned)} that the validator "
                    "could not remove (docker could not kill the process); no action on the port range is needed — "
                    "the validator retries every cycle; a host reboot frees them at once."
                )
                if orphaned
                else None,
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"port_count": port_count, "state": updated_state},
            )
        event = render_message(
            Msg.PORT_COUNT_RECORDED,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "available_port_count": port_count,
                "held_by_preemptible_background_jobs": background_job_port_count,
            },
        )

        return CheckResult(
            passed=True,
            event=event,
            updates={"port_count": port_count, "state": updated_state},
        )

    @staticmethod
    def _ports_held_by_platform_background_jobs(ctx: Context) -> int:
        """Ports held by the platform's preemptible background jobs on this executor, not already answered.

        A customer rent preempts these jobs and takes their ports, so they count toward the floor.
        The backend's filler-port list also carries a miner's default job, which does not count; the
        executor's default-job owner tells the two apart, and anything but "lium" counts nothing.
        """
        rented_data = ctx.state.rented_data
        if rented_data is None:
            return 0
        if rented_data.get_default_job_owner(ctx.executor.uuid) != DEFAULT_JOB_OWNER_LIUM:
            return 0
        answered = {external for _internal, external in ctx.state.verified_port_pairs}
        return len(set(rented_data.get_filler_ports(ctx.executor.uuid)) - answered)
