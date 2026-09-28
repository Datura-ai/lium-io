from __future__ import annotations

from dataclasses import replace
from typing import Any

from services.const import DEFAULT_JOB_OWNER_LIUM, MIN_PORT_COUNT

from ..messages import PortCountMessages as Msg, render_message
from ..pipeline import CheckResult, Context, ContextState


def port_count_below_listing_floor(state: ContextState) -> int | None:
    """The published `available_port_count` when it is below MIN_PORT_COUNT, else None.

    The backend lists a node only at `available_port_count >= MIN_PORT_COUNT` (lium-platform
    `daos/executor.py` get_available_executors), with no exemption for a rented node, so any run that
    publishes a lower count leaves the node's free GPUs hidden from renters, unless the backend also
    counts ports held by preemptible background jobs (lium-platform#840's
    count_preemptible_filler_ports_as_free, see port_floor_impact_text). The rent path gates
    separately, on MIN_PORT_COUNT free `verified_ports` (`services/executor.py`).
    None before PortCountCheck has written the count.
    """
    available: Any = state.specs.get("available_port_count")
    if not isinstance(available, int) or isinstance(available, bool):
        return None
    return available if available < MIN_PORT_COUNT else None


def hidden_from_renters_text(available_port_count: int) -> str:
    return f"Hidden from renters: only {available_port_count} verified ports, need {MIN_PORT_COUNT}"


def listing_needs_background_job_ports_text(
    available_port_count: int, background_job_port_count: int
) -> str:
    return (
        f"Listed only if the platform counts ports held by preemptible background jobs: "
        f"{available_port_count} verified ports plus {background_job_port_count} held, need {MIN_PORT_COUNT}"
    )


def port_floor_impact_text(state: ContextState, available_port_count: int) -> str:
    background_job_port_count = state.preemptible_background_job_port_count
    if background_job_port_count:
        return listing_needs_background_job_ports_text(
            available_port_count, background_job_port_count
        )
    return hidden_from_renters_text(available_port_count)


def port_floor_what(state: ContextState, available_port_count: int) -> dict[str, Any]:
    background_job_port_count = state.preemptible_background_job_port_count
    what: dict[str, Any] = {
        "available_port_count": available_port_count,
        "required": MIN_PORT_COUNT,
        # None: the platform lists the node only while it counts background-job ports (lium-platform#840)
        "listing_hidden": None if background_job_port_count else True,
        "probed_port_count": state.probed_port_count,
        "declared_port_count": state.declared_port_count,
    }
    if background_job_port_count:
        what["held_by_preemptible_background_jobs"] = background_job_port_count
    return what


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
        # Held ports prove no inbound reachability, so they count only once at least one port answered.
        background_job_port_count = (
            0 if is_rented or port_count == 0 else self._ports_held_by_platform_background_jobs(ctx)
        )

        updated_state = replace(
            ctx.state,
            preemptible_background_job_port_count=background_job_port_count,
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
                    "probed_port_count": ctx.state.probed_port_count,
                    "declared_port_count": ctx.state.declared_port_count,
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

        if port_count < MIN_PORT_COUNT:
            # Passed while rented: the published count is under the floor, so the free GPUs cannot be listed.
            # Passed unrented on background-job ports: the platform lists the node only while it counts
            # those ports too (lium-platform#840's flag), so the warning does not claim it is hidden.
            scored_as = (
                "the rented portion is scored as rented"
                if is_rented
                else f"scored with {background_job_port_count} ports held by preemptible background jobs"
            )
            event = render_message(
                Msg.PORT_COUNT_RECORDED,
                ctx=ctx,
                check_id=self.check_id,
                severity="warning",
                impact=f"{port_floor_impact_text(updated_state, port_count)}; {scored_as}",
                what={
                    **port_floor_what(updated_state, port_count),
                    "held_by_preemptible_background_jobs": background_job_port_count,
                    "exempt_because_rented": is_rented,
                },
            )
        else:
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
