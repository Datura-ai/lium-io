from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from services.const import MIN_PORT_COUNT
from services.port_utils import DEFAULT_PORT_RANGE_TEXT

from ..messages import PortCountMessages as Msg, render_message
from ..pipeline import CheckResult, Context, ContextState


def port_count_below_listing_floor(state: ContextState) -> int | None:
    """The published `available_port_count` when it is below MIN_PORT_COUNT, else None.

    The backend lists a node only at `available_port_count >= MIN_PORT_COUNT` (lium-platform
    `daos/executor.py` get_available_executors), with no exemption for a rented node, so any run that
    publishes a lower count leaves the node's free GPUs hidden from renters. The rent path gates
    separately, on MIN_PORT_COUNT free `verified_ports` (`services/executor.py`).
    None before PortCountCheck has written the count.
    """
    available: Any = state.specs.get("available_port_count")
    if not isinstance(available, int) or isinstance(available, bool):
        return None
    return available if available < MIN_PORT_COUNT else None


def hidden_from_renters_text(available_port_count: int) -> str:
    return f"Hidden from renters: only {available_port_count} verified ports, need {MIN_PORT_COUNT}"


# The port check's verdict as the portal names it; the portal shows it only when no earlier
# status (RENTED, a new-rentals pause) applies.
LISTING_PORT_CHECK_CODE = "INSUFFICIENT_PORTS"


def declares_port_mappings(raw: Any) -> bool:
    """True when `raw` holds at least one [internal, external] pair.

    Same rule as the platform's rent-path port-mapping parser: `[]`, `"[]"`, `"{}"` and
    unparsable text declare no mappings. The listing does not read mappings or the range.
    """
    try:
        raw = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(raw, list) and any(isinstance(m, list) and len(m) >= 2 for m in raw)


def port_floor_what(state: ContextState, available_port_count: int) -> dict[str, Any]:
    """`port_range` is the range the node declares, or the default one when it declares none.

    It is None when the node declares mappings. Mappings that are present but hold no pair
    (`"[]"`, `"{}"`) leave the validator nothing to probe, so the range is named while none of it
    was probed (`probed_port_count` 0).
    """
    return {
        "available_port_count": available_port_count,
        "required": MIN_PORT_COUNT,
        "listing_hidden": True,
        "listing_check": LISTING_PORT_CHECK_CODE,
        "port_range": None
        if declares_port_mappings(state.specs.get("port_mappings"))
        else state.specs.get("port_range") or DEFAULT_PORT_RANGE_TEXT,
        "probed_port_count": state.probed_port_count,
        "declared_port_count": state.declared_port_count,
    }


class PortCountCheck:
    """Verify minimum port availability and record port count for scoring.

    Reads the verified port count from ctx.state (set by PortConnectivityCheck)
    instead of querying the database.
    """

    check_id = "executor.validate.port_count"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        port_count: int | None = ctx.state.verified_port_count

        # DAH-2647: failing on an absent measurement zeroes healthy nodes.
        if port_count is None:
            event = render_message(
                Msg.PORT_COUNT_NOT_MEASURED,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=True, event=event)

        # Check if executor is rented (same pattern as rented_machine.py)
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        is_rented = rented_executor is not None and len(rented_executor.pods) > 0

        updated_state = self._state_with_published_port_count(ctx, port_count)

        if not is_rented and port_count < MIN_PORT_COUNT:
            # DAH-2991: when the shortfall is our own leftover — an orphaned rental container the
            # cleanup could not remove — say so, instead of a bare count (ticket-0287 diagnosed it by hand).
            orphaned = ctx.state.orphaned_containers
            event = render_message(
                Msg.INSUFFICIENT_PORTS,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    **port_floor_what(updated_state, port_count),
                    "held_by_orphaned_containers": orphaned,
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
            # The rented portion is scored as rented; the free GPUs still cannot be listed or rented.
            event = render_message(
                Msg.PORT_COUNT_RECORDED,
                ctx=ctx,
                check_id=self.check_id,
                severity="warning",
                impact=f"{hidden_from_renters_text(port_count)}; the rented portion is scored as rented",
                what={**port_floor_what(updated_state, port_count), "exempt_because_rented": True},
            )
        else:
            event = render_message(
                Msg.PORT_COUNT_RECORDED,
                ctx=ctx,
                check_id=self.check_id,
                what={"available_port_count": port_count},
            )

        return CheckResult(
            passed=True,
            event=event,
            updates={"port_count": port_count, "state": updated_state},
        )

    @staticmethod
    def _state_with_published_port_count(ctx: Context, port_count: int) -> ContextState:
        """The state carrying this cycle's port availability into the specs the backend stores."""
        return replace(
            ctx.state,
            specs={
                **ctx.state.specs,
                "available_port_count": port_count,
                "port_range": ctx.executor.port_range,
                "port_mappings": ctx.executor.port_mappings,
            },
        )
