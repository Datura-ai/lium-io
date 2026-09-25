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


# The platform's code for "the listing hides this node for too few verified ports" (lium-platform
# services/node_verification.py). Not INSUFFICIENT_PORTS: that one means this run failed and scored 0.
LISTING_PORT_CHECK_CODE = "INSUFFICIENT_VERIFIED_PORTS"


def declares_port_mappings(raw: Any) -> bool:
    """True when `raw` holds at least one [internal, external] pair.

    Mirrors lium-platform `_parse_nat_mapping` (utils/prepare_ports_data.py): `[]`, `"[]"`, `"{}"` and
    unparsable text declare no mappings, so the platform check names the port range instead.
    """
    if not raw:
        return False
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return False
    if not isinstance(raw, list):
        return False
    return any(isinstance(m, (list, tuple)) and len(m) >= 2 for m in raw)


def port_floor_what(state: ContextState, available_port_count: int) -> dict[str, Any]:
    port_mappings_declared = declares_port_mappings(state.specs.get("port_mappings"))
    return {
        "available_port_count": available_port_count,
        "required": MIN_PORT_COUNT,
        "listing_hidden": True,
        "listing_check": LISTING_PORT_CHECK_CODE,
        "port_range": None if port_mappings_declared else state.specs.get("port_range") or DEFAULT_PORT_RANGE_TEXT,
        "port_mappings_declared": port_mappings_declared,
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
        port_count = ctx.state.verified_port_count

        # Check if executor is rented (same pattern as rented_machine.py)
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        is_rented = rented_executor is not None and len(rented_executor.pods) > 0

        updated_state = replace(
            ctx.state,
            specs={
                **ctx.state.specs,
                "available_port_count": port_count,
                "port_range": ctx.executor.port_range,
                "port_mappings": ctx.executor.port_mappings,
            },
        )

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
