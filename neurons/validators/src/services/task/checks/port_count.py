from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from services.const import MIN_PORT_COUNT
from services.port_utils import DEFAULT_PORT_RANGE_TEXT

from ..availability import AVAILABILITY_CATEGORY, ReachSource, ReachTarget
from ..messages import PortCountMessages as Msg, render_message
from ..pipeline import CheckResult, Context, ContextState

logger = logging.getLogger(__name__)


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


def split_port_requirement(state: ContextState) -> tuple[int, int] | None:
    """(max splits, ports needed) for a node offered as GPU splits, else None.

    DAH-3698 (owner, 29 Oct 2026, "minimum ports as maximum splits"): a split node lists only with one
    free verified port for each of its gpu_count // gpu_splitting_min_count splits (a rental takes one
    port since lium-platform#2122), and never below the plain MIN_PORT_COUNT floor.
    """
    min_count = state.gpu_splitting_min_count
    if not state.supports_gpu_splitting or not min_count or min_count <= 0:
        return None
    max_splits = (state.gpu_count or 0) // min_count
    return max_splits, max(max_splits, MIN_PORT_COUNT)


def split_port_shortfall(
    state: ContextState, verified_override: int | None = None
) -> tuple[int, tuple[int, int]] | None:
    """(port count, split_port_requirement) when a split node has the plain floor but not one port
    per split, else None (below the floor, the floor reports it). The count is the published
    `available_port_count`, or `verified_override` when the caller re-probed the node itself."""
    need = split_port_requirement(state)
    available: Any = state.specs.get("available_port_count")
    if not need or not isinstance(available, int) or isinstance(available, bool):
        return None
    count = available if verified_override is None else verified_override
    return (count, need) if MIN_PORT_COUNT <= available and count < need[1] else None


def split_ports_event(
    ctx: Context,
    check_id: str,
    state: ContextState,
    port_count: int,
    split_need: tuple[int, int],
    *,
    what: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
):
    """INSUFFICIENT_PORTS for an idle split node short of its ports.

    An availability entry, not a runtime event: the backend does not store a zero-score report, so
    only an availability entry takes an already-listed node off the market; the next passing cycle
    lists it again.
    """
    max_splits, required = split_need
    return render_message(
        Msg.INSUFFICIENT_PORTS,
        ctx=ctx,
        check_id=check_id,
        category=AVAILABILITY_CATEGORY,
        what={
            "reach_source": str(ReachSource.VALIDATOR),
            "reach_target": str(ReachTarget.EXECUTOR_PORTS),
            **port_floor_what(state, port_count),
            "required": required,
            "max_splits": max_splits,
            "gpu_splitting_min_count": state.gpu_splitting_min_count,
            **(what or {}),
        },
        remediation=(
            f"Open at least {required} renting ports for this node's {max_splits} GPU splits "
            "(one per split): widen RENTING_PORT_RANGE or add RENTING_PORT_MAPPINGS, "
            "then `docker compose up -d` in neurons/executor."
        ),
        extra=extra,
    )


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

    @staticmethod
    def _count(ctx: Context, rented_data) -> tuple[int, bool, ContextState]:
        """(port count, rented, state with the published port specs) as `rented_data` sees the node."""
        # The probe skips our idle filler's ports, but a rental removes the filler (create-time
        # cleanup) and frees them, so they join both the count and verified_ports, the set the
        # platform's rent guard and prepare_ports_data allocate from; the two stay in step.
        probed = list((ctx.state.specs or {}).get("verified_ports") or [])
        held = rented_data.get_filler_ports(ctx.executor.uuid) if rented_data else []
        filler_ports = sorted(set(held) - set(probed))
        verified_ports = probed + filler_ports  # probed first: rental verification uses [0]
        port_count = ctx.state.verified_port_count + len(filler_ports)

        # the rental TenantEnforcementCheck judges: another miner's on another endpoint is not this node's
        from .rented_machine import _rental_of_validated_miner  # rented_machine imports this module

        rented_executor = _rental_of_validated_miner(
            ctx, rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        )
        is_rented = rented_executor is not None and len(rented_executor.pods) > 0

        updated_state = replace(
            ctx.state,
            specs={
                **ctx.state.specs,
                "available_port_count": port_count,
                "verified_ports": verified_ports,
                "port_range": ctx.executor.port_range,
                "port_mappings": ctx.executor.port_mappings,
            },
        )
        return port_count, is_rented, updated_state

    async def run(self, ctx: Context) -> CheckResult:
        port_count, is_rented, updated_state = self._count(ctx, ctx.state.rented_data)

        split_need = split_port_requirement(ctx.state)
        if not is_rented and split_need and MIN_PORT_COUNT <= port_count < split_need[1]:
            # a rent or filler that started after the cycle's rental snapshot holds ports the probe found
            # busy; re-read the backend once before failing, and a backend that does not answer keeps it
            try:
                fresh = await ctx.services.backend.get_rented_executors_now()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Rented-executors re-read failed; the split port verdict stands: %s", exc)
                fresh = None
            if fresh is not None:
                port_count, is_rented, updated_state = self._count(ctx, fresh)
                updated_state = replace(updated_state, rented_data=fresh)

        # below MIN_PORT_COUNT the plain floor below reports it; a rented node keeps its exemption
        if not is_rented and split_need and MIN_PORT_COUNT <= port_count < split_need[1]:
            event = split_ports_event(ctx, self.check_id, updated_state, port_count, split_need)
            return CheckResult(
                passed=False,
                event=event,
                updates={"port_count": port_count, "state": updated_state},
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
