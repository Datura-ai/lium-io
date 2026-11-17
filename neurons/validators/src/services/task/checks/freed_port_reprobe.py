"""DAH-6548: re-probe the ports a rental or filler freed during the cycle before an idle node is judged on them.

The port probe skips the ports the batch-start snapshot gave a rental or our filler. When a later read shows that
rental ended or that filler stopped, the node is judged idle on a probe that never tried those ports; a node that
forwards only them reads 0 and fails, then passes one cycle later. Called before PortCountCheck's floor and from
TenantEnforcementCheck when every listed pod is stale.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace

from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

from core.utils import _m, get_extra_info

from ..messages import PortCountMessages, render_message
from ..pipeline import CheckResult, Context, ContextState
from . import port_connectivity
from .rental_owner import rental_of_this_node
from .task_time import task_seconds_left

logger = logging.getLogger(__name__)

# bounds the whole re-probe, reads and the DinD probe included (a batch probe is two 60 s attempts)
REPROBE_MAX_SECONDS = 180
# kept for the DinD container's cleanup inside the bound
DIND_CLEANUP_MARGIN_SECONDS = 30
WITHHELD_TTL_SECONDS = 6 * 3600


@dataclass(frozen=True)
class Reprobe:
    # the re-probe's measurement in place of the cycle's first probe; rented_data is the read after it, ended pods
    # dropped. None when withheld.
    state: ContextState | None = None
    # why the verdict is withheld
    withheld: str | None = None


def _without_pods(rented_data: RentedExecutorsResponse, ctx: Context, pod_ids: set[str]) -> RentedExecutorsResponse:
    rental = rented_data.executors.get(ctx.executor.uuid)
    if rental is None or not pod_ids:
        return rented_data
    pods = [pod for pod in rental.pods if pod.pod_id not in pod_ids]
    return rented_data.model_copy(
        update={"executors": {**rented_data.executors, ctx.executor.uuid: rental.model_copy(update={"pods": pods})}}
    )


def _held(rented_data: RentedExecutorsResponse | None, ctx: Context) -> tuple[list[int], list[str], list[int]]:
    """(customer pod ports, pod names, filler ports) as `rented_data` sees the node."""
    if rented_data is None:
        return [], [], []
    rental = rental_of_this_node(ctx, rented_data=rented_data)
    pods = rental.pods if rental else []
    return (
        [port for pod in pods for port in pod.rented_ports],
        [pod.container_name for pod in pods],
        rented_data.get_filler_ports(ctx.executor.uuid),
    )


def freed_ports(ctx: Context, ended_pod_ids: set[str]) -> set[int]:
    """Ports the cycle's probe skipped that the latest rented list in state (less pods known ended) no longer holds."""
    excluded = set(ctx.state.probe_excluded_ports)
    if not excluded or ctx.state.port_check_deferred:
        return set()
    rented_data = ctx.state.rented_data
    if rented_data is not None:
        rented_data = _without_pods(rented_data, ctx, ended_pod_ids)
    pod_ports, _, filler_ports = _held(rented_data, ctx)
    return excluded - set(pod_ports) - set(filler_ports)


async def reprobe_freed_ports(ctx: Context, *, min_ports: int, ended_pod_ids: set[str] | None = None) -> Reprobe | None:
    """Re-probe an idle node short of `min_ports` whose skipped ports were freed. None when nothing was freed, or
    when a withheld verdict is already on record for the node: the cycle then gets its verdict as before."""
    ended = set(ended_pod_ids or ())
    freed = freed_ports(ctx, ended)
    if not freed:
        return None
    extra = {**ctx.default_extra, "freed_port_count": len(freed)}
    started = time.monotonic()
    budget_left = task_seconds_left(ctx) - port_connectivity.CHECKS_AFTER_DIND_RESERVE_SECONDS
    if budget_left < REPROBE_MAX_SECONDS:
        return await _withhold_or_none(ctx, "no_time_budget", extra)
    try:
        async with asyncio.timeout(REPROBE_MAX_SECONDS):
            state, why = await _reprobe(ctx, min_ports, ended, started + REPROBE_MAX_SECONDS)
    except TimeoutError:
        state, why = None, "timed_out"
    except Exception as exc:  # noqa: BLE001 - an unfinished re-probe withholds, never fails the node
        logger.warning(_m("FREED_PORT_REPROBE_FAILED", extra=get_extra_info({**extra, "error": repr(exc)})))
        state, why = None, "probe_raised"
    extra["reprobe_seconds"] = round(time.monotonic() - started, 1)
    extra["task_seconds_left"] = round(task_seconds_left(ctx), 1)
    if state is None:
        return await _withhold_or_none(ctx, why, extra)
    await _clear_withheld(ctx)
    logger.info(
        _m(
            "FREED_PORT_REPROBE_DONE",
            extra=get_extra_info({**extra, "verified_port_count": state.verified_port_count}),
        )
    )
    return Reprobe(state=state)


async def _reprobe(
    ctx: Context, min_ports: int, ended: set[str], deadline: float
) -> tuple[ContextState | None, str | None]:
    backend = ctx.services.backend
    before = await backend.get_rented_executors_now()
    if before is None:
        return None, "rented_list_unreadable"  # ownership unknown: no port is taken as free
    pod_ports, pod_names, filler_ports = _held(_without_pods(before, ctx, ended), ctx)
    result = await ctx.services.connectivity.verify_ports(
        ctx.ssh,
        ctx.miner_hotkey,
        ctx.executor,
        # a filler stop can fail the scrape's own Sysbox test (its first DinD); the re-probe measures it again
        True,
        rented_ports=pod_ports,
        rented_pod_names=pod_names,
        filler_ports=filler_ports,
        log_ctx={
            "pipeline_id": ctx.pipeline_id,
            "job_batch_id": ctx.config.job_batch_id,
            "miner_hotkey": ctx.miner_hotkey,
            "executor_uuid": ctx.executor.uuid,
            "executor_ip": ctx.executor.address,
        },
        min_ports=min_ports,
        dind_deadline_monotonic=deadline - DIND_CLEANUP_MARGIN_SECONDS,
    )
    if result.status == "error" or result.connection_lost or port_connectivity._ssh_session_closed(ctx.ssh):
        return None, f"probe_{result.status}"
    # a rental that started during the re-probe is judged as rented by the caller
    after = await backend.get_rented_executors_now()
    if after is None:
        return None, "rented_list_unreadable"
    # this re-probe supersedes the completed check on record: the next pressured cycle checks before it defers
    port_connectivity._COMPLETED_CHECK_SAVED_AT.pop((ctx.miner_hotkey, ctx.executor.uuid), None)
    # DAH-4537: one outside miss per cycle; a blocked re-probe after this cycle's blocked reading keeps its verdict
    if result.external_probe == "blocked" and ctx.state.external_probe_blocked:
        unreachable = ctx.state.externally_unreachable
    else:
        unreachable = await port_connectivity.PortConnectivityCheck._outside_failure_confirmed(ctx, result)
    successful = [] if unreachable else list(result.successful_ports)
    after = _without_pods(after, ctx, ended)
    return (
        replace(
            ctx.state,
            # probed ports only: PortCountCheck._count adds the filler's
            specs={**ctx.state.specs, "verified_ports": [p.external for p in successful]},
            verified_port_count=len(successful),
            probed_port_count=len(result.selected_ports),
            declared_port_count=result.declared_port_count,
            verified_port_pairs=[(p.internal, p.external) for p in successful],
            externally_unreachable=unreachable,
            external_probe_blocked=ctx.state.external_probe_blocked or result.external_probe == "blocked",
            port_probe_connection_lost=None,
            probe_excluded_ports=sorted(set(pod_ports) | set(filler_ports)),
            rented_data=after,
            sysbox_runtime=ctx.state.sysbox_runtime or result.sysbox_measured is True,
        ),
        None,
    )


async def _withhold_or_none(ctx: Context, why: str | None, extra: dict) -> Reprobe | None:
    """One withheld cycle in a row per node: the next affected cycle gets its verdict as before (None). A Redis
    error also gives the verdict as before, since the bound could not be kept."""
    redis, hotkey, uuid = ctx.services.redis, ctx.miner_hotkey, ctx.executor.uuid
    try:
        first = await redis.record_freed_port_reprobe_withheld(hotkey, uuid, WITHHELD_TTL_SECONDS)
        if not first:
            await redis.clear_freed_port_reprobe_withheld(hotkey, uuid)
    except Exception as exc:  # noqa: BLE001
        logger.warning(_m("freed-port withheld record unavailable", extra=get_extra_info({**extra, "error": repr(exc)})))
        first = False
    logger.info(
        _m(
            "FREED_PORT_REPROBE_WITHHELD" if first else "FREED_PORT_REPROBE_NOT_DONE_VERDICT_STANDS",
            extra=get_extra_info({**extra, "reason": why}),
        )
    )
    return Reprobe(withheld=why) if first else None


async def _clear_withheld(ctx: Context) -> None:
    try:
        await ctx.services.redis.clear_freed_port_reprobe_withheld(ctx.miner_hotkey, ctx.executor.uuid)
    except Exception as exc:  # noqa: BLE001
        extra = {**ctx.default_extra, "error": repr(exc)}
        logger.warning(_m("freed-port withheld record not cleared", extra=get_extra_info(extra)))


def withheld_result(ctx: Context, check_id: str, why: str, what: dict | None = None, **updates) -> CheckResult:
    return CheckResult(
        passed=False,
        event=render_message(
            PortCountMessages.VERDICT_WITHHELD_FREED_PORT_REPROBE,
            ctx=ctx,
            check_id=check_id,
            what={"withheld_reason": why, **(what or {})},
        ),
        updates=updates,
    )
