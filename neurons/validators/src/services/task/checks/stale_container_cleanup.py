from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

from core.utils import _m
from protocol.vc_protocol.validator_requests import ContainerState, PodContainerState
from services.redis_service import CLEANUP_SEEN_EXECUTORS_SET

from ...const import POD_CONTAINER_PREFIX
from ..messages import StaleContainerCleanupMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)


# DAH-2805: how often the download-temporary sweep may run per executor. The check itself runs every
# pipeline cycle (~15 min), but a temporary cannot become eligible until it is
# DOWNLOAD_TEMPORARY_MAX_AGE_MINUTES old, so walking the cache every cycle buys nothing and costs a
# helper container plus a tree walk on every node. CustomBuildOrphanSweepCheck throttles itself for
# the same reason, on 6 h; unlike it, this one also stamps a failed sweep, which only costs the node
# one hour of delay on a transient SSH error.
DOWNLOAD_TEMPORARY_SWEEP_INTERVAL_SECONDS = 60 * 60

# DAH-3338: a `reaped` state exists only after the removal and leaves the validator once, over the
# MACHINE_SPEC_CHANNEL pub/sub — no subscriber at that moment and the backend never learns the
# container is gone (the next cycle finds nothing to reap). So every reaped pod id is queued in
# redis BEFORE its container is removed and re-sent with every cycle's pod_states until it is this
# old; the backend's write is idempotent (a repeat changes nothing), so a re-send costs nothing.
REAPED_POD_STATE_RETENTION_SECONDS = 24 * 60 * 60
REAPED_POD_STATES_KEY_PREFIX = "reaped_pod_states:"


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

    def __init__(self) -> None:
        # executor uuid -> when its cache was last swept. In memory on the check, which the pipeline
        # factory builds once and reuses across cycles; a validator restart simply sweeps again.
        self._last_sweep_at: dict[str, float] = {}

    async def run(self, ctx: Context) -> CheckResult:
        # DAH-1932: when a miner re-adds an executor it gets a new subnet UUID for the same
        # IP:port. `rented_data` is keyed by UUID, so the tenant that is still running on the
        # box is listed under the OLD uuid and this executor looks unrented. Removing "orphans"
        # now would kill a paying customer's pod. A UUID this validator has never run the cleanup
        # for gets one cycle of grace; the set lives in Redis so a validator restart does not
        # reopen the race. The grace is a bridge, not the fix: the backend remaps the row to the
        # new UUID only on a publish with a positive score, and a first cycle that fails
        # PortCountCheck publishes 0 — closing that (in the validator by address, or in the
        # backend's zero-score publish) is the open question on the PR.
        # DAH-3338: the reaped-pod queue is built either way, so a grace cycle still re-sends the
        # ids an earlier cycle reaped and queued.
        queue = _ReapedPodStateQueue(ctx)
        first_sight = await self._first_sight(ctx)
        if first_sight:
            removed_count, removed_names, unremovable_names = 0, [], []
        else:
            (
                removed_count,
                removed_names,
                unremovable_names,
            ) = await ctx.services.container_cleanup.cleanup(
                ssh_client=ctx.ssh,
                rented_data=ctx.state.rented_data,
                executor_uuid=ctx.executor.uuid,
                on_before_remove=queue.add,
            )

        # DAH-2805: killed weight downloads leave `*.incomplete` files nothing reads again — 741 GB
        # on one prod node. Swept from here because this check reaches every node, whatever image it
        # runs; throttled because the files it looks for cannot appear faster than the age window.
        # BEFORE the reclaim below on purpose: the reclaim measures free disk itself, so garbage
        # freed here can be the difference between keeping the node's ~190 GB cache and losing it.
        swept_download_temporaries: int | None = None
        now: float = time.monotonic()
        last_sweep_at: float = self._last_sweep_at.get(ctx.executor.uuid, float("-inf"))
        if now - last_sweep_at >= DOWNLOAD_TEMPORARY_SWEEP_INTERVAL_SECONDS:
            swept_download_temporaries = await ctx.services.container_cleanup.sweep_abandoned_download_temporaries(
                ssh_client=ctx.ssh,
                executor_uuid=ctx.executor.uuid,
            )
            self._last_sweep_at[ctx.executor.uuid] = now

        # DAH-2475: give the DPHN filler cache back when the node can no longer afford it. This is
        # the ONLY caller — the create-time sweep deliberately never reclaims (it would raise free
        # disk, the backend would grant the cache again next cycle, and the node would re-download
        # ~37 GB forever), so a node that has fallen under the rental listing floor is rescued from
        # here, where the decision is made on real free space and outside any launch.
        reclaimed_cache_volumes = await ctx.services.container_cleanup.reclaim_dphn_cache_when_disk_is_tight(
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
                "unremovable_containers": unremovable_names,
                "first_sight_grace": first_sight,
                "reclaimed_cache_volumes": reclaimed_cache_volumes,
                "swept_download_temporaries": swept_download_temporaries,
            },
        )
        # DAH-2991: an orphan that survived removal still holds its ports; PortCountCheck names it
        # instead of reporting a bare count the provider has to diagnose by hand.
        # DAH-3338: a reaped pod_* container is reported to the backend as `reaped`, so the rental
        # it belonged to learns its container is confirmed gone without a route of its own. The
        # report comes from the redis queue: this cycle's reaps plus every earlier one still inside
        # the retention window, so a report lost on the pub/sub hop is sent again next cycle.
        for name in unremovable_names:
            # queued before the attempt (the hook runs first), still on the host: not reaped
            await queue.drop(name)
        reaped = await queue.states()
        if not unremovable_names and not reaped:
            return CheckResult(passed=True, event=event)
        state = replace(
            ctx.state,
            orphaned_containers=unremovable_names,
            pod_states=[*ctx.state.pod_states, *reaped],
        )
        return CheckResult(passed=True, event=event, updates={"state": state})

    async def _first_sight(self, ctx: Context) -> bool:
        """Record the executor UUID; True the first time this validator meets it.

        Redis trouble counts as first sight: skipping one cycle of garbage collection
        costs one cycle of delayed GC, removing a live tenant's container cannot be undone.
        """
        try:
            return bool(await ctx.services.redis.sadd(CLEANUP_SEEN_EXECUTORS_SET, ctx.executor.uuid))
        except Exception as e:  # noqa: BLE001 - best-effort check, never fatal
            logger.warning(
                _m(
                    "Could not record executor as seen; skipping stale container cleanup this cycle",
                    extra={"executor_uuid": ctx.executor.uuid, "error": str(e)},
                )
            )
            return True


def pod_id_of_container(name: str) -> str | None:
    """The pod id a rental container's name carries, or None.

    Only ``pod_<uuid>`` names belong to a rental; health-check and filler containers the sweep
    also removes are not reported. The id must parse as a UUID: the host names its containers,
    and the backend skips a pod id it cannot read (and refuses one over 64 characters), so an
    arbitrary ``pod_*`` string is never put on the wire.
    """
    if not name.startswith(POD_CONTAINER_PREFIX):
        return None
    pod_id = name.removeprefix(POD_CONTAINER_PREFIX)
    try:
        UUID(pod_id)
    except ValueError:
        return None
    return pod_id


class _ReapedPodStateQueue:
    """Per-executor redis hash ``pod_id -> observed_at`` of reaped rental containers (DAH-3338).

    Every method swallows a redis error: the queue is delivery insurance, and a redis hiccup must
    neither stop the removal nor change the executor's verdict. Without a redis service (tests,
    dry runs) the queue holds this cycle's reaps only.
    """

    def __init__(self, ctx: Context) -> None:
        self._redis = ctx.services.redis
        self._key = f"{REAPED_POD_STATES_KEY_PREFIX}{ctx.executor.uuid}"
        self._extra = {"executor_uuid": ctx.executor.uuid}
        self._this_cycle: dict[str, datetime] = {}

    async def add(self, container_name: str) -> None:
        pod_id = pod_id_of_container(container_name)
        if pod_id is None or pod_id in self._this_cycle:
            return
        observed_at = datetime.now(UTC)
        self._this_cycle[pod_id] = observed_at
        if self._redis is None:
            return
        try:
            await self._redis.hset(self._key, pod_id, observed_at.isoformat())
            # the hash of a node that leaves the fleet is not read again; the TTL is what removes it
            await self._redis.expire(self._key, REAPED_POD_STATE_RETENTION_SECONDS)
        except Exception as e:
            logger.warning(_m("reaped pod state not queued", extra={**self._extra, "pod_id": pod_id, "error": str(e)}))

    async def drop(self, container_name: str) -> None:
        pod_id = pod_id_of_container(container_name)
        if pod_id is None:
            return
        self._this_cycle.pop(pod_id, None)
        if self._redis is None:
            return
        try:
            await self._redis.hdel(self._key, pod_id)
        except Exception as e:
            logger.warning(_m("reaped pod state not dropped", extra={**self._extra, "pod_id": pod_id, "error": str(e)}))

    async def states(self) -> list[PodContainerState]:
        queued: dict[str, datetime] = dict(self._this_cycle)
        if self._redis is not None:
            try:
                stored = await self._redis.hgetall(self._key)
            except Exception as e:
                logger.warning(_m("reaped pod state queue unreadable", extra={**self._extra, "error": str(e)}))
                stored = {}
            cutoff = datetime.now(UTC).timestamp() - REAPED_POD_STATE_RETENTION_SECONDS
            expired: list[str] = []
            for raw_pod_id, raw_observed_at in (stored or {}).items():
                pod_id = raw_pod_id.decode() if isinstance(raw_pod_id, bytes) else str(raw_pod_id)
                raw_observed_at = (
                    raw_observed_at.decode() if isinstance(raw_observed_at, bytes) else str(raw_observed_at)
                )
                try:
                    observed_at = datetime.fromisoformat(raw_observed_at)
                except ValueError:
                    expired.append(pod_id)
                    continue
                if observed_at.timestamp() < cutoff:
                    expired.append(pod_id)
                    continue
                queued.setdefault(pod_id, observed_at)
            if expired:
                try:
                    await self._redis.hdel(self._key, *expired)
                except Exception as e:
                    logger.warning(_m("expired reaped pod states not dropped", extra={**self._extra, "error": str(e)}))
        return [
            PodContainerState(pod_id=pod_id, container_state=ContainerState.REAPED, observed_at=observed_at)
            for pod_id, observed_at in sorted(queued.items(), key=lambda item: item[1])
        ]
