import asyncio
import json
import logging
import time

import redis.exceptions

from core.utils import _m
from services.redis_service import (
    MAX_RENTAL_CHANGE_WITHHELD_CYCLES,
    PENDING_RENTER_CREATE,
    RENTAL_CHANGE_STATE_KEY,
    create_started_at_key,
)

from ..messages import TenantEnforcementMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node

logger = logging.getLogger(__name__)

# bounds the two Redis reads, so a starved Redis pool cannot hold a finished step
RENT_MARK_READ_TIMEOUT_SECONDS = 5
# the pod of a renter's create that finished (its pending-pod entry gone) is known only from the stamp
POD_UNKNOWN = "a renter's pod"


async def renter_pod_created_since_cycle_start(ctx: Context) -> str | None:
    """The pod of a renter's create on this node that the cycle-start rented list does not have, else None.

    The same two marks the VerifyX cold retry stops on: a renter's pending-pod entry (a create running now), and the
    node's create stamp written after this cycle started (a create that came and went between two reads). Neither is
    written for a filler, the validator's own rental probe or a GPU operation. `rented_data` is read once, at cycle
    start; a pod it already listed was judged by `TenantEnforcementCheck`, and the stamp is compared with that read's
    time, so a create that ends between the read and this task's start still counts. No Redis: no evidence of a rent.
    """
    rented_data = ctx.state.rented_data
    listed_at = rented_data.read_at if rented_data else None
    if listed_at is None:
        # no read time (a list built outside the backend client): the task start, the latest it can have been read
        listed_at = time.time() - (time.monotonic() - ctx.started_at_monotonic)
    try:
        # asyncio.timeout, not wait_for: on 3.11 wait_for drops a cancel that lands as the read completes
        async with asyncio.timeout(RENT_MARK_READ_TIMEOUT_SECONDS):
            pending = await ctx.services.redis.get_pending_pods(ctx.miner_hotkey, ctx.executor.uuid)
            stamp = await ctx.services.redis.get(create_started_at_key(ctx.miner_hotkey, ctx.executor.uuid))
    except (redis.exceptions.RedisError, TimeoutError, ValueError):
        return None
    at_start = rental_of_this_node(ctx)
    listed = {pod.pod_id for pod in at_start.pods} if at_start else set()
    for item in pending:
        if item.get(PENDING_RENTER_CREATE) and item.get("pod_id") not in listed:
            return item.get("pod_id") or POD_UNKNOWN
    if stamp is not None and float(stamp) > listed_at:
        return POD_UNKNOWN
    return None


async def count_withheld_cycle(ctx: Context) -> bool:
    """Count this cycle in the node's withheld cycles since its last pass, the counter DAH-6564's rule keeps; False
    when it already holds MAX_RENTAL_CHANGE_WITHHELD_CYCLES. A halt gives no verdict, so unbounded halts (renters'
    creates landing in every cycle) left the node no report until the backend's 45 min sweep marked it offline.
    A cycle already counted (the express lane asks again) is not counted twice. No batch id or no Redis: the halt
    stands uncounted, as before."""
    batch = ctx.config.job_batch_id
    if batch is None:
        return True
    field = f"{ctx.miner_hotkey}:{ctx.executor.uuid}"
    try:
        async with asyncio.timeout(RENT_MARK_READ_TIMEOUT_SECONDS):
            raw = await ctx.services.redis.hget(RENTAL_CHANGE_STATE_KEY, field)
            batches = json.loads(raw).get("batches", []) if raw else []
            if batch in batches:
                return True
            if len(batches) >= MAX_RENTAL_CHANGE_WITHHELD_CYCLES:
                return False
            entry = {"batches": [*batches, batch], "at": time.time()}
            await ctx.services.redis.hset(RENTAL_CHANGE_STATE_KEY, field, json.dumps(entry))
    except (redis.exceptions.RedisError, TimeoutError, ValueError):
        return True
    return True


def rented_since_cycle_start_event(ctx: Context, pod: str):
    return render_message(
        Msg.RENTED_SINCE_CYCLE_START,
        ctx=ctx,
        check_id=RentedMidCycleCheck.check_id,
        what={"executor_uuid": ctx.executor.uuid, "pods": [pod]},
    )


class RentedMidCycleCheck:
    """Right before the GPU and network tests: a renter's create since cycle start ends the run with no verdict.

    The run is withheld and its kept result is dropped: the pod is still starting, so it is not judged here.
    A test already running is not stopped: it is read again only before the next one starts. At most
    MAX_RENTAL_CHANGE_WITHHELD_CYCLES cycles since the node's last pass are withheld; past that the tests run.
    """

    check_id = "executor.validate.rented_mid_cycle"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        pod = await renter_pod_created_since_cycle_start(ctx)
        if pod and await count_withheld_cycle(ctx):
            return CheckResult(passed=False, event=rented_since_cycle_start_event(ctx, pod))
        if pod:
            logger.info(
                _m(
                    "Rented since the cycle started, but the node has had no verdict for the most cycles allowed: the tests run",
                    extra={"miner_hotkey": ctx.miner_hotkey, "executor_uuid": ctx.executor.uuid, "pod_id": pod},
                )
            )
        event = render_message(Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id, what={"executor_uuid": ctx.executor.uuid})
        return CheckResult(passed=True, event=event)
