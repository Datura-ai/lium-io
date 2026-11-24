import asyncio
import time

import redis.exceptions

from services.redis_service import PENDING_PODS_PREFIX, PENDING_RENTER_CREATE, _parse_pending_pods, create_started_at_key

from ..messages import TenantEnforcementMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node

# bounds the two Redis reads, so a starved Redis pool cannot hold a finished step
RENT_MARK_READ_TIMEOUT_SECONDS = 5
# a pending-pod entry older than this is a crashed create's, as for RedisService.get_pending_pods
PENDING_ENTRY_MAX_AGE_SECONDS = 30 * 60
# the pod of a renter's create that finished (its pending-pod entry gone) is known only from the stamp
POD_UNKNOWN = "a renter's pod"


async def rent_since_cycle_start(ctx: Context) -> str | None:
    """The pod of a renter's create on this node that the cycle-start rented list does not have, else None.

    The same two marks the VerifyX cold retry stops on: a renter's pending-pod entry (a create running now), and the
    node's create stamp written after this cycle started (a create that came and went between two reads). Neither is
    written for a filler, the validator's own rental probe or a GPU operation. `rented_data` is read once, at cycle
    start; a pod it already listed was judged by `TenantEnforcementCheck`. No Redis: no evidence of a rent.
    """
    node = f"{ctx.miner_hotkey}:{ctx.executor.uuid}"
    cycle_started_at = time.time() - (time.monotonic() - ctx.started_at_monotonic)
    try:
        # asyncio.timeout, not wait_for: on 3.11 wait_for drops a cancel that lands as the read completes
        async with asyncio.timeout(RENT_MARK_READ_TIMEOUT_SECONDS):
            pending = _parse_pending_pods(await ctx.services.redis.hget(PENDING_PODS_PREFIX, node))
            stamp = await ctx.services.redis.get(create_started_at_key(ctx.miner_hotkey, ctx.executor.uuid))
    except (redis.exceptions.RedisError, TimeoutError, ValueError):
        return None
    at_start = rental_of_this_node(ctx)
    listed = {pod.pod_id for pod in at_start.pods} if at_start else set()
    now = time.time()
    for item in pending:
        fresh = now - item.get("time", 0) < PENDING_ENTRY_MAX_AGE_SECONDS
        if fresh and item.get(PENDING_RENTER_CREATE) and item.get("pod_id") not in listed:
            return item.get("pod_id") or POD_UNKNOWN
    if stamp is not None and float(stamp) > cycle_started_at:
        return POD_UNKNOWN
    return None


def rented_since_cycle_start_event(ctx: Context, pod: str):
    return render_message(
        Msg.RENTED_SINCE_CYCLE_START,
        ctx=ctx,
        check_id=RentedMidCycleCheck.check_id,
        what={"executor_uuid": ctx.executor.uuid, "pods": [pod]},
    )


class RentedMidCycleCheck:
    """Right before the GPU and network tests: a renter's create since cycle start ends the run with no verdict.

    The previous result stands, as for a renter GPU reset: the pod is still starting, so it is not judged here.
    While the tests run, `Pipeline` polls the same marks and stops them (`stops_on_rent`).
    """

    check_id = "executor.validate.rented_mid_cycle"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        pod = await rent_since_cycle_start(ctx)
        if pod:
            return CheckResult(passed=False, event=rented_since_cycle_start_event(ctx, pod))
        event = render_message(Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id, what={"executor_uuid": ctx.executor.uuid})
        return CheckResult(passed=True, event=event)
