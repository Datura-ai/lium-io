import asyncio
import time

import redis.exceptions

from services.redis_service import rent_requested_key

from ..messages import TenantEnforcementMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node


async def rent_requested_since_cycle_start(ctx: Context) -> str | None:
    """The pod of a renter's create the connector marked on this node, unless the cycle-start rented list has it.

    `rented_data` is read once, at cycle start; a pod it already listed was judged by `TenantEnforcementCheck`.
    No mark (none sent, the create failed, or every mark expired) or no Redis: no rent to stop for. The read is bounded,
    so a starved Redis pool cannot hold a finished step.
    """
    try:
        marks = await asyncio.wait_for(
            ctx.services.redis.hgetall(rent_requested_key(ctx.miner_hotkey, ctx.executor.uuid)), 5
        )
    except (redis.exceptions.RedisError, TimeoutError):
        return None
    at_start = rental_of_this_node(ctx)
    listed = {pod.pod_id for pod in at_start.pods} if at_start else set()
    now = time.time()
    for pod, expires_at in (marks or {}).items():
        pod = pod.decode() if isinstance(pod, bytes) else pod
        if float(expires_at) > now and pod not in listed:
            return pod
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
    While the tests run, `Pipeline` polls the same mark and stops them (`stops_on_rent`).
    """

    check_id = "executor.validate.rented_mid_cycle"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        pod = await rent_requested_since_cycle_start(ctx)
        if pod:
            return CheckResult(passed=False, event=rented_since_cycle_start_event(ctx, pod))
        event = render_message(Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id, what={"executor_uuid": ctx.executor.uuid})
        return CheckResult(passed=True, event=event)
