import redis.exceptions

from services.redis_service import rent_requested_key

from ..messages import TenantEnforcementMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node


async def rent_requested_since_cycle_start(ctx: Context) -> str | None:
    """The pod of a renter's create the connector marked on this node, unless the cycle-start rented list has it.

    `rented_data` is read once, at cycle start; a pod it already listed was judged by `TenantEnforcementCheck`.
    No mark (none sent, the create failed, or the mark expired) or no Redis: no rent to stop for.
    """
    try:
        pod = await ctx.services.redis.get(rent_requested_key(ctx.miner_hotkey, ctx.executor.uuid))
    except redis.exceptions.RedisError:
        return None
    if pod is None:
        return None
    pod = pod.decode() if isinstance(pod, bytes) else pod
    at_start = rental_of_this_node(ctx)
    return None if at_start and pod in {listed.pod_id for listed in at_start.pods} else pod


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
