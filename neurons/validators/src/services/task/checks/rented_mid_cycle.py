from ..messages import TenantEnforcementMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node


class RentedMidCycleCheck:
    """Re-read the backend's rented list right before the GPU and network tests.

    `rented_data` is read once, at cycle start, so a pod created while the cycle runs is absent from
    it and the tests would run beside the renter's new pod. A pod that is new since the snapshot ends
    the run with no verdict (the previous result stands, as for a renter GPU reset): it is still
    starting, so it is not judged here. Pods the snapshot already listed were judged by
    `TenantEnforcementCheck`; with none new, or a backend that does not answer, the run goes on.
    """

    check_id = "executor.validate.rented_mid_cycle"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        fresh = await ctx.services.backend.get_rented_executors_now()
        rented_now = rental_of_this_node(ctx, rented_data=fresh) if fresh else None
        at_start = rental_of_this_node(ctx)
        known = {pod.pod_id for pod in at_start.pods} if at_start else set()
        new_pods = [pod.pod_id for pod in rented_now.pods if pod.pod_id not in known] if rented_now else []
        what = {"executor_uuid": ctx.executor.uuid}
        if new_pods:
            event = render_message(
                Msg.RENTED_SINCE_CYCLE_START, ctx=ctx, check_id=self.check_id, what={**what, "pods": new_pods}
            )
            return CheckResult(passed=False, event=event)
        event = render_message(
            Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id, what={**what, "rented_list_read": fresh is not None}
        )
        return CheckResult(passed=True, event=event)
