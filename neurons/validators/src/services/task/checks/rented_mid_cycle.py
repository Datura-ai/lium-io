from dataclasses import replace

from ..messages import TenantEnforcementMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node
from .rented_machine import TenantEnforcementCheck


class RentedMidCycleCheck(TenantEnforcementCheck):
    """Re-read the backend's rented list right before the GPU and network tests.

    `rented_data` is read once, at cycle start, so a pod created while the cycle runs is absent from
    it and the tests would run beside the renter's new pod. A node the fresh list shows rented is
    handled exactly as `TenantEnforcementCheck` handles a rented node (halt, rented score); a node
    still unrented, or a backend that does not answer, goes on as before.
    """

    check_id = "executor.validate.rented_mid_cycle"

    async def run(self, ctx: Context) -> CheckResult:
        fresh = await ctx.services.backend.get_rented_executors_now()
        rented_executor = rental_of_this_node(ctx, rented_data=fresh) if fresh else None
        if rented_executor and rented_executor.pods:
            return await super().run(ctx.model_copy(update={"state": replace(ctx.state, rented_data=fresh)}))
        event = render_message(
            Msg.NOT_RENTED,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid, "rented_list_read": fresh is not None},
        )
        return CheckResult(passed=True, event=event)
