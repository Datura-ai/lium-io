from __future__ import annotations

from core.config import settings

from ..messages import SysboxRequiredMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context


class SysboxRequiredCheck:
    """Ban unrented executors that lack the sysbox runtime from appearing on the network.

    DAH-2313: sysbox is required to be allowed on the network. A machine that has no sysbox
    AND is not currently rented is rejected here (fatal) so the pipeline halts before scoring
    and the executor is reported with a zero score / cleared verification. Already-rented
    no-sysbox machines are left untouched so live rentals are never disrupted.
    """

    check_id = "executor.validate.sysbox_required"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.REQUIRE_SYSBOX_FOR_UNRENTED:
            event = render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id)
            return CheckResult(passed=True, event=event)

        # Determine rental status the same way as port_count.py / rented_machine.py.
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        is_rented = rented_executor is not None and len(rented_executor.pods) > 0

        if not ctx.state.sysbox_runtime and not is_rented:
            event = render_message(
                Msg.SYSBOX_MISSING,
                ctx=ctx,
                check_id=self.check_id,
                what={"sysbox_runtime": ctx.state.sysbox_runtime, "is_rented": is_rented},
            )
            # DAH-2742: no clear_verified_job_info. The DinD probe is forced to False on ANY
            # probe failure (executor_connectivity/orchestrator.py:78-83), not only on a genuinely
            # missing runtime, so a transient miss must not flip the executor instantly. A real
            # missing runtime fails every cycle and the stale-executor sweep deactivates it.
            return CheckResult(
                passed=False,
                event=event,
                updates={},
            )

        event = render_message(
            Msg.SYSBOX_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"sysbox_runtime": ctx.state.sysbox_runtime, "is_rented": is_rented},
        )
        return CheckResult(passed=True, event=event)
