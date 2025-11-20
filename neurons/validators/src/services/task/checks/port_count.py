from __future__ import annotations

from dataclasses import replace
from ..messages import PortCountMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class PortCountCheck:
    """Record available ports so scoring can penalise poorly configured hosts.

    The legacy task surfaced the DB/Redis-derived count in its final message and used it
    for scoring. Keeping it as a check keeps that behaviour observable and testable.
    """

    check_id = "executor.validate.port_count"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        port_mapping = ctx.services.port_mapping
        executor_uuid = ctx.executor.uuid

        port_count = await port_mapping.get_successful_ports_count(executor_uuid)

        updated_state = replace(
            ctx.state,
            specs={
                **ctx.state.specs,
                "available_port_count": port_count,
            },
        )
        
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
