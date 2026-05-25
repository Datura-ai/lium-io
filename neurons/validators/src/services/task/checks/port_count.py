from __future__ import annotations

from dataclasses import replace

from services.const import MIN_PORT_COUNT

from ..messages import PortCountMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class PortCountCheck:
    """Verify minimum port availability and record port count for scoring.

    Reads the verified port count from ctx.state (set by PortConnectivityCheck)
    instead of querying the database.
    """

    check_id = "executor.validate.port_count"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        port_count = ctx.state.verified_port_count

        updated_state = replace(
            ctx.state,
            specs={
                **ctx.state.specs,
                "available_port_count": port_count,
                "port_range": ctx.executor.port_range,
                "port_mappings": ctx.executor.port_mappings,
            },
        )

        if port_count < MIN_PORT_COUNT:
            event = render_message(
                Msg.INSUFFICIENT_PORTS,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "available_port_count": port_count,
                    "required": MIN_PORT_COUNT,
                },
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"port_count": port_count, "state": updated_state},
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
