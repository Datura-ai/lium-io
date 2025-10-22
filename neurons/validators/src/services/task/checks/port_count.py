from __future__ import annotations

from typing import Awaitable, Callable

from ..models import build_msg
from ..pipeline import CheckResult, Context


class PortCountCheck:
    """Record available ports so scoring can penalise poorly configured hosts.

    The legacy task surfaced the DB/Redis-derived count in its final message and used it
    for scoring. Keeping it as a check keeps that behaviour observable and testable.
    """

    check_id = "executor.validate.port_count"
    fatal = False

    def __init__(self, *, port_counter: Callable[[str, str], Awaitable[int]]):
        self.port_counter = port_counter

    async def run(self, ctx: Context) -> CheckResult:
        port_count = await self.port_counter(ctx.miner_hotkey, ctx.executor.uuid)

        event = build_msg(
            event="Port availability inspected",
            reason="PORT_COUNT_RECORDED",
            severity="info",
            category="runtime",
            impact="Proceed",
            what={"available_port_count": port_count},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )

        return CheckResult(
            passed=True,
            event=event,
            updates={"port_count": port_count},
        )
