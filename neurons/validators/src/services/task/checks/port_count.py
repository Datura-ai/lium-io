from __future__ import annotations

from ..messages import PortCountMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from services.redis_service import AVAILABLE_PORT_MAPS_PREFIX
from services.const import MIN_PORT_COUNT


class PortCountCheck:
    """Record available ports so scoring can penalise poorly configured hosts.

    The legacy task surfaced the DB/Redis-derived count in its final message and used it
    for scoring. Keeping it as a check keeps that behaviour observable and testable.
    """

    check_id = "executor.validate.port_count"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        port_mapping = ctx.services.port_mapping
        redis_service = ctx.services.redis
        executor_uuid = ctx.executor.uuid

        try:
            port_count = await port_mapping.get_successful_ports_count(executor_uuid)
        except Exception as e:
            event = render_message(
                Msg.PORT_COUNT_DB_ERROR,
                ctx=ctx,
                check_id=self.check_id,
                what={"error": str(e)},
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"port_count": 0},
            )

        if port_count < MIN_PORT_COUNT:
            port_map_key = f"{AVAILABLE_PORT_MAPS_PREFIX}:{ctx.miner_hotkey}:{executor_uuid}"
            port_maps_bytes = await redis_service.lrange(port_map_key)
            port_count = len([tuple(map(int, pm.decode().split(","))) for pm in port_maps_bytes])

        event = render_message(
            Msg.PORT_COUNT_RECORDED,
            ctx=ctx,
            check_id=self.check_id,
            what={"available_port_count": port_count},
        )

        return CheckResult(
            passed=True,
            event=event,
            updates={"port_count": port_count},
        )
