from __future__ import annotations

from ..messages import BannedProviderMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class BannedProviderCheck:
    check_id = "gpu.validate.banned_provider"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        rented_data = ctx.state.rented_data
        gpu_uuids = [gpu_uuid for gpu_uuid in (ctx.state.gpu_uuids or "").split(",") if gpu_uuid]
        is_banned = bool(
            rented_data
            and rented_data.is_provider_banned(
                miner_hotkey=ctx.miner_hotkey,
                miner_coldkey=ctx.miner_coldkey,
                gpu_uuids=gpu_uuids,
            )
        )

        if is_banned:
            event = render_message(
                Msg.PROVIDER_BANNED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "miner_hotkey": ctx.miner_hotkey,
                    "miner_coldkey": ctx.miner_coldkey,
                    "gpu_uuids": gpu_uuids,
                },
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"is_provider_banned": True},
            )

        event = render_message(
            Msg.PROVIDER_ALLOWED,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "miner_hotkey": ctx.miner_hotkey,
                "miner_coldkey": ctx.miner_coldkey,
                "gpu_uuids": gpu_uuids,
            },
        )
        return CheckResult(passed=True, event=event)
