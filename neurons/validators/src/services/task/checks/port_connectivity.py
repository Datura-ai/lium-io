from __future__ import annotations

from dataclasses import replace

from ..messages import PortConnectivityMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class PortConnectivityCheck:
    """Verify Docker port mappings by running the batch verifier exactly like before.

    Connectivity failures used to abort the task immediately because miners could not be
    rented. This check preserves that contract and updates sysbox state for later scoring.
    """

    check_id = "executor.validate.port_connectivity"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if ctx.rented:
            event = render_message(
                Msg.SKIPPED_RENTED,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=True, event=event)

        redis_service = ctx.services.redis
        renting_in_progress = await redis_service.renting_in_progress(ctx.miner_hotkey, ctx.executor.uuid)
        extra = {**ctx.default_extra, "renting_in_progress": renting_in_progress}

        if renting_in_progress:
            event = render_message(
                Msg.RENTING_IN_PROGRESS,
                ctx=ctx,
                check_id=self.check_id,
                what={"renting_in_progress": True},
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={"default_extra": extra, "renting_in_progress": True},
            )

        if not all([ctx.config.job_batch_id, ctx.config.port_private_key, ctx.config.port_public_key]):
            event = render_message(
                Msg.CONFIG_MISSING,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=False, event=event)

        connectivity_service = ctx.services.connectivity
        result = await connectivity_service.verify_ports(
            ctx.ssh,
            ctx.config.job_batch_id or "",
            ctx.miner_hotkey,
            ctx.executor,
            ctx.config.port_private_key or "",
            ctx.config.port_public_key or "",
            ctx.state.sysbox_runtime,
        )

        if not result.success:
            event = render_message(
                Msg.VERIFY_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={"details": result.log_text},
            )
            updated_state = replace(ctx.state, sysbox_runtime=result.sysbox_runtime)
            return CheckResult(
                passed=False,
                event=event,
                updates={"default_extra": extra, "state": updated_state},
            )

        event = render_message(
            Msg.VERIFY_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"message": result.log_text},
        )
        updated_state = replace(ctx.state, sysbox_runtime=result.sysbox_runtime)
        return CheckResult(
            passed=True,
            event=event,
            updates={
                "default_extra": extra,
                "state": updated_state,
            },
        )
