from __future__ import annotations

from dataclasses import replace

from ..messages import PortConnectivityMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context


class PortConnectivityCheck:
    """Verify Docker port mappings by running the batch verifier exactly like before.

    Connectivity failures used to abort the task immediately because miners could not be
    rented. This check preserves that contract and updates sysbox state for later scoring.
    """

    check_id = "executor.validate.port_connectivity"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
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

        # Extract rented ports and pod names from context
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        rented_ports = rented_executor.rented_ports if rented_executor else []
        rented_pod_names = [p.container_name for p in rented_executor.pods] if rented_executor else []

        connectivity_service = ctx.services.connectivity
        result = await connectivity_service.verify_ports(
            ctx.ssh,
            ctx.config.job_batch_id or "",
            ctx.miner_hotkey,
            ctx.executor,
            ctx.config.port_private_key or "",
            ctx.config.port_public_key or "",
            ctx.state.sysbox_runtime,
            rented_ports=rented_ports,
            rented_pod_names=rented_pod_names,
        )
        extra_info = {
            "sysbox_runtime": result.sysbox_runtime,
            "verified_port_count": result.verified_port_count,
        }
        updated_state = replace(
            ctx.state,
            specs={
                **ctx.state.specs,
                "sysbox_runtime": result.sysbox_runtime,
            },
            sysbox_runtime=result.sysbox_runtime,
            verified_port_count=result.verified_port_count,
        )

        if not result.success:
            event = render_message(
                Msg.VERIFY_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "details": result.log_text,
                    "port_range": ctx.executor.port_range,
                    "port_mappings": ctx.executor.port_mappings,
                },
                extra=extra_info,
            )
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
            extra=extra_info,
        )
        return CheckResult(
            passed=True,
            event=event,
            updates={
                "default_extra": {**extra, **extra_info},
                "state": updated_state,
            },
        )
