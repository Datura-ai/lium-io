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
        extra = {**ctx.default_extra}

        if not ctx.config.job_batch_id:
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
            ctx.miner_hotkey,
            ctx.executor,
            ctx.state.sysbox_runtime,
            rented_ports=rented_ports,
            rented_pod_names=rented_pod_names,
        )
        verified_port_count = len(result.successful_ports)
        extra_info = {
            "sysbox_runtime": result.sysbox_runtime,
            "verified_port_count": verified_port_count,
        }
        updated_state = replace(
            ctx.state,
            specs={
                **ctx.state.specs,
                "sysbox_runtime": result.sysbox_runtime,
                "verified_ports": [p.external for p in result.successful_ports],
            },
            sysbox_runtime=result.sysbox_runtime,
            verified_port_count=verified_port_count,
        )

        total = len(result.successful_ports) + len(result.failed_ports)
        pct = (len(result.successful_ports) / total * 100) if total > 0 else 0
        dind_status = "ok" if result.dind_ok else "failed"
        batch_count = len(result.successful_ports) - (1 if result.dind_ok else 0)
        batch_status = "ok" if batch_count > 0 else "failed"
        ok_sample = sorted([p.internal for p in result.successful_ports])[:5]
        fail_sample = sorted([p.internal for p in result.failed_ports])[:5]
        elapsed = result.elapsed_sec or 0.0

        msg = (
            f"verification complete total_time={elapsed:.2f}s {pct:.0f}% available, "
            f"dind={dind_status} batch={batch_status} ok={len(result.successful_ports)}{ok_sample}"
            f" port_range={ctx.executor.port_range}, port_mappings={ctx.executor.port_mappings}"
        )
        if result.failed_ports:
            msg += f" fail={len(result.failed_ports)}{fail_sample}"

        if result.status != "ok":
            if result.status == "no_ports":
                details = "No port available for docker container"
            elif result.status == "no_working_ports":
                details = "No working ports found"
            elif result.status == "skipped_rental_active":
                # Rental is active, skip port check but don't fail
                event = render_message(
                    Msg.VERIFY_OK,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={"message": "Port check skipped - rental container active"},
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
            elif result.status == "error":
                details = f"Verification failed: {result.error}" if result.error else "Verification failed"
            else:
                details = "Verification failed"

            event = render_message(
                Msg.VERIFY_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "details": details,
                    "message": msg,
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
            what={"message": msg},
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
