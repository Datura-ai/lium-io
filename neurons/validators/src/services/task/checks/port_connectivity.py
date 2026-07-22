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
        rented_ports = rented_executor.get_rented_ports() if rented_executor else []
        rented_pod_names = [p.container_name for p in rented_executor.pods] if rented_executor else []

        connectivity_service = ctx.services.connectivity
        result = await connectivity_service.verify_ports(
            ctx.ssh,
            ctx.miner_hotkey,
            ctx.executor,
            ctx.state.sysbox_runtime,
            rented_ports=rented_ports,
            rented_pod_names=rented_pod_names,
            log_ctx={
                "pipeline_id": ctx.pipeline_id,
                "job_batch_id": ctx.config.job_batch_id,
                "miner_hotkey": ctx.miner_hotkey,
                "executor_uuid": ctx.executor.uuid,
                "executor_ip": ctx.executor.address,
            },
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

        # DAH-2272 (tolerate): a customer rental force-removes port-check / DinD
        # probe containers the instant a ContainerCreateRequest lands (see
        # DockerService.wait_for_port_check_containers). That race can flip
        # sysbox_runtime to False for this cycle even though the executor is
        # fine. Don't record a rental-induced sysbox downgrade — keep the last
        # known value and let the next verification cycle re-measure. Mirrors
        # the rented-executor sysbox fallback in ExecutorConnectivityService.
        sysbox_downgraded = ctx.state.sysbox_runtime and not result.sysbox_runtime
        if sysbox_downgraded and await ctx.services.redis.renting_in_progress(
            ctx.miner_hotkey, ctx.executor.uuid
        ):
            extra_info["sysbox_downgrade_tolerated"] = True
            updated_state = replace(
                updated_state,
                specs={
                    **updated_state.specs,
                    "sysbox_runtime": ctx.state.sysbox_runtime,
                },
                sysbox_runtime=ctx.state.sysbox_runtime,
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
            # Fetch fresh rental data from backend to ensure we have latest state
            backend_client = ctx.services.backend
            fresh_rented_data = await backend_client.get_all_rented_executors()

            rental_info = {}
            if fresh_rented_data:
                # Update the state with fresh rental data
                updated_state = replace(
                    updated_state,
                    rented_data=fresh_rented_data,
                )
                # Add rental context to help debug
                rented_executor = fresh_rented_data.executors.get(ctx.executor.uuid) if fresh_rented_data else None
                if rented_executor:
                    rental_info = {
                        "has_rental": True,
                        "rental_pod_count": len(rented_executor.pods),
                        "rental_port_count": len(rented_executor.get_rented_ports()),
                        "rental_pods": [p.container_name for p in rented_executor.pods],
                    }

            # Provide detailed error messages based on status
            if result.status == "no_ports":
                details = "No ports available for docker container - all ports may be in use or misconfigured"
            elif result.status == "no_working_ports":
                details = f"No working ports found - verified {len(result.failed_ports)} ports, all failed connectivity test"
            elif result.status == "skipped_rental_active":
                # This shouldn't happen anymore but provide clear message if it does
                details = "Port verification was incorrectly skipped due to rental detection - this is a bug"
            elif result.status == "error":
                details = f"Port verification error: {result.error}" if result.error else "Port verification encountered an unexpected error"
            else:
                details = f"Port verification failed with status: {result.status}"

            event = render_message(
                Msg.VERIFY_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "details": details,
                    "message": msg,
                    "port_range": ctx.executor.port_range,
                    "port_mappings": ctx.executor.port_mappings,
                    "verification_status": result.status,
                    "total_ports_tested": len(result.successful_ports) + len(result.failed_ports),
                    "successful_ports": len(result.successful_ports),
                    "failed_ports": len(result.failed_ports),
                    **rental_info,
                },
                extra=extra_info,
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"default_extra": {**extra, **extra_info}, "state": updated_state},
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
