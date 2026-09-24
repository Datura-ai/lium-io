from __future__ import annotations

from dataclasses import replace

from core.config import settings
from services.executor_connectivity.models import PortVerificationResult

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
        # DAH-2527: an idle filler holds ports without creating a pod, so an executor running only
        # fillers has no rented_executor entry at all — hence the separate lookup.
        filler_ports = rented_data.get_filler_ports(ctx.executor.uuid) if rented_data else []

        connectivity_service = ctx.services.connectivity
        result = await connectivity_service.verify_ports(
            ctx.ssh,
            ctx.miner_hotkey,
            ctx.executor,
            ctx.state.sysbox_runtime,
            rented_ports=rented_ports,
            rented_pod_names=rented_pod_names,
            filler_ports=filler_ports,
            log_ctx={
                "pipeline_id": ctx.pipeline_id,
                "job_batch_id": ctx.config.job_batch_id,
                "miner_hotkey": ctx.miner_hotkey,
                "executor_uuid": ctx.executor.uuid,
                "executor_ip": ctx.executor.address,
            },
            probe_seed=f"{ctx.executor.uuid}:{ctx.config.job_batch_id}",
        )
        verified_port_count = len(result.successful_ports)
        extra_info: dict[str, object] = {
            "sysbox_runtime": result.sysbox_runtime,
            "verified_port_count": verified_port_count,
            "probed_port_count": result.probed_port_count,
            "declared_port_count": result.declared_port_count,
            "estimated_usable_port_count": result.estimated_usable_port_count,
            "port_selection": result.port_selection,
        }
        if result.dind_error:
            extra_info["dind_error"] = result.dind_error.text
        updated_state = replace(
            ctx.state,
            specs={
                **ctx.state.specs,
                "sysbox_runtime": result.sysbox_runtime,
                "verified_ports": [p.external for p in result.successful_ports],
            },
            sysbox_runtime=result.sysbox_runtime,
            verified_port_count=verified_port_count,
            probed_port_count=result.probed_port_count,
            declared_port_count=result.declared_port_count,
            estimated_usable_port_count=result.estimated_usable_port_count,
            verified_port_pairs=[(p.internal, p.external) for p in result.successful_ports],
            dind_probe_error=result.dind_error,
        )

        if await self._should_keep_last_known_sysbox(ctx, result, extra_info):
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
            f" probed={result.probed_port_count} of declared={result.declared_port_count}"
            f" ({result.port_selection}) est_usable={result.estimated_usable_port_count}"
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
                    "probed_port_count": result.probed_port_count,
                    "declared_port_count": result.declared_port_count,
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

    @staticmethod
    async def _should_keep_last_known_sysbox(
        ctx: Context, result: PortVerificationResult, extra_info: dict[str, object]
    ) -> bool:
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
            return True
        if not settings.DIND_PROBE_FIRST_MISS_GRACE:
            return False
        # DAH-3597: a probe that never reached its container (dind_ok False: docker run
        # refused, sshd not up in 30 s, SSH reset) measured nothing about sysbox, so the first
        # such miss inside the TTL window keeps the last known value and the next cycle
        # re-measures. A second miss inside the window is recorded as before. A probe that
        # reached its container and still says no sysbox is a verdict, never tolerated.
        if sysbox_downgraded and not result.dind_ok:
            first_miss = await ctx.services.redis.record_dind_probe_miss(
                ctx.miner_hotkey,
                ctx.executor.uuid,
                settings.DIND_PROBE_FIRST_MISS_GRACE_TTL_SECONDS,
            )
            if first_miss:
                extra_info["sysbox_downgrade_tolerated_reason"] = "first_dind_probe_miss"
                return True
            extra_info["dind_probe_miss_repeated"] = True
        elif result.dind_ok:
            await ctx.services.redis.clear_dind_probe_miss(ctx.miner_hotkey, ctx.executor.uuid)
        return False
