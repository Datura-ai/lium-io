from __future__ import annotations

from datetime import datetime, timedelta

import asyncssh

from core.config import settings
from core.docker_utils import (
    ContainerDeathKind,
    DockerCommand,
    classify_container_death,
    collect_container_death_diagnostics,
    container_uptime_seconds,
)
from protocol.vc_protocol.compute_requests import (
    GPU_RUNTIME_NVML_MISMATCH_REASON,
    FillerRunActiveResponse,
)

from ...const import (
    FILLER_CONTAINER_PREFIX,
    FILLER_KILL_STRIKE_TTL_SECONDS,
    FILLER_LIVENESS_GRACE_MINUTES,
)
from ..messages import RentalVerificationMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class RentalVerificationCheck:
    """Verify executor rental status via backend API health check.

    This check calls the backend API to verify that the executor can be successfully
    rented and is healthy. This is an additional verification step beyond checking
    the Redis RENTAL_SUCCEED_MACHINE_SET that provides real-time rental verification.
    """

    check_id = "executor.validate.rental_verification"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        """Run rental verification check via backend API.

        Args:
            ctx: Pipeline context

        Returns:
            CheckResult with verification status
        """
        # Skip if rental verification is disabled
        if settings.SKIP_RENTAL_VERIFICATION:
            event = render_message(
                Msg.SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={"skipped": True},
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={},
            )

        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        has_customer_rental = bool(rented_executor and rented_executor.pods)
        filler_container = rented_data.get_filler_container(ctx.executor.uuid) if rented_data else None
        if filler_container and not has_customer_rental:
            # ISSUE-050: the backend's word alone is not proof the filler is alive — some hosts
            # remove Lium filler containers while the run stays RUNNING and keep earning
            # unrented incentive. Verify the container on the host, mirroring the customer-pod
            # flow in RentedMachineCheck (SSH liveness probe + live backend re-check on miss).
            # Rollout is gated: CHECK_ENABLED is the master switch (shadow: observe and log only),
            # ENFORCEMENT additionally withholds incentive.
            if not settings.FILLER_LIVENESS_CHECK_ENABLED:
                event = render_message(
                    Msg.SKIPPED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "skipped": True,
                        "reason": "active filler runtime",
                        "filler_container": filler_container,
                    },
                )
                return CheckResult(passed=True, event=event, updates={})

            return await self._verify_filler_alive(
                ctx, filler_container, enforce=settings.FILLER_LIVENESS_ENFORCEMENT_ENABLED
            )

        # Get required info from context
        backend_client = ctx.services.backend
        executor = ctx.executor
        miner_hotkey = ctx.miner_hotkey

        # Get verified ports from PortConnectivityCheck
        verified_ports = ctx.state.specs.get("verified_ports", []) if ctx.state.specs else []

        # Fail if no verified ports are available (safety check - should be caught by PortConnectivityCheck)
        if not verified_ports:
            event = render_message(
                Msg.FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "verified": False,
                    "executor_uuid": executor.uuid,
                    "error": "No verified ports available for rental verification",
                },
                remediation="Port connectivity check should have failed - this is a safety check",
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={},
            )

        # Use the first verified port
        container_port = verified_ports[0]

        try:
            # Call backend API to verify executor health. Pass the rental hint: when this validator
            # already sees an active customer rental, the backend skips the container-creating check
            # instead of disturbing the tenant (it still re-checks the DB itself when this is False).
            response = await backend_client.check_executor_health(
                miner_address=ctx.miner_address,
                miner_port=ctx.miner_port,
                miner_hotkey=miner_hotkey,
                container_port=container_port,
                executor_id=executor.uuid,
                rental_in_progress=has_customer_rental,
            )

            # Handle API failure (None response) - fail this executor
            if response is None:
                event = render_message(
                    Msg.API_ERROR,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "error": "API returned None",
                        "executor_uuid": executor.uuid,
                    },
                )
                return CheckResult(
                    passed=False,  # Fail this executor, continue with others
                    event=event,
                    updates={},
                )

            # Check if verification was successful
            if response.success:
                event = render_message(
                    Msg.VERIFIED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "verified": True,
                        "executor_uuid": executor.uuid,
                        "details": response.details or {},
                    },
                )
                return CheckResult(
                    passed=True,
                    event=event,
                    updates={},
                )
            elif response.reason_code == GPU_RUNTIME_NVML_MISMATCH_REASON:
                details = response.details or {}
                stderr = details.get("docker_stderr") or response.error
                event = render_message(
                    Msg.GPU_RUNTIME_NVML_MISMATCH,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "verified": False,
                        "executor_uuid": executor.uuid,
                        "source": "rental_verification",
                        "reason_code": response.reason_code,
                        "stderr": stderr,
                        "details": details,
                    },
                    extra={"gpu_runtime_issue_code": response.reason_code},
                )
                return CheckResult(
                    passed=False,
                    event=event,
                    updates={
                        "score": 0.0,
                        "job_score": 0.0,
                        "score_warning": "GPU runtime NVML driver/library mismatch",
                        "clear_verified_job_info": True,
                    },
                )
            else:
                # Verification failed - this is fatal
                event = render_message(
                    Msg.FAILED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "verified": False,
                        "executor_uuid": executor.uuid,
                        "error": response.error or "Unknown error",
                        "details": response.details or {},
                    },
                )
                return CheckResult(
                    passed=False,  # Fatal check - halt validation
                    event=event,
                    updates={},
                )

        except Exception as e:
            # Handle unexpected errors - fail this executor
            event = render_message(
                Msg.API_ERROR,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "error": str(e),
                    "executor_uuid": executor.uuid,
                },
            )
            return CheckResult(
                passed=False,  # Fail this executor, continue with others
                event=event,
                updates={},
            )

        finally:
            # DAH-1991: force-remove the health_check_* probe the backend just
            # spawned so it cannot race a subsequent rental on this executor.
            await ctx.services.container_cleanup.force_remove_health_checks(
                ctx.ssh, ctx.executor.uuid
            )

    async def _verify_filler_alive(
        self, ctx: Context, filler_container: str, *, enforce: bool
    ) -> CheckResult:
        """Verify the Lium filler container actually runs on the host.

        In shadow mode (enforce=False) every outcome keeps passed=True so scoring is
        unchanged; the emitted events are the observable artifact. Only a confirmed
        kill (container gone, run still RUNNING past grace) fails in enforcement mode.
        """
        try:
            ps_result: asyncssh.SSHCompletedProcess = await ctx.ssh.run(
                DockerCommand.ps_running(filler_container)
            )
        except (asyncssh.Error, OSError) as exc:
            # SSH transport died mid-cycle: filler state is unknown, do not re-check.
            event = render_message(
                Msg.FILLER_TRANSPORT_UNREACHABLE,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "filler_container": filler_container,
                    "executor_uuid": ctx.executor.uuid,
                    "transport_error": repr(exc),
                    "enforced": enforce,
                },
            )
            return CheckResult(passed=not enforce, event=event, updates={})

        if ps_result.exit_status != 0:
            # docker daemon error (e.g. restarting) — indistinguishable from a kill, so fail open.
            return self._filler_state_unknown_result(
                ctx,
                filler_container,
                reason="docker ps failed on host",
                details={"exit_status": ps_result.exit_status},
            )

        filler_running: bool = bool(ps_result.stdout.strip())
        if filler_running:
            event = render_message(
                Msg.FILLER_VERIFIED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "verified": True,
                    "filler_container": filler_container,
                    "executor_uuid": ctx.executor.uuid,
                },
            )
            return CheckResult(passed=True, event=event, updates={})

        # Container is missing: the rented_data snapshot may be stale (the run was stopped or is
        # still starting mid-cycle) — re-check the live run state before penalizing, like the pod
        # flow does with get_pod_rental_active.
        filler_run_id: str = filler_container.removeprefix(FILLER_CONTAINER_PREFIX)
        filler_run: FillerRunActiveResponse | None = await ctx.services.backend.get_filler_run_active(
            filler_run_id
        )

        if filler_run is None:
            return self._filler_state_unknown_result(
                ctx, filler_container, reason="filler-run re-check API unavailable"
            )

        if not filler_run.active:
            # Not RUNNING right now: either mid-transition (STARTING/STOPPING — the snapshot also
            # lists those) or already terminal. Nothing provably wrong on the host — pass and let
            # the next cycle see the settled state.
            return self._filler_state_unknown_result(
                ctx,
                filler_container,
                reason="filler run is not in RUNNING state",
                details={"filler_run_status": filler_run.status},
            )

        run_age: timedelta | None = (
            datetime.utcnow() - filler_run.started_at if filler_run.started_at else None
        )
        if run_age is None or run_age < timedelta(minutes=FILLER_LIVENESS_GRACE_MINUTES):
            return self._filler_state_unknown_result(
                ctx,
                filler_container,
                reason="filler run within startup grace window",
                details={"run_age_seconds": run_age.total_seconds() if run_age else None},
            )

        return await self._filler_killed_result(
            ctx,
            filler_container,
            filler_run_status=filler_run.status,
            run_age=run_age,
            enforce=enforce,
        )

    async def _filler_killed_result(
        self,
        ctx: Context,
        filler_container: str,
        *,
        filler_run_status: str | None,
        run_age: timedelta,
        enforce: bool,
    ) -> CheckResult:
        """Classify WHY the container is dead; only an external kill may cost incentive.

        REMOVED (container gone) is punishable outright — nothing legitimate deletes a
        RUNNING run's container. STOPPED (SIGTERM/SIGKILL) could in theory be the worker
        exiting 143 itself, so the first incident per executor is a logged strike and only
        repeat incidents within the strike window are punished. Every other death kind
        (self-crash, OOM, never started, clean exit, unknown) is self-heal territory and
        never punished.
        """
        try:
            diagnostics = await collect_container_death_diagnostics(ctx.ssh, filler_container)
            death_fields: dict[str, object] = diagnostics.to_log_fields()
            death_kind = classify_container_death(diagnostics)
            uptime_seconds = container_uptime_seconds(diagnostics.started_at, diagnostics.finished_at)
        except Exception as exc:
            death_fields = {"diagnostics_capture_error": repr(exc)}
            death_kind = ContainerDeathKind.UNKNOWN
            uptime_seconds = None

        kill_timing: str | None = None
        if uptime_seconds is not None:
            grace_seconds = FILLER_LIVENESS_GRACE_MINUTES * 60
            kill_timing = "at_start" if uptime_seconds < grace_seconds else "after_running"

        common_what: dict[str, object] = {
            "filler_container": filler_container,
            "executor_uuid": ctx.executor.uuid,
            "filler_run_status": filler_run_status,
            "run_age_seconds": run_age.total_seconds(),
            "death_kind": death_kind.value,
            "container_uptime_seconds": uptime_seconds,
            "kill_timing": kill_timing,
            **death_fields,
        }

        if death_kind is ContainerDeathKind.REMOVED:
            return self._external_kill_result(ctx, enforce=enforce, what=common_what)

        if death_kind is ContainerDeathKind.STOPPED:
            strikes = await self._register_kill_strike(ctx, filler_container)
            common_what["kill_strikes"] = strikes
            if strikes is None or strikes >= settings.FILLER_KILL_STRIKE_THRESHOLD:
                return self._external_kill_result(ctx, enforce=enforce, what=common_what)
            event = render_message(
                Msg.FILLER_KILL_SUSPECTED,
                ctx=ctx,
                check_id=self.check_id,
                what=common_what,
            )
            return CheckResult(passed=True, event=event, updates={})

        # SELF_CRASHED / OOM_KILLED / NEVER_STARTED / CLEAN_EXIT / UNKNOWN: not the owner's kill.
        event = render_message(
            Msg.FILLER_CRASHED,
            ctx=ctx,
            check_id=self.check_id,
            what=common_what,
        )
        return CheckResult(passed=True, event=event, updates={})

    async def _register_kill_strike(self, ctx: Context, filler_container: str) -> int | None:
        """One strike per filler run (deduped in Redis); None when Redis is unavailable.

        None is treated as strikes-reached by the caller: an ambiguous stop with no working
        strike storage falls back to the strict verdict rather than a free pass forever.
        """
        filler_run_id: str = filler_container.removeprefix(FILLER_CONTAINER_PREFIX)
        redis_service = ctx.services.redis
        if redis_service is None:
            return None
        try:
            return await redis_service.register_filler_kill_strike(
                ctx.executor.uuid, filler_run_id, FILLER_KILL_STRIKE_TTL_SECONDS
            )
        except Exception:
            return None

    def _external_kill_result(
        self, ctx: Context, *, enforce: bool, what: dict[str, object]
    ) -> CheckResult:
        event = render_message(
            Msg.FILLER_KILLED,
            ctx=ctx,
            check_id=self.check_id,
            severity=None if enforce else "warning",
            impact=None if enforce else "Shadow observation only: incentive was NOT withheld",
            what={"verified": False, "enforced": enforce, **what},
        )
        return CheckResult(passed=not enforce, event=event, updates={})

    def _filler_state_unknown_result(
        self,
        ctx: Context,
        filler_container: str,
        *,
        reason: str,
        details: dict[str, object] | None = None,
    ) -> CheckResult:
        event = render_message(
            Msg.FILLER_STATE_UNKNOWN,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "filler_container": filler_container,
                "executor_uuid": ctx.executor.uuid,
                "reason": reason,
                **(details or {}),
            },
        )
        return CheckResult(passed=True, event=event, updates={})
