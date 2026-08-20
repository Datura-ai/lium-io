from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import asyncssh

from core.config import settings
from core.utils import _m, get_extra_info
from core.docker_utils import DockerCommand, collect_container_death_diagnostics
from protocol.vc_protocol.compute_requests import (
    CPU_QUOTA_EXCEEDS_HOST_REASON,
    GPU_RUNTIME_DEVICE_FAULT_REASON,
    GPU_RUNTIME_NVML_MISMATCH_REASON,
    FillerRunActiveResponse,
)

from ...const import FILLER_CONTAINER_PREFIX, FILLER_LIVENESS_GRACE_MINUTES
from ..messages import MessageTemplate, render_message
from ..messages import RentalVerificationMessages as Msg
from ..pipeline import CheckResult, Context
from .cpu_truth import advertised_cpu_count

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuRuntimeQuarantine:
    message: MessageTemplate
    score_warning: str


# Backend verdicts that quarantine the host: the GPU runtime is broken in a way no rental survives,
# so the score is zeroed and the verified job cleared. The host returns on the next clean check.
_GPU_RUNTIME_QUARANTINE: dict[str, GpuRuntimeQuarantine] = {
    GPU_RUNTIME_NVML_MISMATCH_REASON: GpuRuntimeQuarantine(
        message=Msg.GPU_RUNTIME_NVML_MISMATCH,
        score_warning="GPU runtime NVML driver/library mismatch",
    ),
    GPU_RUNTIME_DEVICE_FAULT_REASON: GpuRuntimeQuarantine(
        message=Msg.GPU_RUNTIME_DEVICE_FAULT,
        score_warning="GPU is not addressable and needs a host reset",
    ),
}

# How bad each per-container filler verdict is, worst wins. A GPU-split node yields one verdict per
# bundle but the pipeline emits a single event, so the node must be judged by its worst container:
# a confirmed kill outranks an indeterminate probe, which outranks a healthy one. Anything
# unlisted sorts as healthy.
_FILLER_VERDICT_SEVERITY: dict[str, int] = {
    Msg.FILLER_KILLED.reason: 3,
    Msg.FILLER_TRANSPORT_UNREACHABLE.reason: 2,
    Msg.FILLER_STATE_UNKNOWN.reason: 1,
    Msg.FILLER_VERIFIED.reason: 0,
}


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
        # EVERY filler container on the node, not just one: a GPU-split DPHN node runs one filler per
        # VRAM bundle (DAH-2465), and checking only the first left the siblings unverifiable — a
        # removed sibling was never observed, so it was neither punished nor reconciled (DAH-2471).
        filler_containers: list[str] = rented_data.get_filler_containers(ctx.executor.uuid) if rented_data else []
        # DAH-2703: same offence, earlier kill. A host reaper that removes the filler seconds after
        # `docker run` leaves no RUNNING run and no container, so the liveness probe below never
        # sees it and the node collects unrented incentive while never running a default job. The
        # backend counts those create-time kills per executor and lists the ones past the streak
        # threshold. Judged BEFORE the liveness probe on purpose: a GPU-split node runs one filler
        # per bundle, and one surviving bundle would otherwise return a clean verdict for a host
        # that destroys every other bundle it is given.
        create_killed: bool = bool(
            rented_data and ctx.executor.uuid in rented_data.filler_create_kill_executor_ids
        )
        if create_killed and not has_customer_rental and settings.FILLER_LIVENESS_CHECK_ENABLED:
            if settings.FILLER_LIVENESS_ENFORCEMENT_ENABLED:
                event = render_message(
                    Msg.FILLER_KILLED_AT_CREATE,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={"verified": False, "executor_uuid": ctx.executor.uuid, "enforced": True},
                )
                return CheckResult(passed=False, event=event, updates={})
            # Shadow: observe and fall through to the normal verification. Returning a pass here
            # would hand the node a free pass — it skips the backend health check the node would
            # otherwise have to survive, so shadow mode would be an upgrade, not an observation.
            # A check owns one event and the fall-through path writes it, so the shadow verdict
            # rides a log line carrying the same reason code the enforced event uses.
            logger.warning(
                _m(
                    "Lium default job is destroyed during create (shadow)",
                    extra=get_extra_info({
                        "reason_code": Msg.FILLER_KILLED_AT_CREATE.reason,
                        "executor_uuid": ctx.executor.uuid,
                        "enforced": False,
                    }),
                )
            )

        if filler_containers and not has_customer_rental:
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
                        "filler_containers": filler_containers,
                    },
                )
                return CheckResult(passed=True, event=event, updates={})

            return await self._verify_every_filler_alive(
                ctx, filler_containers, enforce=settings.FILLER_LIVENESS_ENFORCEMENT_ENABLED
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

        # The UUIDs this cycle's scrape saw. The backend probes for exactly these instead of
        # `--gpus all`, so a host advertising cards it cannot hand over fails here rather than in a
        # customer's rental (DAH-2614). Sent from the scrape, not read from the backend's stored
        # specs: those lag a cycle and do not exist at all on an executor's first pass.
        gpu_details = ctx.state.gpu_details or (ctx.state.specs or {}).get("gpu", {}).get("details", [])
        gpu_uuids = [detail["uuid"] for detail in gpu_details if isinstance(detail, dict) and detail.get("uuid")]

        # DAH-2671 item 2b: hand the backend the advertised core count AS-IS so the probe is created
        # with `--cpus=<advertised>`. The backend classifier compares it against the daemon's own N
        # from the refusal message and only classifies CPU_QUOTA_EXCEEDS_HOST when
        # advertised > 2*N + 4 (honest max divergence is 2x — SMT off; observed spoofs 4-8x), so an
        # honest SMT-off host falls into the backend's retry-without-flag path. No host-read capping:
        # any source read from the judged host (sysfs/procfs) is spoofable, and a cap computed from
        # it would let a spoofer neutralise the daemon check (fake `present` == advertised while
        # `/proc/cpuinfo` stays real → cap = online → daemon accepts). Skipped on TDX hosts (a real
        # rent skips `--cpus` there too) and when the count is unreadable.
        cpu_count = advertised_cpu_count(ctx) if settings.RENTAL_CPU_LIMIT_CHECK_ENABLED and not ctx.executor.tdx_quote else None

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
                gpu_uuids=gpu_uuids,
                cpu_count=cpu_count,
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
            elif response.reason_code == CPU_QUOTA_EXCEEDS_HOST_REASON:
                return self._cpu_quota_verdict(ctx, response)
            elif response.reason_code in _GPU_RUNTIME_QUARANTINE:
                quarantine = _GPU_RUNTIME_QUARANTINE[response.reason_code]
                details = response.details or {}
                stderr = details.get("docker_stderr") or response.error
                event = render_message(
                    quarantine.message,
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
                        "score_warning": quarantine.score_warning,
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

    def _cpu_quota_verdict(self, ctx: Context, response) -> CheckResult:
        """Render the CPU-quota verdict: the host's daemon refused `--cpus=<advertised>` at create.

        Shadow (enforcement off) logs the verdict as a warning and keeps passed=True so scoring is
        unchanged; enforcement zeroes the score and clears the verified job, the same quarantine the
        GPU runtime faults use.
        """
        enforce = settings.RENTAL_CPU_LIMIT_ENFORCEMENT_ENABLED
        details = response.details or {}
        stderr = details.get("docker_stderr") or response.error
        event = render_message(
            Msg.CPU_QUOTA_EXCEEDS_HOST,
            ctx=ctx,
            check_id=self.check_id,
            severity=None if enforce else "warning",
            impact=None if enforce else "Shadow observation only: score was NOT changed",
            what={
                "verified": False,
                "executor_uuid": ctx.executor.uuid,
                "source": "rental_verification",
                "reason_code": response.reason_code,
                "advertised_cpu_count": advertised_cpu_count(ctx),
                "stderr": stderr,
                "enforced": enforce,
                "details": details,
            },
        )
        updates = (
            {
                "score": 0.0,
                "job_score": 0.0,
                "score_warning": "Advertised CPU cores exceed the host's physical cores",
                "clear_verified_job_info": True,
            }
            if enforce
            else {}
        )
        return CheckResult(passed=not enforce, event=event, updates=updates)

    async def _verify_every_filler_alive(
        self, ctx: Context, filler_containers: list[str], *, enforce: bool
    ) -> CheckResult:
        """Verify each of the node's filler containers, one per GPU bundle (DAH-2465).

        Every container is probed independently, so a removed bundle is reported to the backend
        (container_missing=true, feeding the DAH-2471 reconciliation) even while its siblings run
        fine. A check may return only one result, so the node's verdict is its WORST container by
        _FILLER_VERDICT_SEVERITY. Picking positionally would let a healthy or unreachable sibling
        hide a kill: in shadow every verdict passes, so a killed non-last bundle would emit no
        FILLER_KILLED at all, and under enforcement an early SSH failure would decide the node
        while a real kill behind it went unlogged. Each container's reason code is also listed on
        the returned event, so one log line still covers the whole node.
        """
        verdicts: list[tuple[str, CheckResult]] = [
            (filler_container, await self._verify_filler_alive(ctx, filler_container, enforce=enforce))
            for filler_container in filler_containers
        ]
        decisive: CheckResult = max(
            verdicts, key=lambda verdict: _FILLER_VERDICT_SEVERITY.get(verdict[1].event.reason_code, 0)
        )[1]
        decisive.event.what_we_saw["checked_containers"] = [
            {"filler_container": filler_container, "reason_code": verdict.event.reason_code}
            for filler_container, verdict in verdicts
        ]
        return decisive

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
        # container_missing carries the observation to the backend: this re-check only ever happens
        # when the container is gone from the host, and the backend uses the accumulated reports to
        # close zombie RUNNING runs (DAH-2471).
        filler_run: FillerRunActiveResponse | None = await ctx.services.backend.get_filler_run_active(
            filler_run_id, container_missing=True
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
        """Judge why the container is not running, and emit FILLER_KILLED only for a removal.

        DAH-2730: `docker ps` lists RUNNING containers only, so a filler that exited on its own
        looks exactly like one the host removed. The death diagnostics already know the difference
        — a removed container has no state at all — and only a removal is the offence this check
        withholds incentive for. An unprovable state (SSH lost, diagnostics failed) is never a
        penalty.
        """
        try:
            diagnostics = await collect_container_death_diagnostics(ctx.ssh, filler_container)
        except Exception as exc:
            return self._filler_state_unknown_result(
                ctx,
                filler_container,
                reason="container death diagnostics unavailable",
                details={"diagnostics_capture_error": repr(exc)},
            )

        death_fields: dict[str, object] = diagnostics.to_log_fields()
        if not diagnostics.container_missing:
            event = render_message(
                Msg.FILLER_CONTAINER_EXITED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "verified": False,
                    "filler_container": filler_container,
                    "executor_uuid": ctx.executor.uuid,
                    "filler_run_status": filler_run_status,
                    "run_age_seconds": run_age.total_seconds(),
                    "enforced": False,
                    **death_fields,
                },
            )
            return CheckResult(passed=True, event=event, updates={})

        event = render_message(
            Msg.FILLER_KILLED,
            ctx=ctx,
            check_id=self.check_id,
            severity=None if enforce else "warning",
            impact=None if enforce else "Shadow observation only: incentive was NOT withheld",
            what={
                "verified": False,
                "filler_container": filler_container,
                "executor_uuid": ctx.executor.uuid,
                "filler_run_status": filler_run_status,
                "run_age_seconds": run_age.total_seconds(),
                "enforced": enforce,
                **death_fields,
            },
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
