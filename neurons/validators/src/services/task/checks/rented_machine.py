from collections.abc import Iterable

import asyncssh

from core.docker_utils import DockerCommand, collect_container_death_diagnostics
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason

from ...const import (
    GPU_MEMORY_UTILIZATION_LIMIT,
    GPU_UTILIZATION_LIMIT,
)
from ..messages import TenantEnforcementMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context


def _has_gpu_process_outside_container(rented_pods: list[str], processes: Iterable[dict]) -> bool:
    """True when any process is missing a container or belongs to a different container."""
    for process in processes:
        process_container = process.get("container_name")
        if not process_container or process_container not in rented_pods:
            return True
    return False


def _is_gpu_usage_within_limits(gpu_details: Iterable[dict], gpu_processes: Iterable[dict]) -> bool:
    """True when utilisation metrics do not exceed protocol limits."""
    if not gpu_processes:
        return True

    for detail in gpu_details:
        utilisation = detail.get("gpu_utilization", GPU_UTILIZATION_LIMIT)
        memory_utilisation = detail.get("memory_utilization", GPU_MEMORY_UTILIZATION_LIMIT)

        if utilisation >= GPU_UTILIZATION_LIMIT or memory_utilisation > GPU_MEMORY_UTILIZATION_LIMIT:
            return False

    return True


def _gpu_usage_violation_details(
    gpu_details: Iterable[dict],
    gpu_processes: Iterable[dict],
) -> dict:
    """Prepare diagnostic data describing the observed GPU usage."""
    processes = list(gpu_processes)
    utilisation = None
    memory_utilisation = None

    for detail in gpu_details:
        utilisation = detail.get("gpu_utilization")
        memory_utilisation = detail.get("memory_utilization")

        exceeds_utilisation = utilisation is not None and utilisation >= GPU_UTILIZATION_LIMIT
        exceeds_memory = memory_utilisation is not None and memory_utilisation > GPU_MEMORY_UTILIZATION_LIMIT

        if exceeds_utilisation or exceeds_memory:
            break
    else:
        utilisation = None
        memory_utilisation = None

    utilisation_display = (
        f"{utilisation}%"
        if utilisation is not None
        else f">={GPU_UTILIZATION_LIMIT}%"
    )
    memory_display = (
        f"{memory_utilisation}%"
        if memory_utilisation is not None
        else f">{GPU_MEMORY_UTILIZATION_LIMIT}%"
    )

    return {
        "gpu_utilization": utilisation_display,
        "vram_utilization": memory_display,
        "process_count": len(processes),
    }


class TenantEnforcementCheck:
    """Handle the specialised flow when the executor is already rented to a tenant.

    The legacy code short-circuited out of validation in this scenario after checking pod
    health, GPU ownership, ports, and score adjustments. Keeping it as a single check
    documents that bespoke behaviour and ensures we still emit the historical log format.
    """

    check_id = "executor.validate.rented_state"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        # NOTE: stale-container cleanup now runs earlier, in StaleContainerCleanupCheck
        # (before the port checks), so an orphaned non-rented container can't deadlock
        # port verification. See that check for the full rationale (DAH-2164 follow-up).

        # Get rented executor from context instead of Redis
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        filler_container = rented_data.get_filler_container(ctx.executor.uuid) if rented_data else None

        if not rented_executor or not rented_executor.pods:
            extra = {**ctx.default_extra, "rented": False}
            if filler_container:
                extra["filler_container"] = filler_container

            what = {"executor_uuid": ctx.executor.uuid}
            if filler_container:
                what["filler_container"] = filler_container
            event = render_message(
                Msg.NOT_RENTED,
                ctx=ctx,
                check_id=self.check_id,
                what=what,
                extra=extra
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={
                    "rented": False,
                    "ssh_pub_keys": None,
                    "default_extra": extra,
                },
            )

        rented_pods = rented_executor.pods
        extra = {
            **ctx.default_extra,
            "rented": True,
            "rented_pods": [{"name": p.container_name, "pod_id": p.pod_id} for p in rented_pods],
        }

        for pod in rented_pods:
            pod_container_name = pod.container_name
            pod_id = pod.pod_id
            try:
                pod_running, ssh_pub_keys = await _check_pod_running(ctx.ssh, pod_container_name)
            except (asyncssh.Error, OSError) as exc:
                # SSH transport died mid-cycle. Pod state is unknown; do NOT set
                # clear_verified_job_*: that path triggers immediate executor flip
                # in compute-app and would punish honest miners for a network blip.
                # Persistent unreachability is handled by the stale-executor sweep.
                event = render_message(
                    Msg.EXECUTOR_TRANSPORT_UNREACHABLE,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "pod_id": pod_id,
                        "container_name": pod_container_name,
                        "executor_uuid": ctx.executor.uuid,
                        "transport_error": repr(exc),
                    },
                    extra=extra,
                )
                return CheckResult(
                    passed=False,
                    event=event,
                    updates={"default_extra": extra},
                )
            if not pod_running:
                diagnostics = await _collect_pod_diagnostics(ctx.ssh, pod_container_name)
                rental_active = await ctx.services.backend.get_pod_rental_active(pod_id)
                if rental_active and not rental_active.active:
                    event = render_message(
                        Msg.STALE_POD_NOT_RUNNING,
                        ctx=ctx,
                        check_id=self.check_id,
                        what={
                            "pod_id": pod_id,
                            "container_name": pod_container_name,
                            "executor_uuid": ctx.executor.uuid,
                            "rental_closed_at": (
                                rental_active.rental_closed_at.isoformat()
                                if rental_active.rental_closed_at
                                else None
                            ),
                            "diagnostics": diagnostics,
                        },
                        extra=extra,
                    )
                    return CheckResult(
                        passed=True,
                        event=event,
                        updates={
                            "default_extra": extra,
                            "rented": False,
                            "ssh_pub_keys": None,
                        },
                    )

                event = render_message(
                    Msg.POD_NOT_RUNNING,
                    ctx=ctx,
                    check_id=self.check_id,
                    remediation=f"Start container {pod_container_name} and ensure it stays healthy.",
                    what={
                        "pod_id": pod_id,
                        "container_name": pod_container_name,
                        "executor_uuid": ctx.executor.uuid,
                        "diagnostics": diagnostics,
                    },
                    extra=extra,
                )
                return CheckResult(
                    passed=False,
                    event=event,
                    updates={
                        "default_extra": extra,
                        "clear_verified_job_info": True,
                        "clear_verified_job_reason": ResetVerifiedJobReason.POD_NOT_RUNNING.value,
                    },
                )

        container_names = [pod.container_name for pod in rented_pods]
        if filler_container:
            container_names.append(filler_container)
        gpu_processes = list(ctx.state.gpu_processes)
        gpu_running_outside = _has_gpu_process_outside_container(container_names, gpu_processes)

        if not rented_executor.owner_flag and gpu_running_outside:
            gpu_details = ctx.state.gpu_details
            if not _is_gpu_usage_within_limits(gpu_details, gpu_processes):
                observation = _gpu_usage_violation_details(gpu_details, gpu_processes)
                event = render_message(
                    Msg.GPU_OUTSIDE_TENANT,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "expected_containers": container_names,
                        "process_count": observation["process_count"],
                        "gpu_utilization": observation["gpu_utilization"],
                        "vram_utilization": observation["vram_utilization"],
                        "gpu_processes": gpu_processes,
                    },
                    extra=extra
                )
                return CheckResult(
                    passed=False,
                    event=event,
                    updates={"default_extra": extra, "ssh_pub_keys": ssh_pub_keys},
                )

        score_calculator = ctx.services.score_calculator
        actual_score, job_score, warning_message = score_calculator(ctx, True)

        event = render_message(
            Msg.ALREADY_RENTED,
            ctx=ctx,
            check_id=self.check_id,
            impact=f"Reported rented score={job_score} (actual={actual_score})",
            remediation=f"No action needed.{warning_message}" if warning_message else "No action needed.",
            what={
                "contract_version": ctx.contract_version,
                "collateral": ctx.collateral_deposited,
                "actual_score": actual_score,
                "job_score": job_score,
            },
            extra=extra
        )

        return CheckResult(
            passed=True,
            event=event,
            updates={
                "default_extra": extra,
                "rented": True,
                "ssh_pub_keys": ssh_pub_keys,
                "score": actual_score,
                "job_score": job_score,
                "score_warning": warning_message or None,
                "log_status": "info",
                "log_text": event.event,
                "success": True,
            },
            halt=True,
        )


async def _check_pod_running(ssh_client, container_name: str) -> tuple[bool, list[str]]:
    # asyncssh.Error / OSError mean the SSH session itself is dead -> caller must
    # treat this as "pod state unknown", not "pod down". Other exceptions are
    # genuine docker / business failures and keep the legacy fall-through.
    try:
        ps_result = await ssh_client.run(DockerCommand.ps_running(container_name))
        pod_running = bool(ps_result.stdout.strip())
    except (asyncssh.Error, OSError):
        raise
    except Exception:
        pod_running = False

    try:
        # Literal path, not ~: injection writes this exact file (DAH-2534).
        keys_result = await ssh_client.run(
            DockerCommand.exec_command(container_name, "cat /root/.ssh/authorized_keys")
        )
        ssh_keys = keys_result.stdout.strip().split("\n") if keys_result.stdout else []
    except (asyncssh.Error, OSError):
        raise
    except Exception:
        ssh_keys = []

    return pod_running, ssh_keys


async def _collect_pod_diagnostics(
    ssh_client: asyncssh.SSHClientConnection, container_name: str
) -> dict[str, object]:
    """Diagnostics for a pod the ps-running check reported as down (run-time death).

    Shares one collector and one flat Loki field shape with the create-time
    failure path in docker_service (DAH-2395 / DAH-2193), so container deaths at
    create time and run time are queryable together.
    """
    diagnostics = await collect_container_death_diagnostics(ssh_client, container_name)
    return diagnostics.to_log_fields()
