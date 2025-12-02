from typing import Iterable
from dataclasses import replace
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason

from ..messages import TenantEnforcementMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from ...const import (
    GPU_MEMORY_UTILIZATION_LIMIT,
    GPU_UTILIZATION_LIMIT,
    MIN_PORT_COUNT,
)


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
        redis_service = ctx.services.redis
        rented_machine = await redis_service.get_rented_machine(ctx.executor)

        if not rented_machine or not rented_machine.get("containers", None):
            extra = {**ctx.default_extra, "rented": False}
            event = render_message(
                Msg.NOT_RENTED,
                ctx=ctx,
                check_id=self.check_id,
                what={"executor_uuid": ctx.executor.uuid},
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

        rented_pods = rented_machine.get("containers", [])
        extra = {
            **ctx.default_extra,
            "rented": True,
            "rented_pods": rented_pods,
        }

        for pod in rented_pods:
            container_name = pod.get("name", "")
            pod_id = pod.get("pod_id", "")
            pod_running, ssh_pub_keys = await _check_pod_running(ctx.ssh, container_name)
            if not pod_running:
                event = render_message(
                    Msg.POD_NOT_RUNNING,
                    ctx=ctx,
                    check_id=self.check_id,
                    remediation=f"Start container {container_name} and ensure it stays healthy.",
                    what={
                        "pod_id": pod_id,
                        "container_name": container_name,
                        "executor_uuid": ctx.executor.uuid,
                    },
                    extra=extra
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

        container_names = [pod.get("name", "") for pod in rented_pods]
        gpu_processes = list(ctx.state.gpu_processes)
        gpu_running_outside = _has_gpu_process_outside_container(container_names, gpu_processes)

        if not rented_machine.get("owner_flag", False) and gpu_running_outside:
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

        port_count = await _compute_port_count(ctx)
                
        extra = {
            **extra,
            "available_port_count": port_count,
        }
        
        updated_state = replace(
            ctx.state,
            specs={
                **ctx.state.specs,
                "available_port_count": port_count,
            },
        )

        score_calculator = ctx.services.score_calculator
        actual_score, job_score, warning_message = score_calculator(
            ctx.state.gpu_model or "",
            ctx.collateral_deposited,
            ctx.is_rental_succeed,
            ctx.contract_version or "",
            True,
            port_count,
        )

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
                "state": updated_state,
                "default_extra": extra,
                "rented": True,
                "ssh_pub_keys": ssh_pub_keys,
                "port_count": port_count,
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
    try:
        ps_result = await ssh_client.run(f"/usr/bin/docker ps -q -f name={container_name}")
        pod_running = bool(ps_result.stdout.strip())
    except Exception:
        pod_running = False

    try:
        keys_result = await ssh_client.run(
            f"/usr/bin/docker exec -i {container_name} sh -c 'cat ~/.ssh/authorized_keys'"
        )
        ssh_keys = keys_result.stdout.strip().split("\n") if keys_result.stdout else []
    except Exception:
        ssh_keys = []

    return pod_running, ssh_keys


async def _compute_port_count(ctx: Context) -> int:
    port_mapping = ctx.services.port_mapping
    executor_uuid = ctx.executor.uuid

    try:
        port_count = await port_mapping.get_successful_ports_count(executor_uuid)
    except Exception:
        port_count = 0

    return port_count
