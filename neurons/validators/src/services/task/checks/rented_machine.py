from typing import Iterable
from dataclasses import replace
import logging
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

from core.docker_utils import DockerCommand
from ..messages import TenantEnforcementMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from ...const import (
    GPU_MEMORY_UTILIZATION_LIMIT,
    GPU_UTILIZATION_LIMIT,
    MIN_PORT_COUNT,
    POD_CONTAINER_PREFIX,
)

logger = logging.getLogger(__name__)


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
        # Clean up stale containers
        await cleanup_stale_containers(
            ssh_client=ctx.ssh,
            rented_data=ctx.state.rented_data,
            executor_uuid=ctx.executor.uuid,
            stale_threshold_minutes=15,
        )

        # Get rented executor from context instead of Redis
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None

        if not rented_executor or not rented_executor.pods:
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

        rented_pods = rented_executor.pods
        extra = {
            **ctx.default_extra,
            "rented": True,
            "rented_pods": [{"name": p.container_name, "pod_id": p.pod_id} for p in rented_pods],
        }

        for pod in rented_pods:
            pod_container_name = pod.container_name
            pod_id = pod.pod_id
            pod_running, ssh_pub_keys = await _check_pod_running(ctx.ssh, pod_container_name)
            if not pod_running:
                event = render_message(
                    Msg.POD_NOT_RUNNING,
                    ctx=ctx,
                    check_id=self.check_id,
                    remediation=f"Start container {pod_container_name} and ensure it stays healthy.",
                    what={
                        "pod_id": pod_id,
                        "container_name": pod_container_name,
                        "executor_uuid": ctx.executor.uuid,
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
    try:
        ps_result = await ssh_client.run(DockerCommand.ps_running(container_name))
        pod_running = bool(ps_result.stdout.strip())
    except Exception:
        pod_running = False

    try:
        keys_result = await ssh_client.run(
            DockerCommand.exec_command(container_name, 'cat ~/.ssh/authorized_keys')
        )
        ssh_keys = keys_result.stdout.strip().split("\n") if keys_result.stdout else []
    except Exception:
        ssh_keys = []

    return pod_running, ssh_keys


async def cleanup_stale_containers(
    ssh_client,
    rented_data: RentedExecutorsResponse | None,
    executor_uuid: str,
    stale_threshold_minutes: int = 15,
) -> tuple[int, list[str]]:
    """Remove containers that are not in rented data and are older than threshold.

    Returns:
        Tuple of (number_removed, list_of_removed_container_names)
    """
    removed_names = []

    try:
        # Get all containers with the rental prefix
        result = await ssh_client.run(DockerCommand.ps_filter(f"{POD_CONTAINER_PREFIX}*"))
        if not result.stdout or not result.stdout.strip():
            return 0, []

        all_pod_containers = result.stdout.strip().split('\n')

        # Get currently rented containers for this executor
        rented_containers = set()
        if rented_data and executor_uuid in rented_data.executors:
            executor = rented_data.executors[executor_uuid]
            rented_containers = {pod.container_name for pod in executor.pods}

        # Check each container
        for container_name in all_pod_containers:
            if container_name in rented_containers:
                continue

            # Get container creation time and calculate age on the machine
            try:
                created_result = await ssh_client.run(DockerCommand.inspect_created_timestamp(container_name))
                if created_result.exit_status != 0:
                    continue
                created_timestamp = int(created_result.stdout.strip())

                # Get current time on the machine
                current_result = await ssh_client.run("date +%s")
                if current_result.exit_status != 0:
                    continue
                current_timestamp = int(current_result.stdout.strip())

                # Check if stale
                age_minutes = (current_timestamp - created_timestamp) / 60
                if age_minutes > stale_threshold_minutes:
                    # Remove container
                    remove_result = await ssh_client.run(DockerCommand.remove(container_name))
                    if remove_result.exit_status == 0:
                        removed_names.append(container_name)

                        # Remove associated volume
                        pod_id = container_name.removeprefix(POD_CONTAINER_PREFIX)
                        await ssh_client.run(DockerCommand.volume_remove(f"volume_{pod_id}"))

            except Exception:
                continue

    except Exception:
        pass

    return len(removed_names), removed_names
