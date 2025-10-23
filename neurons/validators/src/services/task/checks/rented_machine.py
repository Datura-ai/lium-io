from __future__ import annotations

from typing import Awaitable, Callable, Iterable

from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason

from ..models import build_msg
from ..pipeline import CheckResult, Context
from ...const import GPU_MEMORY_UTILIZATION_LIMIT, GPU_UTILIZATION_LIMIT


def _has_gpu_process_outside_container(container_name: str, processes: Iterable[dict]) -> bool:
    """True when any process is missing a container or belongs to a different container."""
    for process in processes:
        process_container = process.get("container_name")
        if not process_container or process_container != container_name:
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

    def __init__(
        self,
        *,
        rented_machine_fetcher: Callable[[], Awaitable[dict | None]],
        pod_checker: Callable[[object, str, object], Awaitable[tuple[bool, list[str]]]],
        port_counter: Callable[[str, str], Awaitable[int]],
        score_calculator: Callable[[str, bool, bool, str, bool, int], tuple[float, float, str]],
    ):
        self.rented_machine_fetcher = rented_machine_fetcher
        self.pod_checker = pod_checker
        self.port_counter = port_counter
        self.score_calculator = score_calculator

    async def run(self, ctx: Context) -> CheckResult:
        rented_machine = await self.rented_machine_fetcher()

        if not rented_machine or not rented_machine.get("container_name"):
            extra = {**ctx.default_extra, "rented": False}
            event = build_msg(
                event="Executor not rented",
                reason="EXECUTOR_NOT_RENTED",
                severity="info",
                category="policy",
                impact="Proceed",
                what={"executor_uuid": ctx.executor.uuid},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
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

        container_name = rented_machine.get("container_name", "")
        extra = {
            **ctx.default_extra,
            "rented": True,
            "container_name": container_name,
        }

        pod_running, ssh_pub_keys = await self.pod_checker(ctx.ssh, container_name, ctx.executor)
        if not pod_running:
            event = build_msg(
                event="Pod not running",
                reason="POD_NOT_RUNNING",
                severity="error",
                category="runtime",
                impact="Score set to 0; verification cleared",
                remediation=f"Start container {container_name} and ensure it stays healthy.",
                what={
                    "container": container_name,
                    "executor_uuid": ctx.executor.uuid,
                },
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
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

        gpu_processes = list(ctx.gpu_processes or [])
        gpu_running_outside = _has_gpu_process_outside_container(container_name, gpu_processes)

        if not rented_machine.get("owner_flag", False) and gpu_running_outside:
            gpu_details = ctx.gpu_details or []
            if not _is_gpu_usage_within_limits(gpu_details, gpu_processes):
                observation = _gpu_usage_violation_details(gpu_details, gpu_processes)
                event = build_msg(
                    event="Tenant container does not own GPU",
                    reason="GPU_USAGE_OUTSIDE_TENANT",
                    severity="warning",
                    category="runtime",
                    impact="Validation failed; score set to 0",
                    remediation="Terminate host-level GPU processes, make sure nvidia-smi doesn't show any running processes.",
                    what={
                        "expected_container": container_name,
                        "process_count": observation["process_count"],
                        "gpu_utilization": observation["gpu_utilization"],
                        "vram_utilization": observation["vram_utilization"],
                        "gpu_processes": gpu_processes,
                    },
                    check_id=self.check_id,
                    ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
                )
                return CheckResult(
                    passed=False,
                    event=event,
                    updates={"default_extra": extra, "ssh_pub_keys": ssh_pub_keys},
                )

        port_count = await self.port_counter(ctx.miner_hotkey, ctx.executor.uuid)

        actual_score, job_score, warning_message = self.score_calculator(
            ctx.gpu_model or "",
            ctx.collateral_deposited,
            ctx.is_rental_succeed,
            ctx.contract_version or "",
            True,
            port_count,
        )

        event = build_msg(
            event="Executor already rented",
            reason="RENTED",
            severity="info",
            category="policy",
            impact=f"Reported rented score={job_score} (actual={actual_score})",
            remediation=f"No action needed.{warning_message}" if warning_message else "No action needed.",
            what={
                "contract_version": ctx.contract_version,
                "collateral": ctx.collateral_deposited,
                "actual_score": actual_score,
                "job_score": job_score,
            },
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )

        return CheckResult(
            passed=True,
            event=event,
            updates={
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
