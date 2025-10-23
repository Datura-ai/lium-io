from ..models import build_msg
from ..pipeline import CheckResult, Context
from services.const import GPU_MEMORY_UTILIZATION_LIMIT, GPU_UTILIZATION_LIMIT


class GpuUsageCheck:
    """Re-use the legacy GPU utilisation guard for both rented and idle states."""

    check_id = "gpu.validate.usage"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        gpu_details = ctx.state.gpu_details
        gpu_processes = ctx.state.gpu_processes

        violation = _find_violation(gpu_details, gpu_processes)

        if violation is None:
            event = build_msg(
                event="GPU usage within limits",
                reason="GPU_USAGE_OK",
                severity="info",
                category="runtime",
                impact="Proceed",
                what={"process_count": len(gpu_processes)},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=True, event=event)

        event = build_msg(
            event="GPU busy outside validator",
            reason="GPU_USAGE_HIGH",
            severity="warning",
            category="runtime",
            impact="Validation skipped; score set to 0",
            remediation="Stop all GPU processes and re-run your node. If using Docker, ensure no host processes are running.",
            what=violation,
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=False, event=event)


def _find_violation(gpu_details: list[dict], gpu_processes: list[dict]) -> dict | None:
    if not gpu_processes:
        return None

    for detail in gpu_details:
        gpu_utilization = detail.get("gpu_utilization", GPU_UTILIZATION_LIMIT)
        gpu_memory_utilization = detail.get("memory_utilization", GPU_MEMORY_UTILIZATION_LIMIT)

        if gpu_utilization >= GPU_UTILIZATION_LIMIT or gpu_memory_utilization > GPU_MEMORY_UTILIZATION_LIMIT:
            return {
                "gpu_utilization": f"{gpu_utilization}%",
                "vram_utilization": f"{gpu_memory_utilization}%",
                "process_count": len(gpu_processes),
                "gpu_processes": gpu_processes,
            }

    return None
