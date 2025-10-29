from ..messages import GpuUsageMessages as Msg, render_message
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
            event = render_message(
                Msg.USAGE_OK,
                ctx=ctx,
                check_id=self.check_id,
                what={"process_count": len(gpu_processes)},
            )
            return CheckResult(passed=True, event=event)

        # Check for orphaned rental containers
        for process in gpu_processes:
            container_name = process.get("container_name")
            if container_name and container_name.startswith("container_") and not ctx.rented:
                # Found orphaned rental container - rental ended but container still running
                event = render_message(
                    Msg.ORPHANED_CONTAINER,
                    ctx=ctx,
                    check_id=self.check_id,
                    remediation=Msg.ORPHANED_CONTAINER.remediation.format(orphaned_container=container_name),
                    what={
                        **violation,
                        "orphaned_container": container_name,
                        "rental_status": "ended",
                        "container_status": "still running",
                    },
                )
                return CheckResult(passed=False, event=event)

        event = render_message(
            Msg.USAGE_HIGH,
            ctx=ctx,
            check_id=self.check_id,
            what=violation,
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
