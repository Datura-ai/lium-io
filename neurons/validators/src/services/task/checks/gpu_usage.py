import logging
from dataclasses import dataclass

from services.const import (
    GPU_MEMORY_UTILIZATION_LIMIT,
    GPU_UTILIZATION_LIMIT,
    GPU_WEDGE_MEMORY_MAX,
    POD_CONTAINER_PREFIX,
)
from services.gpu_wedge import (
    GpuCureOutcome,
    cure_wedged_gpus,
    matches_wedge_utilization,
    query_wedged_gpu_uuids,
)

from ..messages import GpuUsageMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WedgedGpu:
    gpu_uuid: str
    gpu_utilization: float | None
    memory_utilization: float | None


class GpuUsageCheck:
    """Re-use the legacy GPU utilisation guard for both rented and idle states."""

    check_id = "gpu.validate.usage"
    fatal = True

    def __init__(self, dry_run: bool = False):
        # A dry run reports a ghost GPU but must not touch the executor (no CUDA-context cure).
        self.dry_run = dry_run

    async def run(self, ctx: Context) -> CheckResult:
        gpu_details = ctx.state.gpu_details
        gpu_processes = ctx.state.gpu_processes

        ghost_result = await self._detect_and_cure_ghost_gpus(ctx, gpu_details, gpu_processes)
        if ghost_result is not None:
            return ghost_result

        return self._judge_process_based_usage(ctx, gpu_details, gpu_processes)

    def _judge_process_based_usage(
        self, ctx: Context, gpu_details: list[dict], gpu_processes: list[dict]
    ) -> CheckResult:
        """The legacy guard: flag GPU load owned by live processes outside our workloads."""
        violation = _find_violation(gpu_details, gpu_processes)

        if violation is None:
            event = render_message(
                Msg.USAGE_OK,
                ctx=ctx,
                check_id=self.check_id,
                what={"process_count": len(gpu_processes)},
            )
            return CheckResult(passed=True, event=event)

        rented_data = ctx.state.rented_data
        filler_container = rented_data.get_filler_container(ctx.executor.uuid) if rented_data else None
        if filler_container and all(
            process.get("container_name") == filler_container for process in gpu_processes
        ):
            event = render_message(
                Msg.USAGE_OK,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "process_count": len(gpu_processes),
                    "filler_container": filler_container,
                },
            )
            return CheckResult(passed=True, event=event)

        # Check for orphaned rental containers
        for process in gpu_processes:
            container_name = process.get("container_name")
            if container_name and container_name.startswith(POD_CONTAINER_PREFIX) and not ctx.rented:
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

    async def _detect_and_cure_ghost_gpus(
        self, ctx: Context, gpu_details: list[dict], gpu_processes: list[dict]
    ) -> CheckResult | None:
        """Cure ghost GPUs in place; fail the executor only when a card stays latched.

        DAH-2427: a hard-killed GPU process can latch the busy counter — 100% utilization
        with no memory and no process — which the process-based guard cannot see. Stateless
        by design: the cure (a CUDA context cycle) is harmless, so it runs on first sight,
        and the verdict comes from re-sampling the live card afterwards. Fail-open: this
        detection is an addition to the legacy guard, so an unexpected error here must never
        break validation for the whole executor. Returns None when there is no ghost.
        """
        try:
            if gpu_processes:
                return None

            wedged_gpus: list[WedgedGpu] = [
                WedgedGpu(
                    gpu_uuid=detail["uuid"],
                    gpu_utilization=detail.get("gpu_utilization"),
                    memory_utilization=detail.get("memory_utilization"),
                )
                for detail in gpu_details
                if _is_wedge_candidate(detail)
            ]
            if not wedged_gpus:
                return None

            if self.dry_run:
                event = render_message(
                    Msg.GPU_WEDGED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "wedged_gpus": wedged_gpus,
                        "cure_attempted": False,
                        "cure_results": [],
                    },
                )
                return CheckResult(passed=False, event=event)

            wedged_uuids: list[str] = [gpu.gpu_uuid for gpu in wedged_gpus]
            cure_outcomes: list[GpuCureOutcome] = await cure_wedged_gpus(ctx.runner, wedged_uuids)

            # The verdict comes from the card itself, not from the cure's exit code.
            still_wedged: list[str] = sorted(
                set(await query_wedged_gpu_uuids(ctx.runner)) & set(wedged_uuids)
            )
            what = {
                "wedged_gpus": wedged_gpus,
                "cure_attempted": True,
                "cure_results": cure_outcomes,
                "still_wedged": still_wedged,
            }
            if not still_wedged:
                event = render_message(
                    Msg.GPU_WEDGE_CURED, ctx=ctx, check_id=self.check_id, what=what
                )
                return CheckResult(passed=True, event=event)

            event = render_message(Msg.GPU_WEDGED, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=False, event=event)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ghost GPU detection failed, falling back to the usage guard: %s", exc)
            return None


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


def _is_wedge_candidate(detail: dict) -> bool:
    gpu_utilization = detail.get("gpu_utilization")
    memory_utilization: float = detail.get("memory_utilization", 0) or 0
    has_uuid: bool = bool(detail.get("uuid"))
    return (
        has_uuid
        and matches_wedge_utilization(gpu_utilization)
        and memory_utilization <= GPU_WEDGE_MEMORY_MAX
    )
