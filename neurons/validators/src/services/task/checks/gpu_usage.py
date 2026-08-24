import asyncio
import logging
from dataclasses import dataclass
from uuid import UUID

from core.config import settings
from protocol.vc_protocol.compute_requests import (
    LIVE_FILLER_RUN_STATUSES,
    FillerRunActiveResponse,
    PodRentalActiveResponse,
)
from services.const import (
    FILLER_CONTAINER_PREFIX,
    GPU_HELD_VRAM_MB_LIMIT,
    GPU_MEMORY_UTILIZATION_LIMIT,
    GPU_UTILIZATION_LIMIT,
    GPU_WEDGE_MEMORY_MAX,
    POD_CONTAINER_PREFIX,
)
from services.gpu_wedge import (
    COMPUTE_APPS_QUERY_COMMAND,
    GPU_QUERY_COMMAND,
    NVIDIA_SMI_QUERY_TIMEOUT_SECONDS,
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
class ForeignGpuProcess:
    pid: int | None
    container_name: str | None
    cgroup: str | None


@dataclass(frozen=True)
class HeldVram:
    gpu_uuid: str | None
    memory_used_mb: float


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

        return await self._judge_process_based_usage(ctx, gpu_details, gpu_processes)

    async def _judge_foreign_occupancy(
        self, ctx: Context, gpu_details: list[dict], gpu_processes: list[dict]
    ) -> CheckResult | None:
        """Judge an idle card by WHO holds it, not by how hard it is being used (DAH-2735).

        Competitor marketplaces (Nodexo/SN106) idle below the percentage limits on purpose: a
        rental holding 22.4 GB at 0% load passes every utilization gate. The expected holders
        come from the backend (`rented_data`), never from a container-name prefix — the whole
        point of the ticket is a verdict `docker rename` cannot defeat. A container named like
        ours but missing from the snapshot is asked about directly (DAH-2757); the prefix only
        decides whom to ask, the backend still gives the answer. Shadow by default like
        every other money-withholding gate: without FOREIGN_GPU_WORKLOAD_ENFORCEMENT_ENABLED
        the verdict is logged, not scored. Returns None when the card is clean, leaving the
        legacy verdict to the caller.
        """
        if not settings.FOREIGN_GPU_WORKLOAD_CHECK_ENABLED:
            return None

        # No backend truth this cycle means no allowlist to judge against — an inability to
        # measure, which never withholds money here.
        if ctx.state.rented_data is None:
            return None

        enforce: bool = settings.FOREIGN_GPU_WORKLOAD_ENFORCEMENT_ENABLED
        workload_containers: set[str] = _lium_workload_containers(ctx)
        foreign_processes: list[ForeignGpuProcess] = [
            ForeignGpuProcess(
                pid=process.get("pid"),
                container_name=process.get("container_name"),
                cgroup=process.get("info"),
            )
            for process in gpu_processes
            if process.get("container_name") not in workload_containers
            and not _runs_in_the_executors_own_cgroup(process)
        ]
        foreign_processes = await _drop_containers_the_backend_still_owns(ctx, foreign_processes)

        if foreign_processes:
            what = {
                "foreign_processes": foreign_processes,
                "process_count": len(gpu_processes),
                # Our own name on a container the backend disowns — the id inside it belongs
                # to no run it ever issued. Counted separately, because that is the shape a
                # forged name takes once the late-start race (DAH-2757) is out of the way.
                "lium_named_outside_snapshot": [
                    process for process in foreign_processes if _carries_a_lium_prefix(process)
                ],
            }
            template = Msg.FOREIGN_PROCESS
        elif held_vram := _held_vram_without_owner(gpu_details, gpu_processes, workload_containers):
            if not await self._card_still_holds_vram_with_no_owner(ctx, held_vram):
                return None
            what = {"held_vram": held_vram}
            template = Msg.VRAM_HELD
        else:
            return None

        event = render_message(
            template,
            ctx=ctx,
            check_id=self.check_id,
            severity=None if enforce else "warning",
            impact=None if enforce else "Shadow observation only: score was NOT changed",
            what=what,
        )
        return CheckResult(passed=not enforce, event=event)

    async def _card_still_holds_vram_with_no_owner(
        self, ctx: Context, held_vram: list[HeldVram]
    ) -> bool:
        """Re-sample the live card before calling its memory ownerless.

        The scrape reads NVML memory first and walks `/proc` after, so a GPU process that exits
        in between leaves its memory figure behind with nothing to attribute it to — a clean
        node would then be judged as hiding a workload. The verdict comes from the card itself,
        the same rule the ghost-GPU path above follows. Fail-open on any error: an unreadable
        card is an inability to measure, which never withholds money.

        The two readings are not the same number: the scrape's NVML figure includes the
        driver-reserved block and `nvidia-smi memory.used` does not (measured on an A6000:
        1899 vs 1299 MB). Both are compared against the same floor, which makes this
        confirmation the stricter of the two — deliberately, since it only ever withholds.
        """
        try:
            gpu_query, compute_apps = await asyncio.gather(
                ctx.runner.run(GPU_QUERY_COMMAND, timeout=NVIDIA_SMI_QUERY_TIMEOUT_SECONDS, retryable=False),
                ctx.runner.run(COMPUTE_APPS_QUERY_COMMAND, timeout=NVIDIA_SMI_QUERY_TIMEOUT_SECONDS, retryable=False),
            )
            # A failed query returns empty stdout, which would otherwise read as "no owner".
            if not gpu_query.success or not compute_apps.success:
                return False
            if compute_apps.stdout.strip():
                return False

            live_memory_mb: dict[str, float] = _parse_gpu_memory_used_mb(gpu_query.stdout)
            if not live_memory_mb:
                return False
            return any(
                live_memory_mb.get(held.gpu_uuid or "", 0) > GPU_HELD_VRAM_MB_LIMIT
                for held in held_vram
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Live VRAM re-sample failed, not judging the card: %s", exc)
            return False

    async def _judge_process_based_usage(
        self, ctx: Context, gpu_details: list[dict], gpu_processes: list[dict]
    ) -> CheckResult:
        """The legacy guard: flag GPU load owned by live processes outside our workloads."""
        violation = _find_violation(gpu_details, gpu_processes)

        if violation is None:
            foreign_result = await self._judge_foreign_occupancy(ctx, gpu_details, gpu_processes)
            if foreign_result is not None:
                return foreign_result

            event = render_message(
                Msg.USAGE_OK,
                ctx=ctx,
                check_id=self.check_id,
                what={"process_count": len(gpu_processes)},
            )
            return CheckResult(passed=True, event=event)

        rented_data = ctx.state.rented_data
        # A GPU-split node runs one filler per VRAM bundle (DAH-2465): the GPU is "clean" as long as
        # every process belongs to ONE OF the node's fillers, not a single expected container.
        filler_containers = set(rented_data.get_filler_containers(ctx.executor.uuid)) if rented_data else set()
        if filler_containers and all(
            process.get("container_name") in filler_containers for process in gpu_processes
        ):
            event = render_message(
                Msg.USAGE_OK,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "process_count": len(gpu_processes),
                    "filler_containers": sorted(filler_containers),
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


def _lium_workload_containers(ctx: Context) -> set[str]:
    """This node's fillers and pods, as the BACKEND knows them — not as the node names them.

    A container-name prefix would hand a pass to anything the provider renames `filler_*`,
    and the ticket's whole requirement is a verdict `docker rename` cannot defeat.
    """
    rented_data = ctx.state.rented_data
    containers: set[str] = set(rented_data.get_filler_containers(ctx.executor.uuid))
    rented_executor = rented_data.executors.get(ctx.executor.uuid)
    if rented_executor:
        containers.update(pod.container_name for pod in rented_executor.pods)
    containers.discard("")
    return containers


def _parse_gpu_memory_used_mb(gpu_query_csv: str) -> dict[str, float]:
    """uuid -> memory.used MiB, from GPU_QUERY_COMMAND's `uuid, utilization, memory.used` rows."""
    memory_by_uuid: dict[str, float] = {}
    for line in gpu_query_csv.strip().splitlines():
        parts: list[str] = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        gpu_uuid, _utilization, memory_raw = parts
        try:
            memory_by_uuid[gpu_uuid] = float(memory_raw)
        except ValueError:
            continue
    return memory_by_uuid


def _carries_a_lium_prefix(process: ForeignGpuProcess) -> bool:
    """The container is named like one of ours, but the backend did not report it this cycle."""
    container_name: str = process.container_name or ""
    return container_name.startswith((POD_CONTAINER_PREFIX, FILLER_CONTAINER_PREFIX))


async def _drop_containers_the_backend_still_owns(
    ctx: Context, foreign_processes: list[ForeignGpuProcess]
) -> list[ForeignGpuProcess]:
    """Ask the backend about a Lium-named container that this cycle's snapshot does not hold.

    DAH-2757: `rented_data` is read once, at cycle start. A filler or pod the backend starts
    later in the same cycle carries our own name and is absent from the allowlist, so it reads
    as foreign at 0% load — 7 honest nodes hit this in the first prod shadow day. The name on
    its own still proves nothing, because `docker rename` forges it, so the id inside the name
    is confirmed against the backend, which disowns an id it never issued.

    One question per container, not per process: a single container holds as many GPU processes
    as it has workers (8 in one prod pod), and this check sits on the healthy path of every
    executor in every cycle.
    """
    lium_named_containers: list[str] = sorted(
        {process.container_name for process in foreign_processes if _carries_a_lium_prefix(process)}
    )
    if not lium_named_containers:
        return foreign_processes

    ownership_verdicts: list[bool] = await asyncio.gather(
        *(_the_backend_still_owns(ctx, name) for name in lium_named_containers)
    )
    still_ours: set[str] = {
        name for name, is_ours in zip(lium_named_containers, ownership_verdicts) if is_ours
    }
    return [process for process in foreign_processes if process.container_name not in still_ours]


async def _the_backend_still_owns(ctx: Context, container_name: str) -> bool:
    """True when the backend confirms the id inside the container name as a live run of ours.

    Mirrors the filler re-check in rental_verification. Fail-open on an unreachable backend
    (a None response): an inability to measure never withholds money here.
    """
    is_filler: bool = container_name.startswith(FILLER_CONTAINER_PREFIX)
    prefix: str = FILLER_CONTAINER_PREFIX if is_filler else POD_CONTAINER_PREFIX
    backend_issued_id: str | None = _uuid_after_prefix(container_name, prefix)
    if backend_issued_id is None:
        return False

    if is_filler:
        filler_run: FillerRunActiveResponse | None = await ctx.services.backend.get_filler_run_active(
            backend_issued_id
        )
        return filler_run is None or filler_run.status in LIVE_FILLER_RUN_STATUSES

    pod_rental: PodRentalActiveResponse | None = await ctx.services.backend.get_pod_rental_active(
        backend_issued_id
    )
    return pod_rental is None or pod_rental.active


def _uuid_after_prefix(container_name: str, prefix: str) -> str | None:
    """The uuid the backend issued, or None when the name carries something else."""
    try:
        return str(UUID(container_name.removeprefix(prefix)))
    except ValueError:
        return None


def _runs_in_the_executors_own_cgroup(process: dict) -> bool:
    """True when the process shares the cgroup namespace of the scrape — i.e. it is the executor.

    The scrape reads `/proc/<pid>/cgroup` from inside the executor container, so anything
    living elsewhere escapes that namespace and reads as `0::/../docker-<id>.scope` (another
    container) or `0::/../../user.slice/...` (a host process); every one of the 692 GPU
    processes on the prod fleet has that shape. The executor's own GPU work — a VerifyX run
    the validator starts over SSH, which can outlive its command timeout — reads as a plain
    `0::/…` instead, and carries no container id for the scrape to resolve a name from.
    """
    cgroup: str = process.get("info") or ""
    return bool(cgroup) and ".." not in cgroup


def _held_vram_without_owner(
    gpu_details: list[dict], gpu_processes: list[dict], workload_containers: set[str]
) -> list[HeldVram]:
    """VRAM held above the driver-reserve floor while no process is visible to account for it.

    A node running one of our GPU containers is exempt: NVML sometimes returns no processes at
    all on a node whose filler is demonstrably working (seen in prod at 34% GPU utilization).

    ponytail: node-level, not per-GPU — the scrape does not record which GPU a PID sits on;
    record the GPU uuid in get_gpu_processes if per-card attribution ever matters.
    """
    if gpu_processes or workload_containers:
        return []
    return [
        HeldVram(gpu_uuid=detail.get("uuid"), memory_used_mb=detail.get("memory_used_mb") or 0)
        for detail in gpu_details
        if (detail.get("memory_used_mb") or 0) > GPU_HELD_VRAM_MB_LIMIT
    ]


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
