from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from core.config import settings
from services.const import FILLER_CONTAINER_PREFIX, POD_CONTAINER_PREFIX

from .cpu_truth import advertised_cpu_count
from .custom_build_orphan_sweep import BUILD_DIND_PREFIX
from .gpu_usage import _lium_workload_containers
from ..messages import ProviderSideLoadMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

# DAH-2734: floors above which the host's own workload is judged as taking the machine from
# Lium. Set well above a provider's own housekeeping (nginx, fail2ban, the OS baseline): the
# case that opened the ticket ran 9 cores and 1.5 TB. The shadow week measures how many honest
# nodes land above them before enforcement withholds the first payout.
PROVIDER_SIDE_CPU_CORES_LIMIT = 2.0
PROVIDER_SIDE_DISK_LIMIT_KB = 100 * 1024 * 1024  # 100 GB outside docker

# A second look before any money is withheld, the same rule the foreign-GPU twin follows with
# the live card. The scrape's window is about two seconds, and Lium's own housekeeping
# (watchtower pulling an image, autoheal restarting a container) can hold two cores inside it.
# A provider mining another subnet holds them all day, so the second reading keeps him and
# drops the spike. An unreadable second reading withholds nothing.
CPU_RESAMPLE_TIMEOUT_SECONDS = 45
HOST_CPU_JIFFIES_COMMAND = (
    "awk '/^cpu /{idle=$5+$6; total=0; for(i=2;i<=NF;i++) total+=$i; print total, idle}' /proc/stat"
)
CONTAINER_CPU_COMMAND = (
    "timeout 30 /usr/bin/docker stats --no-stream --format '{{.ID}}|{{.Name}}|{{.CPUPerc}}'"
)


def _usable_percent(raw_percent: Any) -> float:
    percent = float(raw_percent)
    if not math.isfinite(percent) or percent < 0:
        raise ValueError(f"unusable container cpu reading: {percent}")
    return percent


@dataclass(frozen=True)
class ContainerCpuReading:
    """One `docker stats` row: who the container is, and its CPU in core-percent (300% = 3 cores)."""

    container_id: str
    name: str
    cpu_percent: float


def provider_cores_outside_lium(
    host_cores: float, readings: list[ContainerCpuReading], lium_container_keys: set[str]
) -> float:
    """The host's cores minus the containers Lium itself put there.

    Matching by both id and name, because the scrape reports full ids and `docker stats`
    reports the first 12 characters.
    """
    lium_short_ids: set[str] = {key[:12] for key in lium_container_keys}
    lium_percent: float = sum(
        reading.cpu_percent
        for reading in readings
        if reading.name in lium_container_keys
        or reading.container_id in lium_container_keys
        or reading.container_id[:12] in lium_short_ids
    )
    return round(max(0.0, host_cores - lium_percent / 100), 1)


@dataclass(frozen=True)
class ProviderSideLoad:
    """Host consumption outside Lium's docker containers — the provider's own workloads.

    None means that part was not measurable this cycle (scrape input missing or non-numeric).
    """

    cpu_cores: float | None
    disk_kb: int | None

    @property
    def is_measured(self) -> bool:
        return self.cpu_cores is not None or self.disk_kb is not None

    @property
    def is_above_limits(self) -> bool:
        return (
            (self.cpu_cores or 0.0) >= PROVIDER_SIDE_CPU_CORES_LIMIT
            or (self.disk_kb or 0) >= PROVIDER_SIDE_DISK_LIMIT_KB
        )

    def to_specs_fields(self) -> dict[str, float | int]:
        # specs travel to the backend as raw JSON, so the boundary needs a plain dict
        fields: dict[str, float | int] = {}
        if self.cpu_cores is not None:
            fields["cpu_cores"] = self.cpu_cores
        if self.disk_kb is not None:
            fields["disk_kb"] = self.disk_kb
        return fields


def lium_containers(ctx: Context) -> set[str] | None:
    """Every container Lium itself put on this host, by name or by id.

    The node's fillers and pods as the BACKEND knows them - the same set the foreign-GPU twin
    judges by, so a `docker rename` defeats neither gate - plus the executor's own container,
    which is on no rental list but is still ours.

    None when the backend sent no truth this cycle: every filler would then read as a foreign
    workload. The twin returns no verdict in that case and so does this gate.
    """
    if ctx.state.rented_data is None:
        return None
    containers: set[str] = _lium_workload_containers(ctx)
    # A custom image builds in `lium-dind-build-<pod_id>`, which is on no rental list and burns
    # cores. The pod id keeps the name rename-proof: it has to match a pod the BACKEND reports
    # here, the same rule the orphan sweep removes such a container by.
    rented_executor = ctx.state.rented_data.executors.get(ctx.executor.uuid)
    if rented_executor:
        containers.update(f"{BUILD_DIND_PREFIX}{pod.pod_id}" for pod in rented_executor.pods)
    docker_info = (ctx.state.specs or {}).get("docker")
    if isinstance(docker_info, dict):
        containers.update(executor_stack_container_ids(docker_info))
    containers.discard("")
    return containers


def executor_stack_container_ids(docker_info: dict[str, Any]) -> set[str]:
    """Ids of the containers running the executor's own image.

    `docker.container_id` names one of them, but the stack runs that image TWICE - `executor`
    and `monitor` - so the id alone leaves the second one looking like a stranger's workload.
    They are matched by image digest instead. The stack's other containers (postgres, autoheal)
    run other images and stay on the provider's side, where their idle load belongs.
    """
    entries = [entry for entry in (docker_info.get("containers") or []) if isinstance(entry, dict)]
    executor_id = str(docker_info.get("container_id") or "")
    executor_digest = next(
        (entry.get("digest") for entry in entries if entry.get("container_id") == executor_id),
        None,
    )
    ids: set[str] = {executor_id}
    if executor_digest:
        ids.update(
            str(entry.get("container_id") or "")
            for entry in entries
            if entry.get("digest") == executor_digest
        )
    ids.discard("")
    return ids


def compute_provider_side_load(
    specs: dict[str, Any], core_count: int | None, lium_container_keys: set[str] | None
) -> ProviderSideLoad:
    """CPU: host utilization over the `docker stats` window (docker.host_cpu_percent, 0-100
    across all cores) minus the core-percents of LIUM's containers only, in cores. A container
    Lium did not put there is the provider's own workload and stays in the total - subtracting
    it would let the provider mine another subnet with `docker run` and read as idle. Disk:
    hard_disk.used minus the docker breakdown (images+containers+volumes, DAH-2514), in kB —
    includes the OS baseline, so thresholds must leave room for it.

    Each part needs all its inputs present, numeric and well-shaped, else stays None — never
    a guess (every malformed shape lands in the except, so bad specs can never abort the
    validation pipeline). The CPU part additionally requires EVERY listed container to carry a
    cpu_percent: a partial `docker stats` result would silently attribute the missing
    containers' CPU to the provider.
    """
    cpu_cores: float | None = None
    disk_kb: int | None = None

    try:
        docker_info = specs.get("docker") or {}
        containers = docker_info.get("containers") or []
        host_percent = float(docker_info.get("host_cpu_percent"))
        readable = math.isfinite(host_percent) and host_percent >= 0
        if core_count and lium_container_keys is not None and containers and readable:
            readings = [
                # a missing or unusable percent raises into the except and voids the CPU part
                ContainerCpuReading(
                    container_id=str(container.get("container_id") or ""),
                    name=str(container.get("name") or ""),
                    cpu_percent=_usable_percent(container["cpu_percent"]),
                )
                for container in containers
            ]
            cpu_cores = provider_cores_outside_lium(
                host_percent / 100 * core_count, readings, lium_container_keys
            )
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        pass

    try:
        hard_disk = specs.get("hard_disk") or {}
        used_kb = int(hard_disk.get("used"))
        docker_parts_kb = [int(hard_disk.get(key)) for key in ("images", "containers", "volumes")]
        if used_kb >= 0 and all(part_kb >= 0 for part_kb in docker_parts_kb):
            disk_kb = max(0, used_kb - sum(docker_parts_kb))
    except (TypeError, ValueError, AttributeError, OverflowError):
        pass

    return ProviderSideLoad(cpu_cores=cpu_cores, disk_kb=disk_kb)


def lium_named_cores_outside_snapshot(
    specs: dict[str, Any], lium_container_keys: set[str] | None
) -> float | None:
    """CPU of containers that carry one of our prefixes but are absent from the allowlist.

    The backend can start a filler or a pod AFTER this cycle's rented_data snapshot. That
    container is ours, yet it reads as the provider's workload. It still counts against the
    provider - a name is not proof, and the twin gate refuses the same shortcut - but it is
    reported on its own, so the shadow week measures how often the race happens.
    """
    if lium_container_keys is None:
        return None
    try:
        containers = (specs.get("docker") or {}).get("containers") or []
        cores = sum(
            float(container.get("cpu_percent") or 0)
            for container in containers
            if str(container.get("name") or "").startswith(
                (POD_CONTAINER_PREFIX, FILLER_CONTAINER_PREFIX)
            )
            and container.get("name") not in lium_container_keys
        )
    except (TypeError, ValueError, AttributeError, OverflowError):
        return None
    return round(cores / 100, 1)


def parse_resampled_cpu_cores(
    jiffies_before: str,
    container_rows: str,
    jiffies_after: str,
    core_count: int,
    lium_container_keys: set[str],
) -> float | None:
    """Provider-side cores from the second reading, or None if it is not readable.

    Host busy time comes from the /proc/stat jiffies taken on either side of `docker stats`,
    so both readings cover the same seconds - the pairing the whole subtraction rests on.
    """
    try:
        first_total, first_idle = (int(part) for part in jiffies_before.split())
        last_total, last_idle = (int(part) for part in jiffies_after.split())
        total_delta = last_total - first_total
        rows = container_rows.strip().splitlines()
        # Every host runs at least the executor's own container, so no rows means docker did not
        # answer. Reading that as "Lium runs nothing here" would charge the whole host to the
        # provider.
        if total_delta <= 0 or not rows:
            return None
        readings: list[ContainerCpuReading] = []
        for row in rows:
            short_id, name, raw_percent = (part.strip() for part in row.split("|"))
            readings.append(
                ContainerCpuReading(
                    container_id=short_id,
                    name=name,
                    cpu_percent=_usable_percent(raw_percent.rstrip("%")),
                )
            )
    except (TypeError, ValueError, OverflowError):
        return None
    host_cores = (total_delta - (last_idle - first_idle)) / total_delta * core_count
    return provider_cores_outside_lium(host_cores, readings, lium_container_keys)


class ProviderSideLoadCheck:
    """Judge a machine by what its OWNER takes from it, not only by what Lium runs on it.

    DAH-2734 is the CPU/disk twin of the foreign-GPU gate (DAH-2735): a provider who mines
    another subnet on a listed machine (SN13/Data Universe: 9 of 32 threads, 1.5 TB, GPU
    untouched) sells capacity he is already using himself. The GPU gate cannot see him,
    because he never touches the card.

    The verdict comes from the specs the scrape already collects: host CPU minus every Lium
    container's CPU, and disk used minus the docker breakdown. Above the floors the node is
    judged, rented or idle alike — the provider cheats the renter in the first case and the
    marketplace in the second.

    Shadow by default like every other money-withholding gate: without
    PROVIDER_SIDE_LOAD_ENFORCEMENT_ENABLED the verdict is logged, not scored. An inability to
    measure never withholds money — a missing or malformed signal passes silently.
    """

    check_id = "host.validate.provider_side_load"
    fatal = False

    async def _resample_cpu_cores(
        self, ctx: Context, core_count: int | None, lium_container_keys: set[str] | None
    ) -> float | None:
        """Read the load once more over ssh. None on anything unreadable, which withholds nothing.

        Three plain commands through `ctx.runner`, like the twin gate's second look at the card:
        the runner owns the timeout, the failure and the timing log, and the two jiffies readings
        bracket the container sample.
        """
        if not core_count or lium_container_keys is None or ctx.runner is None:
            return None
        jiffies_before = await ctx.runner.run(
            HOST_CPU_JIFFIES_COMMAND, timeout=CPU_RESAMPLE_TIMEOUT_SECONDS, retryable=False
        )
        containers = await ctx.runner.run(
            CONTAINER_CPU_COMMAND, timeout=CPU_RESAMPLE_TIMEOUT_SECONDS, retryable=False
        )
        jiffies_after = await ctx.runner.run(
            HOST_CPU_JIFFIES_COMMAND, timeout=CPU_RESAMPLE_TIMEOUT_SECONDS, retryable=False
        )
        if not (jiffies_before.success and containers.success and jiffies_after.success):
            return None
        return parse_resampled_cpu_cores(
            jiffies_before.stdout,
            containers.stdout,
            jiffies_after.stdout,
            core_count,
            lium_container_keys,
        )

    async def measure_provider_side_load(
        self, ctx: Context, lium_container_keys: set[str] | None
    ) -> ProviderSideLoad:
        """The scrape's reading, confirmed over ssh whenever it is high enough to cost money."""
        core_count: int | None = advertised_cpu_count(ctx)
        measured = compute_provider_side_load(
            ctx.state.specs or {}, core_count, lium_container_keys
        )
        if (measured.cpu_cores or 0.0) < PROVIDER_SIDE_CPU_CORES_LIMIT:
            return measured
        return replace(
            measured,
            cpu_cores=await self._resample_cpu_cores(ctx, core_count, lium_container_keys),
        )

    def _what_we_saw(
        self,
        ctx: Context,
        specs: dict[str, Any],
        provider_side_load: ProviderSideLoad,
        lium_container_keys: set[str] | None,
    ) -> dict[str, object]:
        """The event payload: the readings, the floors they were judged against, and the race
        metric the shadow week needs."""
        # Same rented-status pattern as port_count.py / sysbox_required.py.
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        is_rented: bool = rented_executor is not None and len(rented_executor.pods) > 0
        return {
            "executor_uuid": ctx.executor.uuid,
            "provider_cpu_cores": provider_side_load.cpu_cores,
            "provider_disk_gb": (
                round(provider_side_load.disk_kb / (1024 * 1024), 1)
                if provider_side_load.disk_kb is not None
                else None
            ),
            "cpu_cores_limit": PROVIDER_SIDE_CPU_CORES_LIMIT,
            "disk_gb_limit": PROVIDER_SIDE_DISK_LIMIT_KB // (1024 * 1024),
            "is_rented": is_rented,
            "lium_named_outside_snapshot_cores": lium_named_cores_outside_snapshot(
                specs, lium_container_keys
            ),
        }

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.PROVIDER_SIDE_LOAD_CHECK_ENABLED:
            event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
            return CheckResult(passed=True, event=event)

        lium_container_keys = lium_containers(ctx)
        specs = ctx.state.specs or {}
        provider_side_load = await self.measure_provider_side_load(ctx, lium_container_keys)
        if not provider_side_load.is_measured:
            event = render_message(
                Msg.NOT_MEASURABLE,
                ctx=ctx,
                check_id=self.check_id,
                what={"executor_uuid": ctx.executor.uuid},
            )
            return CheckResult(passed=True, event=event)

        updated_state = replace(
            ctx.state,
            specs={**ctx.state.specs, "provider_side_load": provider_side_load.to_specs_fields()},
        )
        updates: dict[str, object] = {"state": updated_state}

        what = self._what_we_saw(ctx, specs, provider_side_load, lium_container_keys)
        if not provider_side_load.is_above_limits:
            event = render_message(Msg.LOAD_OK, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=True, event=event, updates=updates)

        enforce: bool = settings.PROVIDER_SIDE_LOAD_ENFORCEMENT_ENABLED
        event = render_message(
            Msg.LOAD_ABOVE_LIMIT,
            ctx=ctx,
            check_id=self.check_id,
            severity=None if enforce else "warning",
            impact=None if enforce else "Shadow observation only: score was NOT changed",
            what=what,
        )
        # The check is non-fatal, so passed=False alone changes nothing: the verdict travels as
        # a context flag that calculate_scores gates on, like the CPU-truth one.
        if enforce:
            updates["provider_side_load_passed"] = False
        return CheckResult(passed=not enforce, event=event, updates=updates)
