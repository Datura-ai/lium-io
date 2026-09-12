from __future__ import annotations

import asyncio
import math
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from core.config import settings
from services.const import LIUM_INFRA_CONTAINER_PREFIXES

from .cpu_truth import advertised_cpu_count
from .custom_build_orphan_sweep import BUILD_DIND_PREFIX
from .gpu_usage import (
    carries_a_rental_prefix,
    lium_workload_containers,
    the_backend_still_owns,
)
from ..messages import ProviderSideLoadMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

# DAH-2734: floors above which the host's own workload is judged as taking the machine from
# Lium. Set well above a provider's own housekeeping (nginx, fail2ban, the OS baseline): the
# case that opened the ticket ran 9 cores and 1.5 TB. The shadow week measures how many honest
# nodes land above them before enforcement withholds the first payout.
PROVIDER_SIDE_CPU_CORES_LIMIT = 2.0
PROVIDER_SIDE_DISK_LIMIT_KB = 100 * 1024 * 1024  # 100 GB outside docker

# A probe is excused by its NAME, which anyone can copy, so the name tier is capped together at
# the gate's own floor: names can never excuse enough load to hide a violation. The executor's
# own stack is matched by image digest instead and stays uncapped - the scrape itself runs in
# that container and can hold several cores, and zeroing an honest node is the worse error.
CLAIMED_EXCUSE_CORES = PROVIDER_SIDE_CPU_CORES_LIMIT

# A second look before any money is withheld, the same rule the foreign-GPU twin follows with
# the live card. The scrape's window is about two seconds, and Lium's own housekeeping
# (watchtower pulling an image, autoheal restarting a container) can hold two cores inside it.
# A provider mining another subnet holds them all day, so the second reading keeps him and
# drops the spike. An unreadable second reading withholds nothing.
CPU_RESAMPLE_TIMEOUT_SECONDS = 45
# Linux counts /proc/stat in USER_HZ, 100 everywhere we run.
_JIFFIES_PER_SECOND = 100
# `total idle cores dockerd_jiffies` in one line.
# `cores` counts the kernel's own cpuN lines, not the `cpu.count` the miner advertises: that
# value is the one CpuTruthCheck distrusts, and it only catches over-reporting, so a host
# claiming 8 of its 32 threads would report a quarter of its load (review ask, PR #1245).
# `dockerd_jiffies` is dockerd's and containerd's own CPU: pulling a Lium image is work Lium
# asked for, it belongs to no container row, and a 60 GB pull outlives both samples. Read from
# /proc, not from a cgroup path: the executor container has its own cgroup namespace and cannot
# see the host's. Matched by process name, which a provider can copy, so the subtraction is
# capped like every other forgeable excuse.
_HOST_SAMPLE = (
    r"""awk '/^cpu /{idle=$5+$6; total=0; for(i=2;i<=NF;i++) total+=$i} /^cpu[0-9]/{cores++} """
    r"""END{printf "%d %d %d ", total, idle, cores}' /proc/stat; """
    r"""pgrep -x 'dockerd|containerd' | while read pid; do cat /proc/$pid/stat 2>/dev/null; done """
    r"""| awk '{used+=$14+$15} END{printf "%d\n", used+0}'"""
)
# One command, not three: each extra ssh channel costs ~230 ms of dead time INSIDE the sample
# bracket, and the bracket is what makes the host reading comparable to the container sample.
# `@@@` separates the sections. A docker name holds only [a-zA-Z0-9_.-], so no container can
# carry the marker and cut the output somewhere else.
_SECTION_MARKER = "@@@"
# The chain's exit status is the last command's, so a `docker stats` that dies after printing
# half its rows would look like a clean reading with Lium's containers missing - and their CPU
# would land on the provider. It reports its own failure instead.
_STATS_FAILED_MARKER = "STATS_FAILED"
CPU_RESAMPLE_COMMAND = (
    f"{_HOST_SAMPLE}; echo {_SECTION_MARKER}; "
    "{ timeout 30 /usr/bin/docker stats --no-stream "
    "--format '{{.ID}}|{{.Name}}|{{.CPUPerc}}' "
    f"|| echo {_STATS_FAILED_MARKER}; }}; "
    f"echo {_SECTION_MARKER}; {_HOST_SAMPLE}"
)


@dataclass(frozen=True)
class ContainerRow:
    """One container as the node reported it, from the scrape or from `docker stats`.

    `cpu_percent` is core-percent (300% = 3 cores) and None when that source carries no CPU
    reading for the container. `digest` is empty on the ssh path, which does not report it.
    """

    container_id: str
    name: str
    digest: str = ""
    cpu_percent: float | None = None

    @property
    def measured_cpu_percent(self) -> float:
        """The reading, or a ValueError that voids the whole CPU signal for this cycle.

        A container the window cannot account for must never be silently treated as idle: its
        cores would land on the provider's bill.
        """
        if self.cpu_percent is None:
            raise ValueError(f"no cpu reading for container {self.name or self.container_id}")
        if not math.isfinite(self.cpu_percent) or self.cpu_percent < 0:
            raise ValueError(f"unusable cpu reading: {self.cpu_percent}")
        return self.cpu_percent


def floor_to_tenth(cores: float) -> float:
    """Floor, not round: 1.96 cores must not become 2.0 and cross the limit on its own. The
    inner round absorbs float noise first, so an exact 1.6 does not floor to 1.5."""
    return math.floor(round(cores, 3) * 10) / 10


def lium_cores_among(rows: list[ContainerRow], lium_container_keys: set[str]) -> float:
    """What Lium's own containers burn, in cores.

    Matching by both id and name, because the scrape reports full ids and `docker stats`
    reports the first 12 characters.
    """
    lium_short_ids: set[str] = {key[:12] for key in lium_container_keys}
    return (
        sum(
            row.measured_cpu_percent
            for row in rows
            if row.name in lium_container_keys or row.container_id[:12] in lium_short_ids
        )
        / 100
    )


def provider_cores_outside_lium(
    host_cores: float, rows: list[ContainerRow], lium_container_keys: set[str]
) -> float:
    """The host's cores minus the containers Lium itself put there."""
    return floor_to_tenth(max(0.0, host_cores - lium_cores_among(rows, lium_container_keys)))


@dataclass(frozen=True)
class ConfirmedCpuReading:
    """The CPU reading taken over ssh to confirm a load the scrape put above the limit.

    The verdict alone cannot be argued with: a provider mining beside a renter and a renter pod
    that was never subtracted both come out as "N provider cores". These three numbers separate
    them: `host_cores - lium_container_cores - dockerd_cores` is `provider_cores` before the
    final floor to one decimal, so they are kept at two so the subtraction reconciles.
    """

    host_cores: float
    lium_container_cores: float
    dockerd_cores: float

    @property
    def provider_cores(self) -> float:
        """What is left for the provider. Derived, so the readings can never disagree with it."""
        return floor_to_tenth(
            max(0.0, self.host_cores - self.lium_container_cores - self.dockerd_cores)
        )

    def to_specs_fields(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderSideLoad:
    """Host consumption outside Lium's docker containers — the provider's own workloads.

    None means that part was not measurable this cycle (scrape input missing or non-numeric).
    """

    cpu_cores: float | None
    disk_kb: int | None
    # the scrape reading that decided whether to look again. Kept beside the verdict so the
    # shadow week can count how often the two disagree, the way cpu_truth reports `advertised`
    # next to `present`.
    cpu_cores_before_confirmation: float | None = None
    confirmed_reading: ConfirmedCpuReading | None = None

    @property
    def is_measured(self) -> bool:
        return self.cpu_cores is not None or self.disk_kb is not None

    @property
    def is_above_limits(self) -> bool:
        return self.is_cpu_above_limit or self.is_disk_above_limit

    @property
    def is_cpu_above_limit(self) -> bool:
        return (self.cpu_cores or 0.0) >= PROVIDER_SIDE_CPU_CORES_LIMIT

    @property
    def is_disk_above_limit(self) -> bool:
        return (self.disk_kb or 0) >= PROVIDER_SIDE_DISK_LIMIT_KB

    def to_specs_fields(self) -> dict[str, float | int]:
        # specs travel to the backend as raw JSON, so the boundary needs a plain dict
        fields: dict[str, float | int] = {}
        if self.cpu_cores is not None:
            fields["cpu_cores"] = self.cpu_cores
        if self.disk_kb is not None:
            fields["disk_kb"] = self.disk_kb
        if self.confirmed_reading is not None:
            fields.update(self.confirmed_reading.to_specs_fields())
        return fields


def docker_containers(specs: dict[str, Any]) -> list[ContainerRow]:
    """The container rows of the scrape, or an empty list when the scrape sent none.

    `specs` is the raw JSON the miner's machine produced, so this is the one place that reads
    its shape; everything above it works on ContainerRow.
    """
    docker_info = specs.get("docker")
    if not isinstance(docker_info, dict):
        return []
    return [
        ContainerRow(
            container_id=str(entry.get("container_id") or ""),
            name=str(entry.get("name") or ""),
            digest=str(entry.get("digest") or ""),
            cpu_percent=(
                float(entry["cpu_percent"]) if entry.get("cpu_percent") is not None else None
            ),
        )
        for entry in docker_info.get("containers") or []
        if isinstance(entry, dict)
    ]


def infra_container_names(specs: dict[str, Any], excused_by_digest: set[str]) -> set[str]:
    """Lium's probes and the rest of the executor stack, matched by name because they carry no
    backend-issued id.

    A name is free to forge, so the whole tier is capped TOGETHER at the gate's own floor: six
    containers wearing our names can never excuse a nine-core miner. The honest case is a link
    probe busy-polling one core beside an idle runner, watchtower, autoheal and postgres. What
    the image digest already excuses is left out of the sum - the executor container runs the
    scrape itself and must not spend the probes' budget.
    """
    infra_rows = [
        row
        for row in docker_containers(specs)
        if row.name.startswith(LIUM_INFRA_CONTAINER_PREFIXES)
        and row.container_id not in excused_by_digest
    ]
    if sum(row.cpu_percent or 0.0 for row in infra_rows) > CLAIMED_EXCUSE_CORES * 100:
        return set()
    return {row.name for row in infra_rows}


async def late_started_containers_the_backend_owns(
    ctx: Context, known_containers: set[str]
) -> set[str]:
    """Rental-named containers the snapshot missed, kept only when the backend still owns them.

    DAH-2757: `rented_data` is read once at cycle start, so a filler or pod started later in the
    same cycle is ours and absent from the allowlist. Its cores are real, so without this the
    gate bills them to the provider. The twin resolves the same race the same way - the name
    alone proves nothing, so the id inside it is confirmed against the backend.
    """
    candidates: list[str] = sorted(
        {
            row.name
            for row in docker_containers(ctx.state.specs or {})
            if carries_a_rental_prefix(row.name) and row.name not in known_containers
        }
    )
    if not candidates:
        return set()
    verdicts = await asyncio.gather(*(the_backend_still_owns(ctx, name) for name in candidates))
    return {name for name, is_ours in zip(candidates, verdicts) if is_ours}


async def lium_containers(ctx: Context) -> set[str] | None:
    """Every container Lium itself put on this host, by name or by id.

    Three grades of evidence, weakest last: the node's fillers and pods as the BACKEND knows
    them (the set the foreign-GPU twin judges by, plus the build named after a pod it reports);
    the executor's own stack, matched by the image digest it runs; and Lium's short-lived infra
    containers, matched by name. A name is forgeable, so that last tier only excuses load - it
    never turns a container into a rental.

    None when the backend sent no truth this cycle: every filler would then read as a foreign
    workload. The twin returns no verdict in that case and so does this gate.
    """
    if ctx.state.rented_data is None:
        return None
    containers: set[str] = lium_workload_containers(ctx)
    # A custom image builds in `lium-dind-build-<pod_id>`, which is on no rental list and burns
    # cores. The pod id keeps the name rename-proof: it has to match a pod the BACKEND reports
    # here, the same rule the orphan sweep removes such a container by.
    rented_executor = ctx.state.rented_data.executors.get(ctx.executor.uuid)
    if rented_executor:
        containers.update(f"{BUILD_DIND_PREFIX}{pod.pod_id}" for pod in rented_executor.pods)
    specs = ctx.state.specs or {}
    containers.update(await late_started_containers_the_backend_owns(ctx, containers))
    stack_ids: set[str] = executor_stack_container_ids(specs)
    containers.update(stack_ids)
    containers.update(infra_container_names(specs, stack_ids))
    containers.discard("")
    return containers


def executor_stack_container_ids(specs: dict[str, Any]) -> set[str]:
    """Ids of the containers running the executor's own image.

    `docker.container_id` names one of them, but the stack runs that image TWICE - `executor`
    and `monitor` - so the id alone leaves the second one looking like a stranger's workload.
    They are matched by image digest instead. The stack's other containers (postgres, autoheal)
    run other images and stay on the provider's side, where their idle load belongs.
    """
    docker_info = specs.get("docker")
    executor_id = (
        str(docker_info.get("container_id") or "") if isinstance(docker_info, dict) else ""
    )
    rows = docker_containers(specs)
    executor_digest = next((row.digest for row in rows if row.container_id == executor_id), "")
    ids: set[str] = {executor_id}
    if executor_digest:
        ids.update(row.container_id for row in rows if row.digest == executor_digest)
    # an empty id is "the scrape did not say", never a container to excuse
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
        rows = docker_containers(specs)
        host_percent = float(docker_info.get("host_cpu_percent"))
        readable = math.isfinite(host_percent) and host_percent >= 0
        if core_count and lium_container_keys is not None and rows and readable:
            cpu_cores = provider_cores_outside_lium(
                host_percent / 100 * core_count, rows, lium_container_keys
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
        cores = sum(
            row.cpu_percent or 0.0
            for row in docker_containers(specs)
            if carries_a_rental_prefix(row.name) and row.name not in lium_container_keys
        )
    except (TypeError, ValueError, AttributeError, OverflowError):
        return None
    return round(cores / 100, 1)


def parse_confirmed_cpu_reading(
    sample_before: str,
    container_rows: str,
    sample_after: str,
    lium_container_keys: set[str],
) -> ConfirmedCpuReading | None:
    """What the second reading measured, or None if it is not readable.

    Host busy time comes from the /proc/stat jiffies taken on either side of `docker stats`, so
    both cover the same seconds - the pairing the whole subtraction rests on. Out of that come
    the containers Lium runs and dockerd's own work, and what is left is the provider's.
    """
    try:
        first_total, first_idle, _, first_dockerd = (int(part) for part in sample_before.split())
        last_total, last_idle, kernel_cores, last_dockerd = (
            int(part) for part in sample_after.split()
        )
        total_delta = last_total - first_total
        rows = container_rows.strip().splitlines()
        if any(_STATS_FAILED_MARKER in row for row in rows):
            return None
        # Every host runs at least the executor's own container, so no rows means docker did not
        # answer. Reading that as "Lium runs nothing here" would charge the whole host to the
        # provider.
        if total_delta <= 0 or kernel_cores <= 0 or not rows:
            return None
        stats_rows: list[ContainerRow] = []
        for line in rows:
            short_id, name, raw_percent = (part.strip() for part in line.split("|"))
            stats_rows.append(
                ContainerRow(
                    container_id=short_id, name=name, cpu_percent=float(raw_percent.rstrip("%"))
                )
            )
    except (TypeError, ValueError, OverflowError):
        return None
    window_seconds = total_delta / (_JIFFIES_PER_SECOND * kernel_cores)
    dockerd_cores = (last_dockerd - first_dockerd) / _JIFFIES_PER_SECOND / window_seconds
    # `dockerd` is a process name anyone can copy, so its excuse is capped like the name tier
    excused_dockerd_cores = min(max(0.0, dockerd_cores), CLAIMED_EXCUSE_CORES)
    host_cores = (total_delta - (last_idle - first_idle)) / total_delta * kernel_cores
    return ConfirmedCpuReading(
        host_cores=round(host_cores, 2),
        lium_container_cores=round(lium_cores_among(stats_rows, lium_container_keys), 2),
        dockerd_cores=round(excused_dockerd_cores, 2),
    )


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

    async def _confirm_the_cpu_reading_over_ssh(
        self, ctx: Context, lium_container_keys: set[str] | None
    ) -> ConfirmedCpuReading | None:
        """Read the load once more over ssh. None on anything unreadable, which withholds nothing.

        Through `ctx.runner`, like the twin gate's second look at the card: the runner owns the
        timeout, the failure and the timing log.
        """
        if lium_container_keys is None or ctx.runner is None:
            return None
        result = await ctx.runner.run(
            CPU_RESAMPLE_COMMAND, timeout=CPU_RESAMPLE_TIMEOUT_SECONDS, retryable=False
        )
        if not result.success:
            return None
        sections = result.stdout.split(_SECTION_MARKER)
        if len(sections) != 3:
            return None
        jiffies_before, container_rows, jiffies_after = sections
        return parse_confirmed_cpu_reading(
            jiffies_before, container_rows, jiffies_after, lium_container_keys
        )

    def _what_we_saw(
        self,
        ctx: Context,
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
                ctx.state.specs or {}, lium_container_keys
            ),
            # what the scrape saw before the confirmation, and what the ssh reading measured.
            # Together they say how often the cheap screen and the paid-for verdict disagree.
            # Both None when the floor was never crossed and no second look was taken.
            "cpu_cores_before_confirmation": provider_side_load.cpu_cores_before_confirmation,
            "confirmed_reading": (
                asdict(provider_side_load.confirmed_reading)
                if provider_side_load.confirmed_reading is not None
                else None
            ),
        }

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
        confirmed_reading = await self._confirm_the_cpu_reading_over_ssh(ctx, lium_container_keys)
        return replace(
            measured,
            # None when the second look could not be read: an unconfirmed reading withholds
            # nothing, and `cpu_cores_before_confirmation` keeps that case countable.
            cpu_cores=confirmed_reading.provider_cores if confirmed_reading else None,
            cpu_cores_before_confirmation=measured.cpu_cores,
            confirmed_reading=confirmed_reading,
        )

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.PROVIDER_SIDE_LOAD_CHECK_ENABLED:
            event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
            return CheckResult(passed=True, event=event)

        lium_container_keys = await lium_containers(ctx)
        provider_side_load = await self.measure_provider_side_load(ctx, lium_container_keys)
        if not provider_side_load.is_measured:
            event = render_message(
                Msg.NOT_MEASURABLE,
                ctx=ctx,
                check_id=self.check_id,
                # a scrape reading with no verdict means the second look failed, not that the
                # machine was never measured; the shadow week needs that rate
                what={
                    "executor_uuid": ctx.executor.uuid,
                    "cpu_cores_before_confirmation": provider_side_load.cpu_cores_before_confirmation,
                },
            )
            return CheckResult(passed=True, event=event)

        updated_state = replace(
            ctx.state,
            specs={**ctx.state.specs, "provider_side_load": provider_side_load.to_specs_fields()},
        )
        updates: dict[str, object] = {"state": updated_state}

        what = self._what_we_saw(ctx, provider_side_load, lium_container_keys)
        if not provider_side_load.is_above_limits:
            event = render_message(Msg.LOAD_OK, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=True, event=event, updates=updates)

        return self._verdict_for_a_load_above_the_limits(ctx, provider_side_load, what, updates)

    def _verdict_for_a_load_above_the_limits(
        self,
        ctx: Context,
        provider_side_load: ProviderSideLoad,
        what: dict[str, object],
        updates: dict[str, object],
    ) -> CheckResult:
        """Render the verdict and, under enforcement, set the flag that zeroes the score.

        Only the CPU half withholds money. A high disk figure is reported and never scored: it
        is read once, with no second look, and every docker category nobody enumerated
        (loopback volumes, json logs, BuildCache so far) lands in it and would zero an honest
        node. The shadow week measures the disk numbers before that half is armed.
        """
        enforce: bool = (
            settings.PROVIDER_SIDE_LOAD_ENFORCEMENT_ENABLED
            and provider_side_load.is_cpu_above_limit
        )
        if enforce:
            impact = None
        elif not provider_side_load.is_cpu_above_limit:
            impact = "Disk is observed only: score was NOT changed"
        else:
            impact = "Shadow observation only: score was NOT changed"
        event = render_message(
            Msg.LOAD_ABOVE_LIMIT,
            ctx=ctx,
            check_id=self.check_id,
            severity=None if enforce else "warning",
            impact=impact,
            what=what,
        )
        # The check is non-fatal, so passed=False alone changes nothing: the verdict travels as
        # a context flag that calculate_scores gates on, like the CPU-truth one.
        if enforce:
            updates["provider_side_load_passed"] = False
        return CheckResult(passed=not enforce, event=event, updates=updates)
