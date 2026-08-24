from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from core.config import settings
from services.const import FILLER_CONTAINER_PREFIX, POD_CONTAINER_PREFIX

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
        containers.add(str(docker_info.get("container_id") or ""))
    containers.discard("")
    return containers


def compute_provider_side_load(
    specs: dict[str, Any], lium_container_keys: set[str] | None
) -> ProviderSideLoad:
    """CPU: host utilization over the `docker stats` window (docker.host_cpu_percent, 0-100
    across all cores) minus the core-percents of LIUM's containers only, in cores. A container
    Lium did not put there is the provider's own workload and stays in the total - subtracting
    it would let the provider mine another subnet with `docker run` and read as idle. Disk:
    hard_disk.used minus the docker breakdown (images+containers+volumes, DAH-2514), in kB —
    includes the OS baseline, so thresholds must leave room for it.

    Each part needs all its inputs present, numeric and well-shaped, else stays None — never
    a guess (AttributeError is caught alongside TypeError/ValueError so malformed specs can
    never abort the validation pipeline). The CPU part additionally requires EVERY listed
    container to carry a cpu_percent: a partial `docker stats` result would silently attribute
    the missing containers' CPU to the provider.
    """
    cpu_cores: float | None = None
    disk_kb: int | None = None

    try:
        cpu = specs.get("cpu") or {}
        docker_info = specs.get("docker") or {}
        containers = docker_info.get("containers") or []
        core_count = int(cpu.get("count"))
        host_percent = float(docker_info.get("host_cpu_percent"))
        raw_percents = [container.get("cpu_percent") for container in containers]
        if (
            lium_container_keys is not None
            and math.isfinite(host_percent)
            and host_percent >= 0
            and core_count > 0
            and raw_percents
            and None not in raw_percents
        ):
            container_percents = [float(percent) for percent in raw_percents]
            if all(math.isfinite(percent) and percent >= 0 for percent in container_percents):
                lium_percent = sum(
                    float(container["cpu_percent"])
                    for container in containers
                    if container.get("name") in lium_container_keys
                    or container.get("container_id") in lium_container_keys
                )
                host_cores = host_percent / 100 * core_count
                cpu_cores = round(max(0.0, host_cores - lium_percent / 100), 1)
    except (TypeError, ValueError, AttributeError, OverflowError):
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

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.PROVIDER_SIDE_LOAD_CHECK_ENABLED:
            event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
            return CheckResult(passed=True, event=event)

        lium_container_keys = lium_containers(ctx)
        specs = ctx.state.specs or {}
        provider_side_load = compute_provider_side_load(specs, lium_container_keys)
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

        # Same rented-status pattern as port_count.py / sysbox_required.py.
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        is_rented = rented_executor is not None and len(rented_executor.pods) > 0

        what = {
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
