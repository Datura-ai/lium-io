from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from ..messages import ProviderSideLoadMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

# DAH-2734: triage floors for a provider-side workload on a RENTED machine. Signals only —
# nothing gates or penalizes on them (a provider's nginx/fail2ban legitimately burns a little
# CPU); they exist so an SN13-class miner eating 9 cores and 1.5 TB is visible while it happens.
PROVIDER_SIDE_CPU_CORES_WARN = 2.0
PROVIDER_SIDE_DISK_WARN_KB = 100 * 1024 * 1024  # 100 GB outside docker


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
    def is_above_warn_floors(self) -> bool:
        return (
            (self.cpu_cores or 0.0) >= PROVIDER_SIDE_CPU_CORES_WARN
            or (self.disk_kb or 0) >= PROVIDER_SIDE_DISK_WARN_KB
        )

    def to_specs_fields(self) -> dict[str, float | int]:
        # specs travel to the backend as raw JSON, so the boundary needs a plain dict
        fields: dict[str, float | int] = {}
        if self.cpu_cores is not None:
            fields["cpu_cores"] = self.cpu_cores
        if self.disk_kb is not None:
            fields["disk_kb"] = self.disk_kb
        return fields


def compute_provider_side_load(specs: dict[str, Any]) -> ProviderSideLoad:
    """CPU: host utilization over the `docker stats` window (docker.host_cpu_percent, 0-100
    across all cores) minus the sum of per-container core-percents, in cores. Disk:
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
            math.isfinite(host_percent)
            and host_percent >= 0
            and core_count > 0
            and raw_percents
            and None not in raw_percents
        ):
            container_percents = [float(percent) for percent in raw_percents]
            if all(math.isfinite(percent) and percent >= 0 for percent in container_percents):
                host_cores = host_percent / 100 * core_count
                cpu_cores = round(max(0.0, host_cores - sum(container_percents) / 100), 1)
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


class ProviderSideLoadCheck:
    """Surface provider-side CPU/disk consumption on a rented machine as a triage signal.

    DAH-2734: an SN13 miner on a listed node burned 9 of 32 threads and filled 1.5 TB while the
    machine was rented, and no check even looked. This one computes what the host consumes
    outside Lium's docker containers from the specs the scrape already collected, stores it in
    specs as ``provider_side_load`` (so the backend keeps it queryable), and emits a warning
    event when a rented machine crosses the triage floors.

    Non-fatal and signals-only by design: it always passes, whatever it sees — a provider's own
    nginx also burns CPU, so these numbers guide triage, never money.
    """

    check_id = "host.validate.provider_side_load"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        provider_side_load = compute_provider_side_load(ctx.state.specs or {})
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
            "is_rented": is_rented,
        }
        should_warn = is_rented and provider_side_load.is_above_warn_floors
        template = Msg.LOAD_HIGH if should_warn else Msg.LOAD_RECORDED
        event = render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event, updates={"state": updated_state})
