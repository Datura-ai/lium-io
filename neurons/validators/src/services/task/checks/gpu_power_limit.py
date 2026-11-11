from __future__ import annotations

import json
import shlex
import time
from collections import Counter
from dataclasses import replace
from typing import Any

from pydantic import BaseModel
from services.const import DEFAULT_JOB_OWNER_LIUM, FILLER_CONTAINER_PREFIX
from services.gpu_power_limit import (
    MIN_POWER_LIMIT_RATIO,
    RENTAL_START_RATIO_TTL_SECONDS,
    RENTED_MIN_POWER_LIMIT_RATIO,
    SCORED_TIER_RATIO_TTL_SECONDS,
    STALE_CAP_GRACE_SECONDS,
    GpuPowerCapRevert,
    GpuPowerRestoreRecord,
    detect_cap_revert,
    read_gpu_power_restore_records,
    record_cap_reverts,
    rental_legacy_pod_key,
    rental_start_ratio_key,
    restore_tracked_gpu_power_limits,
    scored_tier_ratio_key,
)
from services.redis_service import RedisWrites

from ..messages import GpuPowerLimitMessages as Msg
from ..messages import render_message, shown
from ..models import ValidationEvent
from ..pipeline import CheckResult, Context


class GpuPowerMeasurement(BaseModel):
    index: int
    name: str | None
    uuid: str | None
    power_limit: float | None
    power_default_limit: float | None
    # Host floor for the cap; None on a validator scrape that predates the field.
    power_min_limit: float | None = None
    power_max_limit: float | None
    power_limit_ratio: float | None = None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _with_host_limits(ctx: Context, records: list[GpuPowerRestoreRecord]) -> dict[str, Any]:
    """Specs with ``power_host_limit`` (watts) on each GPU under our live Pearl cap: the host's own limit,
    recorded before the cap and restored when the filler stops. The backend's Reduced power limit badge
    judges it while ``power_limit`` stays the live cap (DAH-5302). No record = no field."""
    host_watts = {record.gpu_uuid: record.watts for record in records}
    gpu = (ctx.state.specs or {}).get("gpu")
    if not host_watts or not isinstance(gpu, dict) or not isinstance(gpu.get("details"), list):
        return {}
    details = [
        {**detail, "power_host_limit": host_watts[detail["uuid"]]}
        if isinstance(detail, dict) and detail.get("uuid") in host_watts
        else detail
        for detail in gpu["details"]
    ]
    return {"state": replace(ctx.state, specs={**ctx.state.specs, "gpu": {**gpu, "details": details}})}


def _dump_measurements(measurements: list[GpuPowerMeasurement]) -> list[dict[str, Any]]:
    return [measurement.model_dump() for measurement in measurements]


FILLER_INSPECT_TIMEOUT_SECONDS = 30


async def _filler_pods_running(ctx: Context, pod_ids: set[str]) -> set[str]:
    """The pods among these whose filler container the host reports as running now (DAH-2786). A pod the host does
    not answer for is not in the result."""
    if ctx.ssh is None or not pod_ids:
        return set()
    names = " ".join(shlex.quote(f"{FILLER_CONTAINER_PREFIX}{pod_id}") for pod_id in sorted(pod_ids))
    try:
        # exits 1 when one name is missing but still prints the others
        result = await ctx.ssh.run(
            f"/usr/bin/docker inspect --format '{{{{.Name}}}} {{{{.State.Running}}}}' {names}",
            timeout=FILLER_INSPECT_TIMEOUT_SECONDS,
        )
    except Exception:  # noqa: BLE001
        return set()
    running: set[str] = set()
    for line in (getattr(result, "stdout", "") or "").splitlines():
        name, _, state = line.strip().lstrip("/").partition(" ")
        if state == "true" and name.startswith(FILLER_CONTAINER_PREFIX):
            running.add(name.removeprefix(FILLER_CONTAINER_PREFIX))
    return running & pod_ids


class GpuPowerLimitCheck:
    """Gate validation on the current GPU power cap relative to the default cap."""

    check_id = "gpu.validate.power_limit"
    fatal = True

    def __init__(self, restore_stale_caps: bool = True):
        # False in the dry-run pipeline: the verdict logic still runs, but the check must not
        # mutate executor state (nvidia-smi -pl) or consume shared Redis restore records.
        self.restore_stale_caps = restore_stale_caps

    async def run(self, ctx: Context) -> CheckResult:
        # DAH-2356: if Lium is running its own default job (e.g. the PEARL idle filler) on this node,
        # WE lowered the power limit on purpose, so a below-default reading is expected. Skip the
        # penalty — the node keeps its score and stays rentable; the pre-cap limit is restored when the
        # filler stops. Scoped to owner="lium" only: a miner's own default job gets no power-limit pass.
        rented_data = ctx.state.rented_data
        default_job_owner: str | None = (
            rented_data.get_default_job_owner(ctx.executor.uuid) if rented_data else None
        )
        if default_job_owner == DEFAULT_JOB_OWNER_LIUM:
            # DAH-2786: this is the only moment we can tell a host that raises our cap back
            # WHILE our job runs from one that merely resets the limit after the job ends. The
            # first kills the job it is paid to host; the second harms nothing.
            records = await self._record_reverts_of_our_live_cap(ctx)
            event = render_message(
                Msg.SKIPPED_ACTIVE_LIUM_FILLER,
                ctx=ctx,
                check_id=self.check_id,
                what={"executor_uuid": ctx.executor.uuid},
            )
            return CheckResult(passed=True, event=event, updates=_with_host_limits(ctx, records))

        measurements: list[GpuPowerMeasurement] = []
        incomplete: list[GpuPowerMeasurement] = []
        rejected: list[GpuPowerMeasurement] = []
        floors = await self._rented_floors(ctx)

        for index, detail in enumerate(ctx.state.gpu_details or []):
            current_limit = _to_float(detail.get("power_limit"))
            default_limit = _to_float(detail.get("power_default_limit"))
            max_limit = _to_float(detail.get("power_max_limit"))
            measurement = GpuPowerMeasurement(
                index=index,
                name=detail.get("name"),
                uuid=detail.get("uuid"),
                power_limit=current_limit,
                power_default_limit=default_limit,
                power_min_limit=_to_float(detail.get("power_min_limit")),
                power_max_limit=max_limit,
            )

            if current_limit is None or default_limit is None or default_limit <= 0:
                incomplete.append(measurement)
                continue

            ratio = current_limit / default_limit
            measurement.power_limit_ratio = round(ratio, 4)
            measurements.append(measurement)

            if ratio < floors.get(measurement.uuid or "", floors.get("", MIN_POWER_LIMIT_RATIO)):
                rejected.append(measurement)

        if rejected:
            stale_cap_event = await self._rescue_stale_lium_caps(ctx, rejected)
            if stale_cap_event is not None:
                return CheckResult(passed=True, event=stale_cap_event)
            measured = "; ".join(
                f"GPU {m.index}: {shown(m.power_limit, 0, failed_below=True)} W limit, "
                f"default {m.power_default_limit:.0f} W "
                f"({shown(m.power_limit / m.power_default_limit * 100, 0, failed_below=True)}%)"
                for m in rejected
            )
            event = render_message(
                Msg.LIMIT_BELOW_DEFAULT,
                ctx=ctx,
                check_id=self.check_id,
                remediation=(
                    f"{measured}. {Msg.LIMIT_BELOW_DEFAULT.remediation} Restore it with "
                    "`sudo nvidia-smi -i <GPU> -pl <default W>`."
                ),
                what={
                    "threshold": MIN_POWER_LIMIT_RATIO,
                    "rejected_gpus": _dump_measurements(rejected),
                    "measurements": _dump_measurements(measurements),
                    "incomplete_gpus": _dump_measurements(incomplete),
                },
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={
                    "score": 0.0,
                    "job_score": 0.0,
                    "score_warning": (
                        " WARNING: GPU power limit is below 80% of the default power limit,"
                        " or was lowered below 90% during a rental"
                    ),
                },
            )

        if incomplete:
            # No reading is no proof of a full limit: a host that hides the value from NVML fails.
            event = render_message(
                Msg.DATA_INCOMPLETE_REJECTED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "threshold": MIN_POWER_LIMIT_RATIO,
                    "measurements": _dump_measurements(measurements),
                    "incomplete_gpus": _dump_measurements(incomplete),
                },
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={
                    "score": 0.0,
                    "job_score": 0.0,
                    "score_warning": " WARNING: GPU power limit could not be read",
                },
            )

        if self.restore_stale_caps and not await self._remember_tier_ratios(ctx, measurements):  # the dry run writes nothing
            event = render_message(
                Msg.TIER_STATE_UNKNOWN,
                ctx=ctx,
                check_id=self.check_id,
                what={"threshold": MIN_POWER_LIMIT_RATIO, "measurements": _dump_measurements(measurements)},
            )
            return CheckResult(passed=False, event=event)
        event = render_message(
            Msg.LIMIT_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "threshold": MIN_POWER_LIMIT_RATIO,
                "measurements": _dump_measurements(measurements),
            },
        )
        return CheckResult(passed=True, event=event)

    async def _remember_tier_ratios(self, ctx: Context, measurements: list[GpuPowerMeasurement]) -> bool:
        """Keep each GPU's passing reading in the 80 % tier (and clear it at stock) for a rental-start
        read that misses the GPU: it then starts at what the renter's badge showed, not at stock (F-36).

        False when a tier reading may still be kept after a stock reading (Redis failed): the cycle then gives
        no verdict, so the node is neither zeroed by our outage nor shown at stock while 0.8 is kept (F-43)."""
        writes, stock_keys = RedisWrites(), []
        for m in measurements:
            if not m.uuid or m.power_limit_ratio is None:
                continue
            key = scored_tier_ratio_key(ctx.executor.uuid, m.uuid)
            if m.power_limit_ratio < RENTED_MIN_POWER_LIMIT_RATIO:
                writes.set(key, str(m.power_limit_ratio), ex=SCORED_TIER_RATIO_TTL_SECONDS)
            else:
                writes.delete(key)
                stock_keys.append(key)
        try:
            kept: bool | None = any(await ctx.services.redis.mget(stock_keys)) if stock_keys else False
        except Exception:
            kept = None  # unknown, never taken as proof that nothing is kept
        for _ in range(2):
            try:
                await ctx.services.redis.write_atomically(writes)
                return True
            except Exception:
                continue
        # a lost set leaves a missed GPU at stock, the safe side; a lost clear is safe only when nothing was kept
        return kept is False

    async def _rented_floors(self, ctx: Context) -> dict[str, float]:
        """Per-GPU floor while the node is rented, by GPU uuid; "" is the floor for any other GPU.

        A GPU whose rental-start record names a pod rented now keeps min(its ratio then,
        RENTED_MIN_POWER_LIMIT_RATIO), never under MIN_POWER_LIMIT_RATIO, so a renter who rented
        it in the 80 % tier keeps that cap and nobody else's power drops mid-rental. Other GPUs
        get MIN_POWER_LIMIT_RATIO only when every settled pod has a record for each of its
        `gpu_count` GPUs (so they are not rented: a split node's free bundle), else
        RENTED_MIN_POWER_LIMIT_RATIO. A pod whose container the backend has not recorded yet is
        still being created; its records are written before its container starts, so it is not
        judged. A settled pod with no record at all is a rental from before the records: it is
        marked so its next create (a reboot or edit) records 1.0, never the limit of that day.
        Unrented node, or Redis did not answer: {} = MIN_POWER_LIMIT_RATIO everywhere; our own
        outage must not zero a node.
        """
        rented_data = ctx.state.rented_data
        rented = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        if rented is None or not rented.pods:
            return {}
        uuids = [str(detail.get("uuid")) for detail in ctx.state.gpu_details or [] if detail.get("uuid")]
        try:
            raw = await ctx.services.redis.mget([rental_start_ratio_key(ctx.executor.uuid, u) for u in uuids])
        except Exception:
            return {}
        rented_pod_ids = {pod.pod_id for pod in rented.pods}
        floors: dict[str, float] = {}
        records_by_pod: Counter[str] = Counter()
        for gpu_uuid, value in zip(uuids, raw):
            if not value:
                continue
            try:
                record = json.loads(value)
                pod_id, ratio = str(record["pod_id"]), float(record["ratio"])
            except (ValueError, TypeError, KeyError):
                continue  # a corrupt record proves nothing: the GPU takes the "" floor
            if pod_id in rented_pod_ids:
                records_by_pod[pod_id] += 1
                # 0.005 slack: the record is read from nvidia-smi in whole watts, the check from the scrape
                floors[gpu_uuid] = max(MIN_POWER_LIMIT_RATIO, min(RENTED_MIN_POWER_LIMIT_RATIO, ratio - 0.005))
        settled = [pod for pod in rented.pods if not pod.container_name_pending]
        every_gpu_recorded = all(
            pod.gpu_count is not None and records_by_pod[pod.pod_id] >= pod.gpu_count for pod in settled
        )
        floors[""] = MIN_POWER_LIMIT_RATIO if every_gpu_recorded else RENTED_MIN_POWER_LIMIT_RATIO
        if self.restore_stale_caps:  # the dry-run pipeline writes nothing
            for pod in settled:
                if not records_by_pod[pod.pod_id]:
                    try:
                        await ctx.services.redis.set(
                            rental_legacy_pod_key(pod.pod_id), "1", ex=RENTAL_START_RATIO_TTL_SECONDS
                        )
                    except Exception:
                        pass  # marked on a later cycle
        return floors

    async def _record_reverts_of_our_live_cap(self, ctx: Context) -> list[GpuPowerRestoreRecord]:
        """Note every GPU of this executor whose live limit sits above the cap we applied and
        verified for the Lium job running right now (DAH-2786).

        Read-only towards the host: it changes no limit, it only records what it saw, so the
        filler keeps running either way. The unrented incentive gate counts these later.
        Returns this executor's restore records (the host's own limits under the cap).
        """
        live_watts_by_uuid: dict[str, GpuPowerMeasurement] = {
            str(detail.get("uuid")): GpuPowerMeasurement(
                index=index,
                name=detail.get("name"),
                uuid=str(detail.get("uuid")),
                power_limit=_to_float(detail.get("power_limit")),
                power_default_limit=_to_float(detail.get("power_default_limit")),
                power_max_limit=_to_float(detail.get("power_max_limit")),
            )
            for index, detail in enumerate(ctx.state.gpu_details or [])
            if detail.get("uuid")
        }
        if not live_watts_by_uuid:
            return []
        read_result = await read_gpu_power_restore_records(
            ctx.services.redis, list(live_watts_by_uuid), log_extra=ctx.default_extra
        )
        records = [record for record in read_result.records if record.executor_id == ctx.executor.uuid]
        # same reason the stale-cap rescue is gated: the dry-run pipeline must not write shared state
        if not self.restore_stale_caps:
            return records
        observed_at = time.time()
        reverts: list[GpuPowerCapRevert] = []
        for record in records:
            measurement = live_watts_by_uuid.get(record.gpu_uuid)
            if measurement is None or measurement.power_limit is None:
                continue
            revert = detect_cap_revert(
                record, measurement.power_limit, measurement.power_default_limit, observed_at
            )
            if revert is not None:
                reverts.append(revert)
        if reverts:
            # A claim the delete could not clear (Redis down through the teardown) outlives its job, and the backend
            # still reports owner "lium" for a STOPPING run: only a job whose container runs now can be reverted
            running = await _filler_pods_running(ctx, {revert.pod_id for revert in reverts})
            reverts = [revert for revert in reverts if revert.pod_id in running]
        if reverts:
            await record_cap_reverts(
                ctx.services.redis, ctx.executor.uuid, reverts, ctx.default_extra
            )
        return records

    async def _rescue_stale_lium_caps(
        self, ctx: Context, rejected: list[GpuPowerMeasurement]
    ) -> ValidationEvent | None:
        """DAH-2356 safety net: skip the penalty when EVERY below-floor GPU is our own stale filler cap.

        A restore record is written only by this validator when it caps a filler on this executor, so
        a below-floor reading covered by such a record is Lium's doing, not the miner's — never
        penalize it. Restoring (which also deletes the record) waits out STALE_CAP_GRACE_SECONDS and
        requires backend data, so a live filler is never uncapped. Any rejected GPU WITHOUT a matching
        record is a genuine miner-side violation → no rescue, normal penalty. Exception: when the
        Redis read itself FAILED, an uncovered GPU may still be our own cap whose record we simply
        could not read — penalizing would zero an innocent miner over our own outage, so the check
        passes for this cycle and re-runs once Redis answers.

        Returns the pass event, or None when the normal below-default penalty must apply.
        """
        rejected_uuids: list[str] = [
            measurement.uuid for measurement in rejected if measurement.uuid
        ]
        if len(rejected_uuids) < len(rejected):
            return None  # a rejected GPU without a uuid can't be matched to a record
        read_result = await read_gpu_power_restore_records(
            ctx.services.redis, rejected_uuids, log_extra=ctx.default_extra
        )
        own_records: list[GpuPowerRestoreRecord] = [
            record for record in read_result.records if record.executor_id == ctx.executor.uuid
        ]
        covered_uuids = {record.gpu_uuid for record in own_records}
        uncovered_uuids = [uuid for uuid in rejected_uuids if uuid not in covered_uuids]
        if uncovered_uuids:
            if read_result.read_failed:
                return render_message(
                    Msg.RESCUE_STATE_UNAVAILABLE,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "executor_uuid": ctx.executor.uuid,
                        "unmatched_gpu_uuids": uncovered_uuids,
                    },
                )
            return None
        stale_records = [
            record for record in own_records
            if time.time() - record.capped_at >= STALE_CAP_GRACE_SECONDS
        ]
        records_restored = 0
        if self.restore_stale_caps and stale_records and ctx.state.rented_data is not None:
            records_restored = await restore_tracked_gpu_power_limits(
                ctx.ssh,
                ctx.services.redis,
                [record.gpu_uuid for record in stale_records],
                log_extra=ctx.default_extra,
            )
        return render_message(
            Msg.RESTORED_STALE_CAP,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "executor_uuid": ctx.executor.uuid,
                "records_found": len(own_records),
                "records_eligible_for_restore": len(stale_records),
                "records_restored": records_restored,
            },
        )
