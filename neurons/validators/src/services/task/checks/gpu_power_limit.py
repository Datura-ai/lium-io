from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel
from services.const import DEFAULT_JOB_OWNER_LIUM
from services.gpu_power_limit import (
    MIN_POWER_LIMIT_RATIO,
    RENTED_MIN_POWER_LIMIT_RATIO,
    STALE_CAP_GRACE_SECONDS,
    GpuPowerRestoreRecord,
    read_gpu_power_restore_records,
    rental_start_ratio_key,
    restore_tracked_gpu_power_limits,
)

from ..messages import GpuPowerLimitMessages as Msg
from ..messages import render_message
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


def _dump_measurements(measurements: list[GpuPowerMeasurement]) -> list[dict[str, Any]]:
    return [measurement.model_dump() for measurement in measurements]


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
            event = render_message(
                Msg.SKIPPED_ACTIVE_LIUM_FILLER,
                ctx=ctx,
                check_id=self.check_id,
                what={"executor_uuid": ctx.executor.uuid},
            )
            return CheckResult(passed=True, event=event)

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
            event = render_message(
                Msg.LIMIT_BELOW_DEFAULT,
                ctx=ctx,
                check_id=self.check_id,
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

    async def _rented_floors(self, ctx: Context) -> dict[str, float]:
        """Per-GPU floor while the node is rented, by GPU uuid; "" is the floor for any other GPU.

        A GPU whose rental-start record names a pod rented now keeps min(its ratio then,
        RENTED_MIN_POWER_LIMIT_RATIO), never under MIN_POWER_LIMIT_RATIO, so a renter who rented
        it in the 80 % tier keeps that cap and nobody else's power drops mid-rental. Other GPUs
        get MIN_POWER_LIMIT_RATIO when every rented pod has a record (they are not rented: a split
        node's free bundle), else RENTED_MIN_POWER_LIMIT_RATIO (a rental from before the records).
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
        recorded_pods: set[str] = set()
        for gpu_uuid, value in zip(uuids, raw):
            if not value:
                continue
            try:
                record = json.loads(value)
                pod_id, ratio = str(record["pod_id"]), float(record["ratio"])
            except (ValueError, TypeError, KeyError):
                continue  # a corrupt record proves nothing: the GPU takes the "" floor
            if pod_id in rented_pod_ids:
                recorded_pods.add(pod_id)
                # 0.005 slack: the record is read from nvidia-smi in whole watts, the check from the scrape
                floors[gpu_uuid] = max(MIN_POWER_LIMIT_RATIO, min(RENTED_MIN_POWER_LIMIT_RATIO, ratio - 0.005))
        floors[""] = MIN_POWER_LIMIT_RATIO if rented_pod_ids <= recorded_pods else RENTED_MIN_POWER_LIMIT_RATIO
        return floors

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
