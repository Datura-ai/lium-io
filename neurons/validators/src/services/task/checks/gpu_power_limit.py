from __future__ import annotations

import logging
import time
from typing import Any

from core.config import settings
from core.utils import _m, get_extra_info
from pydantic import BaseModel
from services.const import DEFAULT_JOB_OWNER_LIUM
from services.gpu_power_limit import (
    MIN_POWER_LIMIT_RATIO,
    STALE_CAP_GRACE_SECONDS,
    GpuPowerRestoreRecord,
    read_gpu_power_restore_records,
    restore_tracked_gpu_power_limits,
)

from ..messages import GpuPowerLimitMessages as Msg
from ..messages import render_message
from ..models import ValidationEvent
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

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
        measurements: list[GpuPowerMeasurement] = []
        incomplete: list[GpuPowerMeasurement] = []
        rejected: list[GpuPowerMeasurement] = []

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

            if ratio < MIN_POWER_LIMIT_RATIO:
                rejected.append(measurement)

        # DAH-2356: while Lium runs its own default job (e.g. the PEARL idle filler) on this node, WE
        # may have lowered the power limit on purpose. Scoped to owner="lium" only: a miner's own
        # default job gets no power-limit pass.
        rented_data = ctx.state.rented_data
        default_job_owner: str | None = (
            rented_data.get_default_job_owner(ctx.executor.uuid) if rented_data else None
        )
        if default_job_owner == DEFAULT_JOB_OWNER_LIUM:
            return await self._verdict_under_lium_filler(ctx, rejected, measurements, incomplete)

        if rejected:
            stale_cap_event = await self._rescue_stale_lium_caps(ctx, rejected)
            if stale_cap_event is not None:
                return CheckResult(passed=True, event=stale_cap_event)
            return self._below_floor_result(ctx, rejected, measurements, incomplete)

        if incomplete:
            event = render_message(
                Msg.DATA_INCOMPLETE,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "threshold": MIN_POWER_LIMIT_RATIO,
                    "measurements": _dump_measurements(measurements),
                    "incomplete_gpus": _dump_measurements(incomplete),
                },
            )
            return CheckResult(passed=True, event=event)

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

    def _below_floor_result(
        self,
        ctx: Context,
        rejected: list[GpuPowerMeasurement],
        measurements: list[GpuPowerMeasurement],
        incomplete: list[GpuPowerMeasurement],
    ) -> CheckResult:
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
                    " WARNING: GPU power limit is below 90% of the default power limit"
                ),
            },
        )

    async def _verdict_under_lium_filler(
        self,
        ctx: Context,
        rejected: list[GpuPowerMeasurement],
        measurements: list[GpuPowerMeasurement],
        incomplete: list[GpuPowerMeasurement],
    ) -> CheckResult:
        """DAH-3630: the verdict while a Lium filler runs on this node.

        A GPU at or above the floor passes, whatever the host did to our cap: a host that raised
        the limit back keeps PEARL and its unrented incentive. A below-floor GPU passes only when
        this validator holds a restore record for it on this executor, i.e. Lium capped it. A
        below-floor GPU without one is the host's own limit (the cap failed and PEARL started
        uncapped, or a filler that never caps): it earns nothing under the floor rule, so with
        ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS the node is scored like any below-floor
        node; without it the breach is logged and the node passes. Unknowns pass (a GPU without a
        uuid, a failed Redis read). Never restores: the filler is live.
        """
        skipped = render_message(
            Msg.SKIPPED_ACTIVE_LIUM_FILLER,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid},
        )
        if not rejected or any(measurement.uuid is None for measurement in rejected):
            return CheckResult(passed=True, event=skipped)
        read_result = await read_gpu_power_restore_records(
            ctx.services.redis,
            [measurement.uuid for measurement in rejected if measurement.uuid],
            log_extra=ctx.default_extra,
        )
        capped_by_lium: set[str] = {
            record.gpu_uuid for record in read_result.records if record.executor_id == ctx.executor.uuid
        }
        host_limited: list[GpuPowerMeasurement] = [
            measurement for measurement in rejected if measurement.uuid not in capped_by_lium
        ]
        if not host_limited:
            return CheckResult(passed=True, event=skipped)
        if read_result.read_failed:
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.RESCUE_STATE_UNAVAILABLE,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "executor_uuid": ctx.executor.uuid,
                        "unmatched_gpu_uuids": [measurement.uuid for measurement in host_limited],
                    },
                ),
            )
        enforced: bool = settings.ENABLE_POWER_FLOOR_FOR_UNCAPPED_LIUM_FILLER_GPUS
        logger.info(
            _m(
                "Lium filler runs on a GPU the host holds below the power floor, not capped by Lium"
                + ("" if enforced else " (shadow only - flag off)"),
                extra=get_extra_info(
                    {
                        **ctx.default_extra,
                        "executor_uuid": ctx.executor.uuid,
                        "host_limited_gpus": _dump_measurements(host_limited),
                        "threshold": MIN_POWER_LIMIT_RATIO,
                        "enforced": enforced,
                        "reason": "power_floor_uncapped_lium_filler_gpu",
                    }
                ),
            )
        )
        if not enforced:
            return CheckResult(passed=True, event=skipped)
        return self._below_floor_result(ctx, host_limited, measurements, incomplete)

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
