"""RENTED_GPU_DROP — a rented node shows fewer GPUs than it rents, or NVML failed listing them.

A card that falls off the bus on a rented node breaks the renter's workload, and without this check nobody
is told until the next fatal cycle is investigated. The scrape's per-device loop stops at the first NVML
error, so `gpu.count` keeps the driver's count while `gpu.details` holds only the cards read before it and
`gpu_scrape_error` carries the NVML error.

The fatal checks after the scrape (GpuModelValidCheck's DETAILS_MISMATCH, GpuFingerprintCheck) halt such a
cycle long before TenantEnforcementCheck, so this check runs right after MachineSpecScrapeCheck and never
fails the cycle. For a rented executor it compares what the scrape listed with:

- the GPUs the rental holds (sum of the pods' `gpu_count`, when the backend sent every one),
- the driver's own count (`gpu.count`),
- the anchored UUID set of the verified-job record (`missing_uuids` is the anchor minus the listed set),

and reads the NVML error code out of `gpu_scrape_error`. A GPU-loss code (NVML_GPU_LOSS_CODES) is a fault
on its own; any other scrape error only rides along with a count fault.

The counts are executor-wide, and a fault is posted for every RUNNING pod on the executor; each post
carries that pod's own `gpu_count` next to the executor totals so the backend can tell which renters of
a split node are affected.

Per RUNNING pod a Redis mark `rented_gpu_drop:<pod_id>` holds the incident: `first_seen_at`,
`consecutive_cycles`, `reported` (the backend answered with a delivery that needs no retry) and
`recorded` (the backend holds the incident, so it must hear the recovery). The first faulty cycle posts
`POST /internal/pods/{pod_id}/gpu-drop` with `state=fault`; later cycles post again only while `reported`
is False. A fault that rests only on detail rows cut short by a non-loss NVML error, while the driver's
count still covers the rental and the anchor (`confirm_first`), posts from its second consecutive cycle
instead. The first clean cycle after a recorded incident posts
`state=recovered` and deletes the mark once the backend answered. A backend that is down or older (404) is
no answer: the next cycle asks again. Redis down: the fault is still posted every cycle (the backend keeps
one open incident per pod, so the renter is told once), except a `confirm_first` one, which cannot count
cycles, and the recovery is not; the check's verdict never depends on Redis.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import redis.exceptions
from protocol.vc_protocol.compute_requests import (
    GPU_DROP_DELIVERY_DISABLED,
    GPU_DROP_DELIVERY_NOT_RENTED,
    GPU_DROP_DELIVERY_NOTIFY_FAILED,
    RentedGpuDropResponse,
    RentedPod,
)
from services.redis_service import GPU_ANCHOR_KEY

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import RentedGpuDropMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .gpu_fingerprint import split_uuids

logger = logging.getLogger(__name__)

RENTED_GPU_DROP_KEY_PREFIX = "rented_gpu_drop"
# the backend's pod status (RentedPod.status) for a pod the renter is using
POD_STATUS_RUNNING = "RUNNING"
# a Redis outage surfaces as RedisError or, below the client, as a socket OSError
REDIS_ERRORS: tuple[type[BaseException], ...] = (redis.exceptions.RedisError, OSError)

FAULT_BELOW_RENTED = "below_rented_count"
FAULT_DETAILS_SHORT = "details_short_of_count"
FAULT_ANCHORED_MISSING = "anchored_gpu_missing"
FAULT_NVML_ERROR = "nvml_error"

# NVML return codes that mean a card is gone or unusable: DRIVER_NOT_LOADED, GPU_IS_LOST, RESET_REQUIRED,
# GPU_NOT_FOUND, UNKNOWN (what a card that fell off the bus answers). Any other code (NOT_SUPPORTED on
# an optional query, say) is not a fault unless a card is also missing.
NVML_GPU_LOSS_CODES = frozenset({9, 15, 16, 28, 999})
_NVML_ERROR_CODE = re.compile(r"NVMLError\w*\((\d{1,6})\)")

# The scrape and the anchor are provider-reported: bound what goes into the event and the POST.
MAX_MISSING_UUIDS = 16
MAX_UUID_CHARS = 64
MAX_SCRAPE_ERROR_CHARS = 120
# The backend refuses (422) a GPU count above this or a cycle count above MAX_REPORTED_CYCLES; an out-of-range
# value is clamped and logged so the report is not refused on every cycle.
MAX_REPORTED_GPU_COUNT = 64
MAX_REPORTED_CYCLES = 100_000

STATE_FAULT = "fault"
STATE_RECOVERED = "recovered"

# A delivery that needs no retry of the fault report, and one that means the backend holds the incident.
_NO_RETRY_DELIVERIES = frozenset({None, "notified", "recorded", GPU_DROP_DELIVERY_NOT_RENTED})
_NOT_RECORDED_DELIVERIES = frozenset({GPU_DROP_DELIVERY_DISABLED, GPU_DROP_DELIVERY_NOT_RENTED})


@dataclass(frozen=True)
class GpuDrop:
    """What the scrape shows against what the rental should see; only built when there is a fault."""

    expected: int
    visible: int
    nvml_count: int
    missing_uuids: list[str]
    nvml_error_code: int | None
    faults: list[str]
    # the detail rows are short only because a non-loss NVML error (a timeout, say) cut the scrape's
    # loop while the driver still counts every rented and anchored card: a one-off glitch looks the
    # same, so the report waits for a second faulty cycle
    confirm_first: bool = False


def nvml_error_code(scrape_error: object) -> int | None:
    """The NVML return code in the scrape's `gpu_scrape_error` (`repr` of the exception), else None."""
    if not isinstance(scrape_error, str):
        return None
    match = _NVML_ERROR_CODE.search(scrape_error)
    return int(match.group(1)) if match else None


def judge_rented_gpus(
    *,
    rented_gpu_count: int | None,
    anchor_uuids: list[str],
    nvml_count: int,
    listed_uuids: list[str],
    listed_count: int,
    scrape_error: object,
) -> GpuDrop | None:
    """The fault this scrape shows on a rented node, or None when every rented card is there.

    The count faults read the number of detail rows: a scrape can list one card twice (`split_uuids`,
    GpuFingerprintCheck's `duplicate_uuid`), so the distinct UUID set can be one short on a healthy node.
    Only `anchored_gpu_missing` reads the distinct set.
    """
    listed = set(listed_uuids)
    visible = listed_count
    anchor = set(anchor_uuids)
    code = nvml_error_code(scrape_error)
    missing = sorted(anchor - listed)[:MAX_MISSING_UUIDS] if listed or not listed_count else []

    faults: list[str] = []
    if rented_gpu_count and visible < rented_gpu_count:
        faults.append(FAULT_BELOW_RENTED)
    if visible < nvml_count:
        faults.append(FAULT_DETAILS_SHORT)
    if missing:
        faults.append(FAULT_ANCHORED_MISSING)
    if code in NVML_GPU_LOSS_CODES or (scrape_error and faults):
        faults.append(FAULT_NVML_ERROR)
    if not faults:
        return None
    return GpuDrop(
        expected=max(rented_gpu_count or 0, len(anchor), nvml_count),
        visible=visible,
        nvml_count=nvml_count,
        missing_uuids=[uuid[:MAX_UUID_CHARS] for uuid in missing],
        nvml_error_code=code,
        faults=faults,
        confirm_first=bool(scrape_error)
        and code not in NVML_GPU_LOSS_CODES
        and FAULT_DETAILS_SHORT in faults
        and nvml_count >= max(rented_gpu_count or 0, len(anchor)),
    )


@dataclass(frozen=True)
class ExecutorGpus:
    """The executor-wide counts every pod's report carries: the rental's total and the driver's count."""

    rented_gpu_count: int | None
    nvml_gpu_count: int


@dataclass(frozen=True)
class DropMark:
    """One pod's open incident, as kept in Redis between cycles."""

    first_seen_at: str
    consecutive_cycles: int = 1
    reported: bool = False
    recorded: bool = False

    @classmethod
    def load(cls, raw: object) -> DropMark | None:
        if isinstance(raw, bytes):
            raw = raw.decode()
        if not isinstance(raw, str):
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(data, dict) or not isinstance(data.get("first_seen_at"), str):
            return None
        cycles = data.get("consecutive_cycles")
        return cls(
            first_seen_at=data["first_seen_at"],
            consecutive_cycles=cycles if isinstance(cycles, int) and cycles > 0 else 1,
            reported=data.get("reported") is True,
            recorded=data.get("recorded") is True,
        )

    def dump(self) -> str:
        return json.dumps(
            {
                "first_seen_at": self.first_seen_at,
                "consecutive_cycles": self.consecutive_cycles,
                "reported": self.reported,
                "recorded": self.recorded,
            }
        )

    def after_answer(self, answer: RentedGpuDropResponse) -> DropMark:
        return replace(
            self,
            reported=answer.delivery in _NO_RETRY_DELIVERIES,
            recorded=self.recorded or answer.delivery not in _NOT_RECORDED_DELIVERIES,
        )


@dataclass(frozen=True)
class PodDropOutcome:
    pod_id: str
    state: str
    first_seen_at: str
    consecutive_cycles: int
    # this cycle posted and the backend answered; its delivery, None when nothing was posted or answered
    posted: bool = False
    delivery: str | None = None
    reported: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def log_fields(self) -> dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "state": self.state,
            "first_seen_at": self.first_seen_at,
            "consecutive_cycles": self.consecutive_cycles,
            "posted": self.posted,
            "delivery": self.delivery,
            "reported": self.reported,
            **self.extra,
        }


def _key(pod_id: str) -> str:
    return f"{RENTED_GPU_DROP_KEY_PREFIX}:{pod_id}"


def _clamp(ctx: Context, pod_id: str, name: str, value: int, cap: int) -> int:
    bounded = min(max(value, 0), cap)
    if bounded != value:
        logger.warning(
            _m(
                "RENTED_GPU_DROP_VALUE_CLAMPED",
                extra=get_extra_info(
                    {
                        **ctx.default_extra,
                        "pod_id": pod_id,
                        "field": name,
                        "value": value,
                        "sent": bounded,
                    }
                ),
            )
        )
    return bounded


def _listed_uuids(gpu_details: list[dict]) -> list[str]:
    return [
        str(detail["uuid"])
        for detail in gpu_details
        if isinstance(detail, dict) and detail.get("uuid")
    ]


def _rented_gpu_count(pods: list[RentedPod]) -> int | None:
    counts = [pod.gpu_count for pod in pods]
    return sum(counts) if counts and all(counts) else None


class RentedGpuDropCheck:
    """Report a rented node that lost a GPU on the cycle it is seen, once per incident per pod.

    Non-fatal and never changes the score: it runs before the fatal GPU checks so a cycle those checks halt
    still reports. See the module docstring for the rule and the Redis mark.
    """

    check_id = "gpu.validate.rented_gpu_drop"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.RENTED_GPU_DROP_CHECK_ENABLED:
            return CheckResult(
                passed=True, event=render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id)
            )

        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        all_pods = list(rented_executor.pods) if rented_executor else []
        pods = [pod for pod in all_pods if pod.status in (None, POD_STATUS_RUNNING)]
        if not pods:
            event = render_message(
                Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id, what={"listed_pods": len(all_pods)}
            )
            return CheckResult(passed=True, event=event)

        specs = ctx.state.specs or {}
        gpu_details = list(ctx.state.gpu_details or [])
        nvml_count = ctx.state.gpu_count
        if nvml_count is None:
            nvml_count = (specs.get("gpu") or {}).get("count", 0) or 0
        scrape_error = specs.get("gpu_scrape_error")
        totals = ExecutorGpus(
            rented_gpu_count=_rented_gpu_count(all_pods), nvml_gpu_count=nvml_count
        )
        drop = judge_rented_gpus(
            rented_gpu_count=totals.rented_gpu_count,
            anchor_uuids=split_uuids((ctx.verified or {}).get(GPU_ANCHOR_KEY) or ""),
            nvml_count=nvml_count,
            listed_uuids=_listed_uuids(gpu_details),
            listed_count=len(gpu_details),
            scrape_error=scrape_error,
        )

        now_iso = datetime.now(UTC).isoformat()
        outcomes = [
            outcome
            for pod in pods
            if (outcome := await self._track_pod(ctx, pod, drop, totals, now_iso)) is not None
        ]

        if drop is not None:
            what = {
                "executor_uuid": ctx.executor.uuid,
                "expected_gpu_count": drop.expected,
                "visible_gpu_count": drop.visible,
                "rented_gpu_count": totals.rented_gpu_count,
                "nvml_gpu_count": drop.nvml_count,
                "missing_uuids": drop.missing_uuids,
                "nvml_error_code": drop.nvml_error_code,
                "gpu_scrape_error": scrape_error[:MAX_SCRAPE_ERROR_CHARS]
                if isinstance(scrape_error, str)
                else None,
                "faults": drop.faults,
                "pods": [outcome.log_fields() for outcome in outcomes],
            }
            logger.warning(
                _m("RENTED_GPU_DROP", extra=get_extra_info({**ctx.default_extra, **what}))
            )
            return CheckResult(
                passed=True,
                event=render_message(Msg.DROP, ctx=ctx, check_id=self.check_id, what=what),
            )

        if outcomes:
            what = {
                "executor_uuid": ctx.executor.uuid,
                "pods": [outcome.log_fields() for outcome in outcomes],
            }
            logger.info(
                _m("RENTED_GPU_RECOVERED", extra=get_extra_info({**ctx.default_extra, **what}))
            )
            return CheckResult(
                passed=True,
                event=render_message(Msg.RECOVERED, ctx=ctx, check_id=self.check_id, what=what),
            )

        event = render_message(
            Msg.OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"visible_gpu_count": len(gpu_details), "nvml_gpu_count": nvml_count},
        )
        return CheckResult(passed=True, event=event)

    async def _track_pod(
        self,
        ctx: Context,
        pod: RentedPod,
        drop: GpuDrop | None,
        totals: ExecutorGpus,
        now_iso: str,
    ) -> PodDropOutcome | None:
        redis = ctx.services.redis
        key = _key(pod.pod_id)
        redis_ok = True
        mark: DropMark | None = None
        try:
            mark = DropMark.load(await redis.get(key))
        except REDIS_ERRORS:
            redis_ok = False
            self._log_redis_unavailable(ctx, pod.pod_id, "read")

        if drop is None:
            if mark is None:
                return None
            return await self._recover(ctx, pod, mark, totals, redis_ok)

        mark = (
            replace(mark, consecutive_cycles=mark.consecutive_cycles + 1)
            if mark
            else DropMark(now_iso)
        )
        answer: RentedGpuDropResponse | None = None
        held = drop.confirm_first and mark.consecutive_cycles < 2
        if not mark.reported and not held and not settings.DRY_RUN:
            answer = await self._post(ctx, pod, STATE_FAULT, mark, drop, totals)
            if answer is not None:
                mark = mark.after_answer(answer)
        if redis_ok:
            try:
                await redis.set(key, mark.dump(), ex=settings.RENTED_GPU_DROP_STATE_TTL_SECONDS)
            except REDIS_ERRORS:
                self._log_redis_unavailable(ctx, pod.pod_id, "write")
        return PodDropOutcome(
            pod_id=pod.pod_id,
            state=STATE_FAULT,
            first_seen_at=mark.first_seen_at,
            consecutive_cycles=mark.consecutive_cycles,
            posted=answer is not None,
            delivery=answer.delivery if answer else None,
            reported=mark.reported,
            extra={"gpu_count": pod.gpu_count, "held": held},
        )

    async def _recover(
        self, ctx: Context, pod: RentedPod, mark: DropMark, totals: ExecutorGpus, redis_ok: bool
    ) -> PodDropOutcome:
        answer: RentedGpuDropResponse | None = None
        done = not mark.recorded
        if mark.recorded and not settings.DRY_RUN:
            answer = await self._post(ctx, pod, STATE_RECOVERED, mark, None, totals)
            done = answer is not None and answer.delivery != GPU_DROP_DELIVERY_NOTIFY_FAILED
        if done and redis_ok:
            try:
                await ctx.services.redis.delete(_key(pod.pod_id))
            except REDIS_ERRORS:
                self._log_redis_unavailable(ctx, pod.pod_id, "delete")
        return PodDropOutcome(
            pod_id=pod.pod_id,
            state=STATE_RECOVERED,
            first_seen_at=mark.first_seen_at,
            consecutive_cycles=mark.consecutive_cycles,
            posted=answer is not None,
            delivery=answer.delivery if answer else None,
            reported=done,
            extra={"gpu_count": pod.gpu_count},
        )

    async def _post(
        self,
        ctx: Context,
        pod: RentedPod,
        state: str,
        mark: DropMark,
        drop: GpuDrop | None,
        totals: ExecutorGpus,
    ) -> RentedGpuDropResponse | None:
        pod_id = pod.pod_id
        expected = _clamp(
            ctx, pod_id, "expected_gpu_count", drop.expected if drop else 0, MAX_REPORTED_GPU_COUNT
        )
        visible = _clamp(
            ctx, pod_id, "visible_gpu_count", drop.visible if drop else 0, MAX_REPORTED_GPU_COUNT
        )
        cycles = _clamp(
            ctx, pod_id, "consecutive_cycles", mark.consecutive_cycles, MAX_REPORTED_CYCLES
        )
        pod_gpus = (
            _clamp(ctx, pod_id, "pod_gpu_count", pod.gpu_count, MAX_REPORTED_GPU_COUNT)
            if pod.gpu_count is not None
            else None
        )
        rented = (
            _clamp(ctx, pod_id, "rented_gpu_count", totals.rented_gpu_count, MAX_REPORTED_GPU_COUNT)
            if totals.rented_gpu_count is not None
            else None
        )
        nvml = _clamp(ctx, pod_id, "nvml_gpu_count", totals.nvml_gpu_count, MAX_REPORTED_GPU_COUNT)
        # Never fatal: a backend that is down, older (404) or raising is no answer, and the next cycle asks again.
        try:
            answer = await ctx.services.backend.report_rented_gpu_drop(
                pod_id,
                state=state,
                executor_id=ctx.executor.uuid,
                first_seen_at=mark.first_seen_at,
                consecutive_cycles=cycles,
                expected_gpu_count=expected,
                visible_gpu_count=visible,
                missing_uuids=drop.missing_uuids if drop else [],
                nvml_error_code=drop.nvml_error_code if drop else None,
                faults=drop.faults if drop else [],
                pod_gpu_count=pod_gpus,
                rented_gpu_count=rented,
                nvml_gpu_count=nvml,
            )
        except Exception:
            logger.warning(
                _m(
                    "RENTED_GPU_DROP_REPORT_FAILED",
                    extra=get_extra_info({**ctx.default_extra, "pod_id": pod_id, "state": state}),
                ),
                exc_info=True,
            )
            return None
        return answer if isinstance(answer, RentedGpuDropResponse) else None

    @staticmethod
    def _log_redis_unavailable(ctx: Context, pod_id: str, on: str) -> None:
        logger.warning(
            _m(
                "RENTED_GPU_DROP_REDIS_UNAVAILABLE",
                extra=get_extra_info({**ctx.default_extra, "pod_id": pod_id, "on": on}),
            ),
            exc_info=True,
        )
