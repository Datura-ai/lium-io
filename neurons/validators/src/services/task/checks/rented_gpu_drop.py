"""RENTED_GPU_DROP — a rented node shows fewer GPUs than it rents, or NVML failed listing them.

A card that falls off the bus on a rented node breaks the renter's workload, and without this check nobody
is told until the next fatal cycle is investigated. The scrape's per-device loop stops at the first NVML
error, so `gpu.count` keeps the driver's count while `gpu.details` holds only the cards read before it and
`gpu_scrape_error` carries the NVML error.

The fatal checks after the scrape (GpuModelValidCheck's DETAILS_MISMATCH, GpuFingerprintCheck) halt such a
cycle long before TenantEnforcementCheck, so this check runs right after MachineSpecScrapeCheck and never
fails the cycle. For a rented executor it compares what the scrape listed with:

- the GPUs the rental holds (sum of the pods' `gpu_count`, when the backend sent every one; at least one,
  so a scrape that lists no card at all is a fault on any rented node; such a scrape fails
  MachineSpecScrapeCheck, which runs this check on it before the cycle halts),
- the driver's own count (`gpu.count`),
- the anchored UUID set of the verified-job record (`missing_uuids` is the anchor minus the listed set),

and reads the NVML error code out of `gpu_scrape_error`. A GPU-loss code (NVML_GPU_LOSS_CODES) is a fault
on its own; any other NVML code only rides along with a count fault, and a scrape error with no NVML code
adds no `nvml_error` label.

The counts are executor-wide, and a fault is posted for every RUNNING pod on the executor; each post
carries that pod's own `gpu_count` next to the executor totals so the backend can tell which renters of
a split node are affected.

Per RUNNING pod a Redis mark `rented_gpu_drop:<pod_id>` holds the incident: `first_seen_at`,
`consecutive_cycles`, `reported` (the backend answered with a delivery that needs no retry), `evidence`
(the expected and visible counts, fault names and missing UUIDs of that acknowledged report), `recorded` (the backend accepted a
report), `unanswered` (a report got no answer, so the backend may hold it) and `recovering` (a recovery
was posted). The first faulty cycle posts `POST /internal/pods/{pod_id}/gpu-drop` with `state=fault`;
later cycles post again while `reported` is False or the evidence changed (on a split node another
renter's card can go next). A fault that rests only on detail rows cut short by a non-loss NVML error,
while the driver's count still covers the rental and the anchor, or on an anchored UUID missing from a
scrape that lists another card twice while its rows still cover every card (`confirm_first`), posts from
its second consecutive cycle instead. The first clean cycle after an incident the backend recorded or
never answered posts `state=recovered` and deletes the mark once the backend answered with a delivery
that needs no retry; a fault after a posted recovery starts a new incident. A mark nothing was posted
for is deleted without a post. A backend that is down or older (404) is no answer: the next cycle asks
again. A dry run judges and logs only: it reads and writes no mark. Every mark is also kept in this process, and a cycle that cannot read Redis uses that copy, so an
incident that spans a Redis outage is still posted once and recovered; the check's verdict never depends
on Redis.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

import redis.exceptions
from protocol.vc_protocol.compute_requests import (
    GPU_DROP_DELIVERY_DISABLED,
    GPU_DROP_DELIVERY_NOT_RENTED,
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

# The one pod status that holds a renter's cards; a backend that predates the field sends none.
POD_STATUS_RUNNING = "RUNNING"
# What a failing Redis raises through RedisService: the client's errors and the socket errors under them.
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
# The backend refuses (422) a GPU count above this; this check runs before GpuCountCheck, so the
# scrape's counts are unbounded here.
MAX_REPORTED_GPU_COUNT = 64

STATE_FAULT = "fault"
STATE_RECOVERED = "recovered"

# A delivery that needs no retry of the report (any other, unknown ones included, is retried next cycle), and
# one that means the backend did not write the incident.
_NO_RETRY_DELIVERIES = frozenset({None, "notified", "recorded", GPU_DROP_DELIVERY_NOT_RENTED})
_NOT_RECORDED_DELIVERIES = frozenset({GPU_DROP_DELIVERY_DISABLED, GPU_DROP_DELIVERY_NOT_RENTED})

# key -> (mark, monotonic expiry): the marks as this process last wrote them, read when Redis cannot be.
_LOCAL_MARKS: dict[str, tuple[DropMark, float]] = {}


@dataclass(frozen=True)
class GpuDrop:
    """What the scrape shows against what the rental should see; only built when there is a fault."""

    expected: int
    visible: int
    nvml_count: int
    missing_uuids: list[str]
    nvml_error_code: int | None
    faults: list[str]
    # the fault may be a one-off glitch, so the report waits for a second faulty cycle: the detail rows
    # are short only because a non-loss NVML error (a timeout, say) cut the scrape's loop, or an anchored
    # UUID is missing only from a scrape that lists another card twice, while the rows (or the driver)
    # still cover every rented and anchored card
    confirm_first: bool

    @property
    def evidence(self) -> list[Any]:
        """What decides which renters of a split node are affected; a change is reported again."""
        return [self.expected, self.visible, self.faults, self.missing_uuids]


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
    Only `anchored_gpu_missing` reads the distinct set, and it waits for a second cycle when the listed
    UUIDs repeat one while the rows still cover every rented, anchored and driver-counted card.
    """
    listed = set(listed_uuids)
    visible = listed_count
    anchor = set(anchor_uuids)
    # a rented node holds at least one GPU, even when a pod predates `gpu_count`
    rented = rented_gpu_count or 1
    code = nvml_error_code(scrape_error)
    missing = sorted(anchor - listed)[:MAX_MISSING_UUIDS] if listed or not listed_count else []

    faults: list[str] = []
    if visible < rented:
        faults.append(FAULT_BELOW_RENTED)
    if visible < nvml_count:
        faults.append(FAULT_DETAILS_SHORT)
    if missing:
        faults.append(FAULT_ANCHORED_MISSING)
    if code in NVML_GPU_LOSS_CODES or (code is not None and faults):
        faults.append(FAULT_NVML_ERROR)
    if not faults:
        return None
    expected = max(rented, len(anchor), nvml_count)
    cut_short = (
        bool(scrape_error)
        and FAULT_DETAILS_SHORT in faults
        and nvml_count >= max(rented, len(anchor))
    )
    duplicate_listed = (
        FAULT_ANCHORED_MISSING in faults
        and len(listed) < len(listed_uuids)
        and listed_count >= expected
    )
    return GpuDrop(
        expected=expected,
        visible=visible,
        nvml_count=nvml_count,
        missing_uuids=[uuid[:MAX_UUID_CHARS] for uuid in missing],
        nvml_error_code=code,
        faults=faults,
        confirm_first=code not in NVML_GPU_LOSS_CODES and (cut_short or duplicate_listed),
    )


@dataclass(frozen=True)
class DropMark:
    """One pod's open incident, as kept in Redis between cycles."""

    first_seen_at: str
    consecutive_cycles: int = 1
    reported: bool = False
    recorded: bool = False
    unanswered: bool = False
    recovering: bool = False
    evidence: list[Any] | None = None

    @classmethod
    def load(cls, raw: object) -> DropMark | None:
        if raw is None:
            return None
        try:
            return cls(**json.loads(raw))
        except (TypeError, ValueError):
            return None

    def dump(self) -> str:
        return json.dumps(asdict(self))

    @property
    def needs_recovery(self) -> bool:
        return self.recorded or self.unanswered

    def after_answer(self, answer: RentedGpuDropResponse | None, evidence: list[Any]) -> DropMark:
        if answer is None:
            return replace(self, unanswered=True)
        reported = answer.delivery in _NO_RETRY_DELIVERIES
        return replace(
            self,
            reported=reported,
            recorded=self.recorded or answer.delivery not in _NOT_RECORDED_DELIVERIES,
            evidence=evidence if reported else self.evidence,
        )


def _key(pod_id: str) -> str:
    return f"{RENTED_GPU_DROP_KEY_PREFIX}:{pod_id}"


def _local_mark(key: str) -> DropMark | None:
    entry = _LOCAL_MARKS.get(key)
    if entry is None or entry[1] <= time.monotonic():
        _LOCAL_MARKS.pop(key, None)
        return None
    return entry[0]


def _gpus(count: int | None) -> int | None:
    return None if count is None else min(max(count, 0), MAX_REPORTED_GPU_COUNT)


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
        rented_total = _rented_gpu_count(all_pods)
        drop = judge_rented_gpus(
            rented_gpu_count=rented_total,
            anchor_uuids=split_uuids((ctx.verified or {}).get(GPU_ANCHOR_KEY) or ""),
            nvml_count=nvml_count,
            listed_uuids=_listed_uuids(gpu_details),
            listed_count=len(gpu_details),
            scrape_error=scrape_error,
        )

        now_iso = datetime.now(UTC).isoformat()
        # A dry run shares Redis with the live validator, so it leaves the live incidents alone.
        outcomes: list[dict[str, Any]] = []
        if not settings.DRY_RUN:
            for pod in pods:
                outcome = await self._track_pod(ctx, pod, drop, rented_total, nvml_count, now_iso)
                if outcome is not None:
                    outcomes.append(outcome)

        if drop is not None:
            what = {
                "executor_uuid": ctx.executor.uuid,
                "expected_gpu_count": drop.expected,
                "visible_gpu_count": drop.visible,
                "rented_gpu_count": rented_total,
                "nvml_gpu_count": drop.nvml_count,
                "missing_uuids": drop.missing_uuids,
                "nvml_error_code": drop.nvml_error_code,
                "gpu_scrape_error": scrape_error[:MAX_SCRAPE_ERROR_CHARS]
                if isinstance(scrape_error, str)
                else None,
                "faults": drop.faults,
                "pods": outcomes,
            }
            logger.warning(
                _m("RENTED_GPU_DROP", extra=get_extra_info({**ctx.default_extra, **what}))
            )
            return CheckResult(
                passed=True,
                event=render_message(Msg.DROP, ctx=ctx, check_id=self.check_id, what=what),
            )

        if outcomes:
            what = {"executor_uuid": ctx.executor.uuid, "pods": outcomes}
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
        rented_total: int | None,
        nvml_count: int,
        now_iso: str,
    ) -> dict[str, Any] | None:
        key = _key(pod.pod_id)
        redis_ok = True
        try:
            mark = DropMark.load(await ctx.services.redis.get(key))
        except REDIS_ERRORS:
            redis_ok = False
            self._log_redis_unavailable(ctx, pod.pod_id, "read")
            mark = _local_mark(key)

        if drop is None:
            if mark is None:
                return None
            return await self._recover(ctx, pod, mark, rented_total, nvml_count, redis_ok)

        if mark is None or mark.recovering:
            mark = DropMark(now_iso)
        else:
            mark = replace(mark, consecutive_cycles=mark.consecutive_cycles + 1)
        answer: RentedGpuDropResponse | None = None
        posted = False
        held = drop.confirm_first and mark.consecutive_cycles < 2
        due = not mark.reported or mark.evidence != drop.evidence
        if due and not held:
            answer = await self._post(ctx, pod, STATE_FAULT, mark, drop, rented_total, nvml_count)
            posted = True
            mark = mark.after_answer(answer, drop.evidence)
        await self._save(ctx, pod.pod_id, mark, redis_ok)
        return {
            "pod_id": pod.pod_id,
            "state": STATE_FAULT,
            "first_seen_at": mark.first_seen_at,
            "consecutive_cycles": mark.consecutive_cycles,
            "posted": posted,
            "delivery": answer.delivery if answer else None,
            "reported": mark.reported,
            "gpu_count": pod.gpu_count,
            "held": held,
        }

    async def _recover(
        self,
        ctx: Context,
        pod: RentedPod,
        mark: DropMark,
        rented_total: int | None,
        nvml_count: int,
        redis_ok: bool,
    ) -> dict[str, Any] | None:
        if not mark.needs_recovery:
            await self._forget(ctx, pod.pod_id)
            return None
        answer = await self._post(ctx, pod, STATE_RECOVERED, mark, None, rented_total, nvml_count)
        done = answer is not None and answer.delivery in _NO_RETRY_DELIVERIES
        if done:
            await self._forget(ctx, pod.pod_id)
        elif not mark.recovering:
            await self._save(ctx, pod.pod_id, replace(mark, recovering=True), redis_ok)
        return {
            "pod_id": pod.pod_id,
            "state": STATE_RECOVERED,
            "first_seen_at": mark.first_seen_at,
            "consecutive_cycles": mark.consecutive_cycles,
            "posted": True,
            "delivery": answer.delivery if answer else None,
            "reported": done,
            "gpu_count": pod.gpu_count,
        }

    async def _save(self, ctx: Context, pod_id: str, mark: DropMark, redis_ok: bool) -> None:
        key = _key(pod_id)
        ttl = settings.RENTED_GPU_DROP_STATE_TTL_SECONDS
        now = time.monotonic()
        for stale in [k for k, (_, expires) in _LOCAL_MARKS.items() if expires <= now]:
            del _LOCAL_MARKS[stale]
        _LOCAL_MARKS[key] = (mark, now + ttl)
        if redis_ok:
            try:
                await ctx.services.redis.set(key, mark.dump(), ex=ttl)
            except REDIS_ERRORS:
                self._log_redis_unavailable(ctx, pod_id, "write")

    async def _forget(self, ctx: Context, pod_id: str) -> None:
        # Deleted even when this cycle's read failed: a mark left in Redis would pass the next incident off as this one.
        key = _key(pod_id)
        _LOCAL_MARKS.pop(key, None)
        try:
            await ctx.services.redis.delete(key)
        except REDIS_ERRORS:
            self._log_redis_unavailable(ctx, pod_id, "delete")

    async def _post(
        self,
        ctx: Context,
        pod: RentedPod,
        state: str,
        mark: DropMark,
        drop: GpuDrop | None,
        rented_total: int | None,
        nvml_count: int,
    ) -> RentedGpuDropResponse | None:
        # Never fatal: a backend that is down, older (404) or raising is no answer, and the next cycle asks again.
        try:
            answer = await ctx.services.backend.report_rented_gpu_drop(
                pod.pod_id,
                state=state,
                executor_id=ctx.executor.uuid,
                first_seen_at=mark.first_seen_at,
                consecutive_cycles=mark.consecutive_cycles,
                expected_gpu_count=_gpus(drop.expected if drop else 0),
                visible_gpu_count=_gpus(drop.visible if drop else 0),
                missing_uuids=drop.missing_uuids if drop else [],
                nvml_error_code=drop.nvml_error_code if drop else None,
                faults=drop.faults if drop else [],
                pod_gpu_count=_gpus(pod.gpu_count),
                rented_gpu_count=_gpus(rented_total),
                nvml_gpu_count=_gpus(nvml_count),
            )
        except Exception:
            logger.warning(
                _m(
                    "RENTED_GPU_DROP_REPORT_FAILED",
                    extra=get_extra_info(
                        {**ctx.default_extra, "pod_id": pod.pod_id, "state": state}
                    ),
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
