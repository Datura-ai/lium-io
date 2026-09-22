"""Where a node's verification stands right now, for support.

One in-memory record per executor this validator is verifying or has recently verified: the lane
(cycle or express), the check running and since when, the last event's reason code and the last
failure. The pipeline reports each check as it starts and finishes (`Pipeline(progress=...)`), the
express lane reports its own waits (discovered, retry scheduled, left to the cycle, published).
Read through the validator's `GET /validation-progress[/{executor_uuid}]` (token-gated,
routes/validation_progress.py) and, on every state change, one `[progress]` log line (INFO for
the express lane, DEBUG for the cycle's runs). What leaves this process names a node by executor
uuid with its steps and timings; the miner hotkey stays in the log line only, and an error is its
class and step, never the exception text. Records of finished runs are kept for RETENTION so a
support reader arriving after the verdict still sees the timeline; nothing here is persisted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

# How long a finished record stays readable.
RETENTION_SECONDS = 2 * 60 * 60
# Hard cap on records so a fleet-wide cycle cannot grow the map without bound: finished records
# go first; live ones are evicted only when the live set alone is over the cap.
MAX_RECORDS = 5000
# Finished records are also swept on a write at most this often, not only when one is created.
PRUNE_EVERY_SECONDS = 60
# A check that raised or was cancelled: what support sees as its reason code.
ABORTED_PREFIX = "ABORTED:"

# Phases of one record. `waiting` states name what the node waits for; `running` names a check.
DISCOVERED = "discovered"
RUNNING = "running"
WAITING_FOR_MINER = "waiting_for_miner"
CONNECTING = "connecting"
WAITING_TO_RETRY = "waiting_to_retry"
LEFT_TO_CYCLE = "left_to_cycle"
# The checks are done: VERIFIED with a positive score, FAILED otherwise. The cycle publishes a
# verified node at its end; the express lane publishes it at once and then marks it PUBLISHED.
VERIFIED = "verified"
PUBLISHED = "published"
FAILED = "failed"
FINISHED_PHASES = frozenset({VERIFIED, PUBLISHED, FAILED, LEFT_TO_CYCLE})
# The lane names TaskService / ExpressLane report (services.miner_service.EXPRESS_LANE is "express").
EXPRESS = "express"


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class StepRecord:
    check_id: str
    started_at: datetime
    finished_at: datetime | None = None
    passed: bool | None = None
    reason_code: str | None = None


@dataclass
class ProgressRecord:
    executor_uuid: str
    miner_hotkey: str | None = None
    lane: str | None = None
    phase: str = DISCOVERED
    phase_since: datetime = field(default_factory=_now)
    current_check_id: str | None = None
    current_check_since: datetime | None = None
    steps: list[StepRecord] = field(default_factory=list)
    last_reason_code: str | None = None
    last_error: str | None = None
    attempts: int = 0
    detail: str | None = None
    updated_at: datetime = field(default_factory=_now)
    _touched_monotonic: float = field(default_factory=time.monotonic, repr=False)

    def as_dict(self) -> dict[str, Any]:
        """The payload the route serves: no hotkey, no exception text (see the module docstring)."""
        data = asdict(self)
        data.pop("_touched_monotonic", None)
        data.pop("miner_hotkey", None)
        return data


class ValidationProgress:
    """The registry. Every method is cheap and never raises into the pipeline."""

    def __init__(self) -> None:
        self._records: dict[str, ProgressRecord] = {}
        self._lock = Lock()
        self._last_prune = time.monotonic()

    # -- writes -------------------------------------------------------------------------------
    def _record(self, executor_uuid: str) -> ProgressRecord:
        record = self._records.get(executor_uuid)
        if record is None or time.monotonic() - self._last_prune > PRUNE_EVERY_SECONDS:
            self._prune_locked()
            record = self._records.get(executor_uuid)
        if record is None:
            record = self._records[executor_uuid] = ProgressRecord(executor_uuid=executor_uuid)
        record.updated_at = _now()
        record._touched_monotonic = time.monotonic()
        return record

    def _set_phase(self, record: ProgressRecord, phase: str, detail: str | None = None) -> None:
        changed = record.phase != phase or record.detail != detail
        record.phase = phase
        record.detail = detail
        if changed:
            record.phase_since = _now()
            # One line per state change: INFO for a new node's express run (a handful a day),
            # DEBUG for the cycle's runs (three per executor per wave).
            logger.log(
                logging.INFO if record.lane == EXPRESS else logging.DEBUG,
                _m(
                    "[progress] Executor verification state changed",
                    extra=get_extra_info(
                        {
                            "executor_uuid": record.executor_uuid,
                            "miner_hotkey": record.miner_hotkey,
                            "lane": record.lane,
                            "phase": phase,
                            "detail": detail,
                            "attempts": record.attempts,
                            "last_reason_code": record.last_reason_code,
                        }
                    ),
                )
            )

    def discovered(self, executor_uuid: str, miner_hotkey: str, lane: str) -> None:
        with self._lock:
            record = self._record(executor_uuid)
            record.miner_hotkey = miner_hotkey
            record.lane = lane
            self._set_phase(record, DISCOVERED)

    def asking_miner(self, executor_uuid: str, miner_hotkey: str, lane: str, attempt: int) -> None:
        """The lane asked the miner to install the validator's key on this node."""
        with self._lock:
            record = self._record(executor_uuid)
            record.miner_hotkey = miner_hotkey
            record.lane = lane
            record.attempts = attempt
            self._set_phase(record, WAITING_FOR_MINER)

    def run_started(self, executor_uuid: str, miner_hotkey: str | None, lane: str) -> None:
        """The miner answered; the validator is opening its SSH session to the node."""
        with self._lock:
            record = self._record(executor_uuid)
            record.miner_hotkey = miner_hotkey or record.miner_hotkey
            record.lane = lane
            record.steps = []
            record.current_check_id = None
            record.current_check_since = None
            self._set_phase(record, CONNECTING)

    def step_started(self, executor_uuid: str, check_id: str) -> None:
        with self._lock:
            record = self._record(executor_uuid)
            now = _now()
            record.current_check_id = check_id
            record.current_check_since = now
            record.steps.append(StepRecord(check_id=check_id, started_at=now))
            if record.phase != RUNNING:
                self._set_phase(record, RUNNING)

    def step_finished(self, executor_uuid: str, check_id: str, reason_code: str | None, passed: bool) -> None:
        with self._lock:
            record = self._record(executor_uuid)
            self._close_step(record, check_id, reason_code, passed)

    def step_aborted(self, executor_uuid: str, check_id: str, error_class: str) -> None:
        """The check raised or was cancelled; only the exception's class name is recorded."""
        with self._lock:
            record = self._record(executor_uuid)
            self._close_step(record, check_id, f"{ABORTED_PREFIX}{error_class}", passed=False)

    @staticmethod
    def _close_step(record: ProgressRecord, check_id: str, reason_code: str | None, passed: bool) -> None:
        for step in reversed(record.steps):
            if step.check_id == check_id and step.finished_at is None:
                step.finished_at = _now()
                step.passed = passed
                step.reason_code = reason_code
                break
        record.last_reason_code = reason_code
        if not passed:
            record.last_error = f"{check_id}: {reason_code}" if reason_code else check_id
        if record.current_check_id == check_id:
            record.current_check_id = None
            record.current_check_since = None

    def retry_scheduled(self, executor_uuid: str, reason: str, in_seconds: float, attempt: int) -> None:
        with self._lock:
            record = self._record(executor_uuid)
            record.attempts = attempt
            record.last_error = reason
            self._set_phase(record, WAITING_TO_RETRY, detail=f"retry in {int(in_seconds)} s: {reason}")

    def left_to_cycle(self, executor_uuid: str, reason: str, attempt: int) -> None:
        with self._lock:
            record = self._record(executor_uuid)
            record.attempts = attempt
            record.last_error = reason
            self._set_phase(record, LEFT_TO_CYCLE, detail=reason)

    def run_finished(self, executor_uuid: str, passed: bool, reason_code: str | None = None) -> None:
        with self._lock:
            record = self._record(executor_uuid)
            record.current_check_id = None
            record.current_check_since = None
            if reason_code is not None:
                record.last_reason_code = reason_code
                if not passed:
                    record.last_error = reason_code
            self._set_phase(record, VERIFIED if passed else FAILED)

    def published(self, executor_uuid: str, passed: bool) -> None:
        with self._lock:
            record = self._record(executor_uuid)
            self._set_phase(record, PUBLISHED if passed else FAILED)

    # -- reads --------------------------------------------------------------------------------
    def get(self, executor_uuid: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(executor_uuid)
            return record.as_dict() if record else None

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            return [record.as_dict() for record in self._records.values()]

    def _prune_locked(self) -> None:
        self._last_prune = time.monotonic()
        cutoff = self._last_prune - RETENTION_SECONDS
        for executor_uuid in [
            e for e, r in self._records.items() if r.phase in FINISHED_PHASES and r._touched_monotonic < cutoff
        ]:
            del self._records[executor_uuid]
        if len(self._records) > MAX_RECORDS:
            # Finished records first, oldest first; live records only when they alone exceed the cap.
            by_age = sorted(self._records.values(), key=lambda r: (r.phase not in FINISHED_PHASES, r._touched_monotonic))
            for record in by_age[: len(self._records) - MAX_RECORDS]:
                del self._records[record.executor_uuid]


class PipelineProgress:
    """The pipeline's ProgressSink: routes each check start/finish to the registry."""

    def __init__(self, registry: ValidationProgress):
        self.registry = registry

    def step_started(self, ctx, check_id: str) -> None:
        try:
            self.registry.step_started(ctx.executor.uuid, check_id)
        except Exception:  # noqa: BLE001 — progress never fails a run
            logger.debug("progress step_started failed", exc_info=True)

    def step_finished(self, ctx, check_id: str, event, passed: bool) -> None:
        try:
            self.registry.step_finished(ctx.executor.uuid, check_id, getattr(event, "reason_code", None), passed)
        except Exception:  # noqa: BLE001
            logger.debug("progress step_finished failed", exc_info=True)

    def step_aborted(self, ctx, check_id: str, error_class: str) -> None:
        try:
            self.registry.step_aborted(ctx.executor.uuid, check_id, error_class)
        except Exception:  # noqa: BLE001
            logger.debug("progress step_aborted failed", exc_info=True)


# One registry per validator process: the pipeline writes it, the HTTP route reads it.
progress = ValidationProgress()
