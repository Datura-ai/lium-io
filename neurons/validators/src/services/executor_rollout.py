"""DAH-3405: no verdict for an executor that fails during a known executor-image rollout.

11 Sep 2026, 07:50–08:00Z: executor-v1.126 was pushed to Docker Hub in the middle of a validation
cycle. Watchtower on every standard node pulled it and recreated the executor container while the
validator's SSH sessions were inside it. Rustam's cycle log for that cycle: 168 transport-unreachable,
51 scrape-failed and 50 insufficient-ports rows, plus 70 EXECUTOR_IMAGE_OUTDATED rows against a
normal background of ~31 (rows, not distinct executors); tsdb prod_executors: 506 of 514 executors
at 0 against 161 of 517 the cycle before. None of those were verdicts about the machines.

The validator learns about a release from the same place watchtower does: the registry digest of
EXECUTOR_IMAGE_REF. `ExecutorRolloutTracker` remembers the moment that digest changed and the cycle
it was seen in (in Redis, so a restart inside the window keeps it). For that cycle and the next
(`EXECUTOR_ROLLOUT_GRACE_CYCLES`, counted by the cycle's job block, never by the clock)
`withhold_rollout_verdicts` takes out of the cycle every result whose failure is one a rollout
produces and whose executor is not yet on the new image. A withheld result is neither scored nor
published: the backend keeps the previous cycle's row, the weights are computed without it. A
failure on an executor that already runs the new image, any failure outside the set, and every
result after the window stand exactly as before.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from core.config import settings
from core.utils import _m
from services.executor_image_policy import ImageVerdict, normalize_sha256_digest
from services.redis_service import RedisService
from services.task.availability import AvailabilityErrorCode
from services.task.checks.executor_image import observed_executor_digest
from services.task.messages import (
    ExecutorImageMessages,
    FinalizeMessages,
    MachineSpecMessages,
    PortCountMessages,
    RentalVerificationMessages,
    TenantEnforcementMessages,
    UploadFilesMessages,
)
from services.task.models import JobResult

logger = logging.getLogger(__name__)

ROLLOUT_STATE_KEY = "executor_image_rollout"
ROLLOUT_GRACE = "ROLLOUT_GRACE"
# A withheld executor publishes nothing, and the backend marks an executor inactive (with the
# EXECUTOR_INACTIVE_MID_RENTAL penalty) when its row was not updated for 1 h (lium-io-backend
# apps/server/src/worker.py, check_and_update_executors). With two withheld cycles the gap between
# an executor's publishes is three cycle periods plus the duration of the next scored cycle minus
# the duration of the last one: about 2700 + 850 s (JOB_TIME_OUT − 50) = 3550 s while cycles keep
# their 15-minute cadence (job blocks 75 × 12 s apart) — a lower bound, since a cycle that runs
# past 900 s delays the next start. A third withheld cycle puts the gap past the hour in every
# case, and the hourly sweep catches it whenever it runs inside the excess. A new window may open
# only once one uncovered cycle has published (`cycles_seen >= grace_cycles + 2`), so two pushes
# back to back cannot chain windows either.
MAX_ROLLOUT_GRACE_CYCLES = 2

# What a watchtower container recreate makes a running validation look like, in the order the
# pipeline meets them: the connect never opens; the shell dies under the upload, under a check
# that lets asyncssh raise, or under the filler liveness read; the scrape that ran inside the old
# container returns nothing; the connectivity probe cannot verify the ports of a container being
# replaced; and an executor that has not pulled yet (or a validator whose snapshot predates the
# push) reads OUTDATED.
ROLLOUT_FAILURE_REASONS = frozenset(
    {
        AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE.value,
        UploadFilesMessages.UPLOAD_FAILED.reason,
        TenantEnforcementMessages.EXECUTOR_TRANSPORT_UNREACHABLE.reason,
        RentalVerificationMessages.FILLER_TRANSPORT_UNREACHABLE.reason,
        MachineSpecMessages.SCRAPE_FAILED.reason,
        PortCountMessages.INSUFFICIENT_PORTS.reason,
        ExecutorImageMessages.OUTDATED.reason,
    }
)
_OUTDATED = ExecutorImageMessages.OUTDATED.reason
# The two ways a run ends with score 0 and the OUTDATED report attached instead of failing at the
# image check: a rented executor's image check passes and the tenant-enforcement halt ends the run
# (RENTED); a run that reaches finalize ends on VALIDATION_COMPLETED.
_RUN_ENDED_WITHOUT_FAILING = frozenset(
    {FinalizeMessages.COMPLETED.reason, TenantEnforcementMessages.ALREADY_RENTED.reason}
)


@dataclass(frozen=True)
class RolloutState:
    """What Redis remembers between cycles, under `ROLLOUT_STATE_KEY`, as one JSON object.

    `opened_job_block` stays None until a digest change opens a window. `cycles_seen` counts the
    cycles observed since that change, the opening one included, and `last_job_block` is the cycle
    the last count was made for. `closed` records that the window-end line was logged, so it is
    logged once.
    """

    digest: str | None = None
    previous_digest: str | None = None
    started_at: datetime | None = None
    opened_job_block: int | None = None
    last_job_block: int | None = None
    cycles_seen: int = 0
    withheld: int = 0
    closed: bool = False

    @classmethod
    def from_json(cls, raw: str | bytes) -> RolloutState:
        """Raises ValueError or TypeError on anything this class did not write."""
        fields = json.loads(raw)
        if not isinstance(fields, dict):
            raise ValueError("rollout state is not a JSON object")
        started_at = fields.get("started_at")
        opened_job_block = fields.get("opened_job_block")
        last_job_block = fields.get("last_job_block")
        return cls(
            digest=fields.get("digest"),
            previous_digest=fields.get("previous_digest"),
            started_at=datetime.fromisoformat(started_at) if started_at else None,
            opened_job_block=int(opened_job_block) if opened_job_block is not None else None,
            last_job_block=int(last_job_block) if last_job_block is not None else None,
            cycles_seen=int(fields.get("cycles_seen", 0)),
            withheld=int(fields.get("withheld", 0)),
            closed=bool(fields.get("closed", False)),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "digest": self.digest,
                "previous_digest": self.previous_digest,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "opened_job_block": self.opened_job_block,
                "last_job_block": self.last_job_block,
                "cycles_seen": self.cycles_seen,
                "withheld": self.withheld,
                "closed": self.closed,
            }
        )


@dataclass(frozen=True)
class WithheldVerdict:
    """One result the rollout took out of the cycle, with the miner that reported it."""

    miner_hotkey: str
    result: JobResult


@dataclass(frozen=True)
class RolloutWindow:
    """What the validator knows about the current executor image and its last change.

    `opened_job_block` is the job block of the cycle in which the change was seen; `cycles_seen` is
    how many distinct cycles (that one included) the tracker has been told about since. The window
    covers a cycle while `cycles_seen <= grace_cycles`. Cycles are counted as the validator runs
    them, so a long cycle, a slow clock or a cycle that lands two job blocks later cannot stretch or
    shorten the window.
    """

    digest: str | None
    previous_digest: str | None
    started_at: datetime | None
    opened_job_block: int | None
    cycles_seen: int
    grace_cycles: int

    def covers(self, job_block: int) -> bool:
        if self.grace_cycles <= 0 or self.opened_job_block is None:
            return False
        return job_block >= self.opened_job_block and 0 < self.cycles_seen <= self.grace_cycles

    def as_extra(self) -> dict[str, object]:
        return {
            "rollout_digest": self.digest,
            "rollout_previous_digest": self.previous_digest,
            "rollout_started_at": self.started_at.isoformat() if self.started_at else None,
            "rollout_opened_job_block": self.opened_job_block,
            "rollout_cycles_seen": self.cycles_seen,
            "rollout_grace_cycles": self.grace_cycles,
        }


class ExecutorRolloutTracker:
    """Remembers the authorized executor digest, and when and in which cycle it last changed.

    State lives in Redis as a `RolloutState`. A Redis error propagates; the caller decides what a
    cycle without this knowledge does. An unreadable value is treated as no state and re-seeded.
    With `grace_cycles` 0 the digest is still tracked but no window ever covers a cycle and neither
    window line is logged.
    """

    def __init__(self, redis_service: RedisService, grace_cycles: int | None = None):
        self.redis_service = redis_service
        configured_grace_cycles = (
            settings.EXECUTOR_ROLLOUT_GRACE_CYCLES if grace_cycles is None else grace_cycles
        )
        self.grace_cycles = min(configured_grace_cycles, MAX_ROLLOUT_GRACE_CYCLES)
        if configured_grace_cycles > MAX_ROLLOUT_GRACE_CYCLES:
            logger.warning(
                _m(
                    "[rollout-grace] EXECUTOR_ROLLOUT_GRACE_CYCLES capped: more cycles without a "
                    "published row would trip the backend's 1-hour inactive sweep",
                    extra={"wanted": configured_grace_cycles, "used": self.grace_cycles},
                )
            )

    async def observe(
        self, digest: str | None, job_block: int, now: datetime | None = None
    ) -> RolloutWindow:
        """Compare the digest fetched now with the one remembered; open a window on a change.

        `job_block` is the current cycle's job block (the one its `job_batch_id` was derived from);
        a job block not seen before counts as one more cycle of an open window. `None` for the
        digest (the registry could not be read) changes nothing else: the window the validator
        already knows about stays as it is. The first digest ever seen opens no window —
        there is nothing to compare it with, and a window on every fresh Redis would grace every
        failure after a redeploy.
        """
        now = now or datetime.now(UTC)
        # The same normalisation the image check applies to what the node reports, so the two
        # sides of the comparison in `rollout_grace_reason` are spelled the same way.
        digest = normalize_sha256_digest(digest)
        state = await self._count_this_cycle(await self._load(), job_block)
        if digest is None or digest == state.digest:
            return await self._close_the_window_when_it_is_over(state, job_block)
        return await self._react_to_the_changed_digest(state, digest, job_block, now)

    async def record_withheld(self, count: int) -> None:
        """Add this cycle's withheld results to the window's running total."""
        if count <= 0:
            return
        state = await self._load()
        await self._save(replace(state, withheld=state.withheld + count))

    async def _count_this_cycle(self, state: RolloutState, job_block: int) -> RolloutState:
        """Count one more cycle of an open window, the first time this job block is observed.

        The cycle-start and the cycle-end observation of one cycle share a job block, so the
        second of the two counts nothing.
        """
        if state.opened_job_block is None or job_block == state.last_job_block:
            return state
        counted = replace(state, last_job_block=job_block, cycles_seen=state.cycles_seen + 1)
        await self._save(counted)
        return counted

    async def _react_to_the_changed_digest(
        self, state: RolloutState, digest: str, job_block: int, now: datetime
    ) -> RolloutWindow:
        """Seed the first digest ever seen, open a window on a change, or refuse a second push."""
        if state.digest is None:
            seeded = RolloutState(digest=digest)
            await self._save(seeded)
            return self._window(seeded)

        if self._a_new_window_may_open(state):
            opened = RolloutState(
                digest=digest,
                previous_digest=state.digest,
                started_at=now,
                opened_job_block=job_block,
                last_job_block=job_block,
                cycles_seen=1,
            )
            await self._save(opened)
            return self._log_window(
                "[rollout-grace] executor image rollout detected; verdicts of executors "
                "that fail while restarting are withheld while the window is open",
                self._window(opened),
                job_block,
            )

        # A second push (a hotfix, a rollback) while the window is open or before one uncovered
        # cycle has published: the new digest is what "current" means from here on, but the
        # window is NOT reopened or extended — withheld cycles would chain, and three in a row
        # are what trips the backend's 1-hour inactive sweep.
        chained = replace(state, digest=digest, previous_digest=state.digest)
        await self._save(chained)
        return self._log_window(
            "[rollout-grace] executor image changed again before one cycle after the "
            "rollout window has published; the window is not reopened",
            self._window(chained),
            job_block,
        )

    def _a_new_window_may_open(self, state: RolloutState) -> bool:
        """True when no window was ever opened, or one uncovered cycle has published since."""
        return state.opened_job_block is None or state.cycles_seen >= self.grace_cycles + 2

    async def _close_the_window_when_it_is_over(
        self, state: RolloutState, job_block: int
    ) -> RolloutWindow:
        """Log the window-end line on the first cycle the window no longer covers, once."""
        window = self._window(state)
        if (
            self.grace_cycles <= 0
            or state.opened_job_block is None
            or state.closed
            or window.covers(job_block)
        ):
            return window
        await self._save(replace(state, closed=True))
        logger.warning(
            _m(
                "[rollout-grace] executor image rollout window ended",
                extra={
                    "outcome": ROLLOUT_GRACE,
                    "job_block": job_block,
                    "withheld_total": state.withheld,
                    **window.as_extra(),
                },
            )
        )
        return window

    def _log_window(self, message: str, window: RolloutWindow, job_block: int) -> RolloutWindow:
        """Log one window line and give the window back, so a caller can return in one statement."""
        if self.grace_cycles > 0:
            logger.warning(
                _m(
                    message,
                    extra={"outcome": ROLLOUT_GRACE, "job_block": job_block, **window.as_extra()},
                )
            )
        return window

    def _window(self, state: RolloutState) -> RolloutWindow:
        return RolloutWindow(
            digest=state.digest,
            previous_digest=state.previous_digest,
            started_at=state.started_at,
            opened_job_block=state.opened_job_block,
            cycles_seen=state.cycles_seen,
            grace_cycles=self.grace_cycles,
        )

    async def _load(self) -> RolloutState:
        raw = await self.redis_service.get(ROLLOUT_STATE_KEY)
        if not raw:
            return RolloutState()
        try:
            return RolloutState.from_json(raw)
        except (ValueError, TypeError):
            logger.error(
                _m(
                    "[rollout-grace] rollout state is unreadable; starting over",
                    extra={"raw": str(raw)[:200]},
                )
            )
            return RolloutState()

    async def _save(self, state: RolloutState) -> None:
        await self.redis_service.set(ROLLOUT_STATE_KEY, state.to_json())


def _observed_digest(result: JobResult) -> str | None:
    report = result.executor_image_report or {}
    observed = report.get("observed_digest")
    if observed:
        return str(observed)
    return observed_executor_digest(result.spec or {})


def rollout_grace_reason(result: JobResult, window: RolloutWindow, job_block: int) -> str | None:
    """The reason code this result's verdict is withheld for, or None when the verdict stands.

    Withheld: the window covers this cycle, the run ended on one of `ROLLOUT_FAILURE_REASONS` (or
    it ended without failing — the rented halt or finalize — with score 0 and the image check had
    read OUTDATED: a rented executor's image check passes, so the run halts as RENTED instead of
    failing), and, for every reason but OUTDATED, the executor's observed image is not the new
    digest. An executor that already runs
    the new image failed for a reason of its own; a rented executor that failed a later check of
    its own (pod not running, filler killed) failed for that reason, OUTDATED or not; and a result
    with a score is never touched.
    """
    if not window.covers(job_block) or result.score > 0:
        return None
    reason = result.failure_reason_code
    if reason not in ROLLOUT_FAILURE_REASONS:
        ended_without_failing = reason in _RUN_ENDED_WITHOUT_FAILING
        outdated = (result.executor_image_report or {}).get("status") == ImageVerdict.OUTDATED.value
        if not (ended_without_failing and outdated):
            return None
        reason = _OUTDATED
    # OUTDATED is withheld whatever the observed digest says: old = not pulled yet, new = the
    # validator's own snapshot predated the push. Every other reason on an executor that already
    # runs the new image is that executor's own failure.
    if reason != _OUTDATED and _observed_digest(result) == window.digest:
        return None
    return reason


def withhold_rollout_verdicts(
    job_results: dict[str, list[JobResult]],
    window: RolloutWindow,
    job_block: int,
) -> tuple[dict[str, list[JobResult]], list[WithheldVerdict]]:
    """Split a cycle's results into the ones that stand and the ones the rollout withholds.

    Returns the standing results per miner and the withheld ones. Every withheld result is logged
    once with its reason, so Loki can count them (`outcome="ROLLOUT_GRACE"`). Outside the window
    the input is returned as it is and nothing is logged.
    """
    if not window.covers(job_block):
        return job_results, []

    standing_by_miner: dict[str, list[JobResult]] = {}
    withheld: list[WithheldVerdict] = []
    for miner_hotkey, results in job_results.items():
        standing_results: list[JobResult] = []
        for result in results:
            reason = rollout_grace_reason(result, window, job_block)
            if reason is None:
                standing_results.append(result)
                continue
            withheld.append(WithheldVerdict(miner_hotkey=miner_hotkey, result=result))
            logger.warning(
                _m(
                    "[rollout-grace] verdict withheld: the executor failed during a known "
                    "executor-image rollout; no score is written this cycle",
                    extra={
                        "outcome": ROLLOUT_GRACE,
                        "miner_hotkey": miner_hotkey,
                        "executor_uuid": result.executor_info.uuid,
                        "job_batch_id": result.job_batch_id,
                        "job_block": job_block,
                        "reason_code": reason,
                        "observed_digest": _observed_digest(result),
                        **window.as_extra(),
                    },
                )
            )
        standing_by_miner[miner_hotkey] = standing_results
    return standing_by_miner, withheld
