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
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from core.config import settings
from core.utils import _m
from incentive.config import BASE_GPU_MAP
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse
from services.executor_image_policy import ImageVerdict, normalize_sha256_digest
from services.redis_service import (
    LAST_SCORED_RESULT_TTL_SECONDS,
    RedisService,
    RedisWrites,
    last_scored_result_key,
    last_scored_result_vouch_key,
)
from services.task.availability import AvailabilityErrorCode, _is_the_nodes_own_fault
from services.task.checks.rental_owner import listed_under_another_miner, rental_owned_by
from services.task.checks.executor_image import observed_executor_digest
from services.task.messages import (
    CYCLE_END_READ_REASONS,
    ExecutorImageMessages,
    FinalizeMessages,
    GpuPowerLimitMessages,
    GpuUsageMessages,
    LoginUserMessages,
    MachineSpecMessages,
    NO_TASK_TIME_REASONS,
    PortConnectivityMessages,
    PortCountMessages,
    RentalVerificationMessages,
    SCRAPE_HOST_SIDE_FAILURE_REASONS,
    SCRAPE_UNDETERMINED_FAILURE_REASONS,
    TenantEnforcementMessages,
    UploadFilesMessages,
)
from services.task.models import VALIDATOR_REDIS_UNAVAILABLE, JobResult
from services.task.result_handler import rental_gpu_count

logger = logging.getLogger(__name__)

ROLLOUT_STATE_KEY = "executor_image_rollout"
ROLLOUT_GRACE = "ROLLOUT_GRACE"
# A withheld executor publishes nothing, and the backend marks an executor inactive (with the
# EXECUTOR_INACTIVE_MID_RENTAL penalty) when its row was not updated for 1 h (lium-io-backend
# apps/server/src/worker.py, check_and_update_executors). With two withheld cycles the gap between
# an executor's publishes is three cycle periods plus the duration of the next scored cycle minus
# the duration of the last one: about 2700 + 850 s (JOB_TIME_OUT − 50) = 3550 s while cycles keep
# their 15-minute cadence (job blocks 75 × 12 s apart) — a lower bound, since a cycle that runs
# past 900 s delays the next start. A third withheld cycle puts the gap past the hour whenever
# the next scored cycle is not shorter than the last (3600 s + d_next − d_last), and the hourly
# sweep catches it whenever it runs inside the excess, so the cap stays at 2. A new window may open
# only once one uncovered cycle has published (`cycles_seen >= grace_cycles + 2`), so two pushes
# back to back cannot chain windows either.
MAX_ROLLOUT_GRACE_CYCLES = 2
# DAH-6560: how long the release just replaced still reads CURRENT, counted like the window (the cycle
# the change is seen is cycle 1) and restarted by every change, a second push inside the window too.
# Separate from the window: it changes only the image verdict, never withholds a result, so it has no
# 1-hour cap. Many more nodes are still on the previous image after 2 cycles than after 8.
PREVIOUS_RELEASE_CURRENT_CYCLES = 8

# What a watchtower container recreate makes a running validation look like, in the order the
# pipeline meets them: the connect never opens; the shell dies under the upload, under a check
# that lets asyncssh raise, or under the filler liveness read; the scrape that ran inside the old
# container returns nothing; the connectivity probe cannot verify the ports of a container being
# replaced; and an executor that has not pulled yet (or a validator whose snapshot predates the
# push) reads OUTDATED.
ROLLOUT_FAILURE_REASONS = frozenset(
    {
        AvailabilityErrorCode.EXECUTOR_SSH_UNREACHABLE.value,
        # DAH-3558: the miner left the restarting executor out of its answer; the validator then
        # wrote this row for the rented node.
        AvailabilityErrorCode.RENTED_EXECUTOR_NOT_LISTED.value,
        # its idle twin: a restarting unrented executor the miner cannot reach during the rollout
        AvailabilityErrorCode.EXECUTOR_UNREACHABLE_BY_MINER.value,
        UploadFilesMessages.UPLOAD_FAILED.reason,
        TenantEnforcementMessages.EXECUTOR_TRANSPORT_UNREACHABLE.reason,
        RentalVerificationMessages.FILLER_TRANSPORT_UNREACHABLE.reason,
        MachineSpecMessages.SCRAPE_FAILED.reason,
        # the codes SCRAPE_FAILED split into keep the grace it had
        *SCRAPE_HOST_SIDE_FAILURE_REASONS,
        *SCRAPE_UNDETERMINED_FAILURE_REASONS,
        PortCountMessages.INSUFFICIENT_PORTS.reason,
        # DAH-5981: PortCountCheck's verdict when the SSH connection dropped under the port probe
        PortConnectivityMessages.VERIFY_FAILED.reason,
        ExecutorImageMessages.OUTDATED.reason,
    }
)
_OUTDATED = ExecutorImageMessages.OUTDATED.reason
# The ways a run ends with score 0 and the OUTDATED report attached instead of failing at the
# image check: a rented executor's image check passes and the tenant-enforcement halt ends the run
# (RENTED); a run that reaches finalize ends on VALIDATION_COMPLETED.
RUN_ENDED_WITHOUT_FAILING = frozenset(
    {
        FinalizeMessages.COMPLETED.reason,
        TenantEnforcementMessages.ALREADY_RENTED.reason,
    }
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
    # cycles since the last digest change, that cycle included; 0 until a change is seen
    previous_current_cycles_seen: int = 0
    # releases replaced before `previous_digest` whose own 8 CURRENT cycles have not run out, as
    # (digest, cycles since it was replaced): a second push must not cut a node's grace short
    earlier_current: tuple[tuple[str, int], ...] = ()

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
            previous_current_cycles_seen=int(fields.get("previous_current_cycles_seen", 0)),
            earlier_current=tuple((str(d), int(n)) for d, n in fields.get("earlier_current") or ()),
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
                "previous_current_cycles_seen": self.previous_current_cycles_seen,
                "earlier_current": [list(entry) for entry in self.earlier_current],
            }
        )

    def earlier_after_replacing(self, new_digest: str) -> tuple[tuple[str, int], ...]:
        """What `earlier_current` becomes when `digest` is replaced by `new_digest`: the outgoing
        previous release joins it while its own grace runs, and a digest that is current again leaves it."""
        entries = list(self.earlier_current)
        if self.previous_digest and 0 < self.previous_current_cycles_seen <= PREVIOUS_RELEASE_CURRENT_CYCLES:
            entries.append((self.previous_digest, self.previous_current_cycles_seen))
        kept: dict[str, int] = {}
        for digest, cycles in entries:
            if digest not in (new_digest, self.digest):
                kept[digest] = min(cycles, kept.get(digest, cycles))
        return tuple(kept.items())


@dataclass(frozen=True)
class WithheldVerdict:
    """One result the rollout took out of the cycle, with the miner that reported it."""

    miner_hotkey: str
    result: JobResult
    keeps_miner_active: bool = True


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
    previous_current_cycles_seen: int = 0
    earlier_current: tuple[tuple[str, int], ...] = ()

    def covers(self, job_block: int) -> bool:
        if self.grace_cycles <= 0 or self.opened_job_block is None:
            return False
        return job_block >= self.opened_job_block and 0 < self.cycles_seen <= self.grace_cycles

    def previous_reads_current(self, job_block: int) -> bool:
        """DAH-6560: a node still on the release just replaced reads CURRENT, not OUTDATED, in this cycle."""
        if self.previous_digest is None or self.opened_job_block is None or job_block < self.opened_job_block:
            return False
        return 0 < self.previous_current_cycles_seen <= PREVIOUS_RELEASE_CURRENT_CYCLES

    def earlier_reading_current(self, job_block: int) -> tuple[str, ...]:
        """Releases replaced before the previous one that still read CURRENT in this cycle."""
        if self.opened_job_block is None or job_block < self.opened_job_block:
            return ()
        return tuple(d for d, n in self.earlier_current if 0 < n <= PREVIOUS_RELEASE_CURRENT_CYCLES)

    def as_extra(self) -> dict[str, object]:
        return {
            "rollout_digest": self.digest,
            "rollout_previous_digest": self.previous_digest,
            "rollout_started_at": self.started_at.isoformat() if self.started_at else None,
            "rollout_opened_job_block": self.opened_job_block,
            "rollout_cycles_seen": self.cycles_seen,
            "rollout_grace_cycles": self.grace_cycles,
            "rollout_previous_current_cycles_seen": self.previous_current_cycles_seen,
            "rollout_earlier_current": [list(entry) for entry in self.earlier_current],
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
        counted = replace(
            state,
            last_job_block=job_block,
            cycles_seen=state.cycles_seen + 1,
            previous_current_cycles_seen=state.previous_current_cycles_seen
            + (1 if state.previous_current_cycles_seen else 0),
            earlier_current=tuple(
                (d, n + 1) for d, n in state.earlier_current if n + 1 <= PREVIOUS_RELEASE_CURRENT_CYCLES
            ),
        )
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
                previous_current_cycles_seen=1,
                earlier_current=state.earlier_after_replacing(digest),
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
        # its previous release (the one just replaced) gets a fresh CURRENT grace of its own
        chained = replace(
            state,
            digest=digest,
            previous_digest=state.digest,
            previous_current_cycles_seen=1,
            earlier_current=state.earlier_after_replacing(digest),
        )
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
            previous_current_cycles_seen=state.previous_current_cycles_seen,
            earlier_current=state.earlier_current,
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
    digest. The OUTDATED-without-failing case counts only while `EXECUTOR_IMAGE_CHECK_ENFORCE` is
    on: off (the default since DAH-3439), the image check passes an OUTDATED node and leaves its
    score alone, so a score of 0 under an OUTDATED report came from another gate (price cap, TDX) and stands. An executor that already runs
    the new image failed for a reason of its own; a rented executor that failed a later check of
    its own (pod not running, filler killed) failed for that reason, OUTDATED or not; and a result
    with a score is never touched.
    """
    if not window.covers(job_block) or result.score > 0:
        return None
    reason = result.failure_reason_code
    if reason not in ROLLOUT_FAILURE_REASONS:
        ended_without_failing = reason in RUN_ENDED_WITHOUT_FAILING
        outdated = (
            settings.EXECUTOR_IMAGE_CHECK_ENFORCE
            and (result.executor_image_report or {}).get("status") == ImageVerdict.OUTDATED.value
        )
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
                        "is_rented": result.is_rented,
                        **window.as_extra(),
                    },
                )
            )
        standing_by_miner[miner_hotkey] = standing_results
    return standing_by_miner, withheld


_PORT_CHECK_DEFERRED = PortCountMessages.VERDICT_WITHHELD_PORT_CHECK_DEFERRED.reason


def withhold_renter_gpu_reset_verdicts(
    job_results: dict[str, list[JobResult]],
) -> tuple[dict[str, list[JobResult]], list[WithheldVerdict]]:
    """DAH-4658: a scrape held for the renter's own GPU reset gets no verdict, as a rollout failure gets none.
    DAH-5805: neither does a run the validator's own Redis ended; the node is not scored 0 for it.
    DAH-5933: nor a rental probe the task had no time left for, when no last scored result was carried.
    DAH-5851: neither does an idle node's port check deferred for disk pressure; it is never carried, so no
    idle pay comes from the republished ports. Neither does a run whose login uid could not be read (LOGIN_USER_UNREAD)."""
    standing_by_miner: dict[str, list[JobResult]] = {}
    withheld: list[WithheldVerdict] = []
    for miner_hotkey, results in job_results.items():
        standing_by_miner[miner_hotkey] = []
        for result in results:
            redis_fault = result.failure_reason_code == VALIDATOR_REDIS_UNAVAILABLE
            port_check_deferred = result.failure_reason_code == _PORT_CHECK_DEFERRED
            # the login uid could not be read: the run stopped before any check, so it says nothing about the node
            login_unread = result.failure_reason_code == LoginUserMessages.UNREAD.reason
            if not (redis_fault or port_check_deferred or login_unread or _held_for_renter_gpu_reset(result)):
                standing_by_miner[miner_hotkey].append(result)
                continue
            withheld.append(
                WithheldVerdict(
                    miner_hotkey=miner_hotkey,
                    result=result,
                    # the node's own checks can spend the task's time, and an unread uid passed no check: neither
                    # keeps a miner active (review)
                    keeps_miner_active=not login_unread and result.failure_reason_code not in NO_TASK_TIME_REASONS,
                )
            )
            logger.info(
                _m(
                    "Verdict withheld: the validator's Redis was unavailable"
                    if redis_fault
                    else "Verdict withheld: the port check was deferred for disk pressure"
                    if port_check_deferred
                    else "Verdict withheld: the validator's login uid could not be read"
                    if login_unread
                    else "Verdict withheld: the renter is resetting the node's GPUs, its power tier is unknown, or the task had no time left for a check",
                    extra={
                        "miner_hotkey": miner_hotkey,
                        "executor_uuid": result.executor_info.uuid,
                        "job_batch_id": result.job_batch_id,
                    },
                )
            )
    return standing_by_miner, withheld


def withhold_session_lost_verdicts(
    job_results: dict[str, list[JobResult]],
) -> tuple[dict[str, list[JobResult]], list[WithheldVerdict]]:
    """DAH-5718: the validator's session died in the port check, so the cycle is unscored, in a rollout or not."""
    standing_by_miner: dict[str, list[JobResult]] = {}
    withheld: list[WithheldVerdict] = []
    for miner_hotkey, results in job_results.items():
        standing_by_miner[miner_hotkey] = []
        for result in results:
            if not result.session_lost_in_port_check:
                standing_by_miner[miner_hotkey].append(result)
                continue
            withheld.append(WithheldVerdict(miner_hotkey=miner_hotkey, result=result))
            logger.info(
                _m(
                    "Verdict withheld: the validator's SSH session closed during the port check",
                    extra={
                        "miner_hotkey": miner_hotkey,
                        "executor_uuid": result.executor_info.uuid,
                        "job_batch_id": result.job_batch_id,
                    },
                )
            )
    return standing_by_miner, withheld


# DAH-6564: failures a rental or filler that started or ended during the cycle produces (the checks judge the node
# against the list read at the cycle start)
RENTAL_CHANGE_REASONS = frozenset(
    {
        PortCountMessages.INSUFFICIENT_PORTS.reason,
        RentalVerificationMessages.FAILED.reason,
        GpuUsageMessages.USAGE_HIGH.reason,
    }
)
# one hash for the fleet: "<miner hotkey>:<executor uuid>" -> {"batches": [withheld job_batch_ids], "at": epoch}
RENTAL_CHANGE_STATE_KEY = "rental_change_withheld"
# cycles withheld by this rule since the node last passed; the backend's 45 min inactive sweep is not reset by it
MAX_RENTAL_CHANGE_WITHHELD_CYCLES = 2
RENTAL_CHANGE_STATE_MAX_AGE_SECONDS = 7 * 24 * 3600


def has_rental_change_candidates(job_results: dict[str, list[JobResult]]) -> bool:
    return any(
        result.failure_reason_code in RENTAL_CHANGE_REASONS for results in job_results.values() for result in results
    )


def node_rentals(rented_data: RentedExecutorsResponse, miner_hotkey: str, result: JobResult) -> frozenset[str]:
    """The node's pod ids and filler container names in one rented list, owned as the checks own them
    (checks/rental_owner.py): the executor uuid is the miner's word."""
    executor = result.executor_info
    uuid = str(executor.uuid)
    rental = rental_owned_by(rented_data.executors.get(uuid), miner_hotkey, executor.address, executor.port)
    pods = {f"pod:{pod.pod_id}" for pod in rental.pods} if rental else set()
    if listed_under_another_miner(rented_data, uuid, miner_hotkey):
        return frozenset(pods)
    return frozenset(pods | {f"filler:{name}" for name in rented_data.get_filler_containers(uuid)})


def _rental_change(
    result: JobResult,
    miner_hotkey: str,
    at_start: RentedExecutorsResponse,
    at_end: RentedExecutorsResponse,
    published_early: set[str],
) -> tuple[frozenset[str], frozenset[str]] | None:
    """(rentals at the cycle start, at its end) when the rule may withhold this failure, else None."""
    reason = result.failure_reason_code
    if reason not in RENTAL_CHANGE_REASONS or str(result.executor_info.uuid) in published_early:
        return None
    # a node that declares fewer ports than it needs fails whatever its rentals are
    if result.validation_event and _is_the_nodes_own_fault(result.validation_event.model_dump()):
        return None
    # an earlier check's finding (a GPU fault) hides the node and must be published
    if any(error.get("reason_code") != reason for error in result.availability_errors or []):
        return None
    before, after = node_rentals(at_start, miner_hotkey, result), node_rentals(at_end, miner_hotkey, result)
    return None if before == after else (before, after)


async def withhold_rental_change_verdicts(
    job_results: dict[str, list[JobResult]],
    rented_at_start: RentedExecutorsResponse,
    rented_at_end: RentedExecutorsResponse | None,
    redis_service: RedisService,
    published_early: set[str],
) -> tuple[dict[str, list[JobResult]], list[WithheldVerdict]]:
    """DAH-6564: a ports, rental-check or GPU-usage failure on a node whose pods or fillers changed between the
    cycle-start list and the cycle-end list gets no verdict, at most MAX_RENTAL_CHANGE_WITHHELD_CYCLES cycles
    since the node's last real pass. No cycle-end list, or Redis not answering: every verdict stands."""
    candidates = has_rental_change_candidates(job_results)
    if candidates and rented_at_end is None:
        logger.warning(_m("Rental-change rule: the cycle-end rented list gave no answer; every verdict stands"))
    try:
        state = {
            (field.decode() if isinstance(field, bytes) else field): json.loads(raw)
            for field, raw in (await redis_service.hgetall(RENTAL_CHANGE_STATE_KEY)).items()
        }
    except Exception as exc:
        if candidates:
            logger.warning(_m("Rental-change rule: state unreadable; every verdict stands", extra={"error": str(exc)}))
        return job_results, []
    now = time.time()
    writes = RedisWrites()
    standing: dict[str, list[JobResult]] = {}
    pending: list[tuple[WithheldVerdict, dict]] = []
    for miner_hotkey, results in job_results.items():
        standing[miner_hotkey] = []
        for result in results:
            field = f"{miner_hotkey}:{result.executor_info.uuid}"
            change = None
            if result.is_successful and not result.carried_for_renter_gpu_reset:
                if field in state:
                    writes.hdel(RENTAL_CHANGE_STATE_KEY, field)
                    state.pop(field)
            elif rented_at_end is not None:
                change = _rental_change(result, miner_hotkey, rented_at_start, rented_at_end, published_early)
            if change is None:
                standing[miner_hotkey].append(result)
                continue
            before, after = change
            batches = state.get(field, {}).get("batches", [])
            extra = {
                "miner_hotkey": miner_hotkey,
                "executor_uuid": result.executor_info.uuid,
                "job_batch_id": result.job_batch_id,
                "reason_code": result.failure_reason_code,
                "started": sorted(after - before),
                "ended": sorted(before - after),
            }
            if result.job_batch_id not in batches and len(batches) >= MAX_RENTAL_CHANGE_WITHHELD_CYCLES:
                logger.info(
                    _m(
                        "Rental-change rule: cap reached, the verdict stands",
                        extra={**extra, "withheld_cycles_since_pass": len(batches)},
                    )
                )
                standing[miner_hotkey].append(result)
                continue
            if result.job_batch_id not in batches:
                batches = [*batches, result.job_batch_id]
            state[field] = {"batches": batches, "at": now}
            writes.hset(RENTAL_CHANGE_STATE_KEY, field, json.dumps(state[field]))
            pending.append(
                (
                    WithheldVerdict(miner_hotkey=miner_hotkey, result=result, keeps_miner_active=False),
                    {**extra, "withheld_cycles_since_pass": len(batches)},
                )
            )
    for field, entry in state.items():
        if now - entry.get("at", 0) > RENTAL_CHANGE_STATE_MAX_AGE_SECONDS:
            writes.hdel(RENTAL_CHANGE_STATE_KEY, field)
    try:
        await redis_service.write_atomically(writes)
    except Exception as exc:
        if pending:
            logger.warning(_m("Rental-change rule: state not written; every verdict stands", extra={"error": str(exc)}))
        return job_results, []
    for withheld, extra in pending:
        logger.info(_m("Verdict withheld: the node's rentals or fillers changed during the cycle", extra=extra))
    return standing, [withheld for withheld, _ in pending]


def session_kept_hotkeys(withheld_results: list[WithheldVerdict]) -> set[str]:
    """The miners a withheld verdict keeps active this cycle. Not one whose session died in the port check (review
    4469609776): a miner can close SSH itself, and that must not earn the eligibility floor. Nor one withheld for a
    rental change (DAH-6564): a provider can change his own node's rentals."""
    return {
        withheld.miner_hotkey
        for withheld in withheld_results
        if withheld.keeps_miner_active and not withheld.result.session_lost_in_port_check
    }


def has_renter_gpu_reset_holds(job_results: dict[str, list[JobResult]]) -> bool:
    return any(_held_for_renter_gpu_reset(result) for results in job_results.values() for result in results)


# a cycle that cannot tell the node's state gives no verdict: the renter's GPU reset (DAH-4658), a power tier
# reading whose clear was not confirmed (F-43), or a check the task had no time left for (DAH-5933)
_HELD_REASONS = (
    MachineSpecMessages.SCRAPE_GPU_RESET_IN_PROGRESS.reason,
    GpuPowerLimitMessages.TIER_STATE_UNKNOWN.reason,
    *CYCLE_END_READ_REASONS,
)


def _held_for_renter_gpu_reset(result: JobResult) -> bool:
    return result.failure_reason_code in _HELD_REASONS


# rental and pay flags the held result of this cycle reads fresh from the cycle's rental snapshot
_FRESH_PAY_FIELDS = (
    "is_spot",
    "is_on_spot_pay_list",
    "is_provider_email_held",
    "provider_discord_connected",
    "is_new_rentals_paused",
    "default_job_owner",
    "has_lium_filler",
)


def rental_pod_ids(rental: RentedExecutor | None) -> frozenset[str] | None:
    """The rental state a carry compares: no rental -> empty, a rental -> its pod ids, one listed with no pod -> None."""
    if rental is None:
        return frozenset()
    if not rental.pods:
        return None
    return frozenset(str(pod.pod_id) for pod in rental.pods)


def rental_state_unchanged(
    held: JobResult,
    previous: JobResult | None,
    rental_at_start: RentedExecutor | None,
    rental_at_end: RentedExecutor | None,
) -> bool:
    """The one rule every carry goes through (DAH-5933): a kept result stands in for a held cycle only when it was
    scored with exactly the node's rental state that cycle saw at its start (the snapshot) and at the cycle-end read:
    no rental, or the same rental pod ids holding the same known GPU count. A cycle that itself saw the node rented
    after its start (RENTAL_STARTED_DURING_CHECK) saw a change. Anything else is withheld: never scored or paid.
    `rental_at_start` / `rental_at_end` are the node's own rentals (rental_owned_by), None for none."""
    if held.failure_reason_code == RentalVerificationMessages.RENTED_SINCE_THE_SNAPSHOT.reason:
        return False
    if previous is None or previous.rental_pod_ids is None:
        return False
    at_start, at_end = rental_pod_ids(rental_at_start), rental_pod_ids(rental_at_end)
    if at_start is None or at_end is None or not at_start == at_end == frozenset(previous.rental_pod_ids):
        return False
    if previous.is_rented != bool(at_end):
        return False
    if not at_end:
        return True
    # a split node is paid rented and idle GPUs apart: the same pods holding another count, or a count nobody knows,
    # is not the allocation the kept result was scored with
    count_at_end = rental_gpu_count(rental_at_end)
    return count_at_end is not None and rental_gpu_count(rental_at_start) == count_at_end == previous.rented_gpu_count


async def carry_renter_gpu_reset_results(
    job_results: dict[str, list[JobResult]],
    redis_service: RedisService,
    rented_executor_ids: set[str],
    rented_data: RentedExecutorsResponse | None,
    miner_coldkeys: dict[str, str | None],
    rented_at_end: RentedExecutorsResponse | None = None,
    *,
    job_block: int,
) -> None:
    """DAH-4658: a held scrape (the renter's own GPU reset, an unconfirmed power tier, no task time) takes the node's
    last scored result, in place, scored as in the cycle it came from with this cycle's pay flags and a fresh ban
    lookup. It carries only by `rental_state_unchanged`, on the cycle-start snapshot `rented_data` and the cycle-end
    read `rented_at_end`. With no end read (None: the backend gave none) a GPU-reset or power-tier hold still carries,
    on the start snapshot: its own scrape ran inside that rental; a no-task-time or rented-during-check one does not. A kept result that fails the rule is
    dropped, and so is one a cycle saw rented after its start even when a read is missing: the rental state it was
    scored with is gone. A held result not carried is taken out by
    `withhold_renter_gpu_reset_verdicts`. The GPU-reset and power-tier holds carry a rented node only; a no-task-time
    hold carries a rented or an idle node once (`_spend_kept_result`, `_take_idle_pass`). A kept result is eligible
    only while every cycle since it was scored vouched for it (`_last_scored_result`, `_take_idle_pass`).
    `rented_executor_ids` is kept for the caller's signature.
    """
    def by_uuid(rented: RentedExecutorsResponse | None) -> dict[str, RentedExecutor] | None:
        return None if rented is None else {str(uuid).lower(): rental for uuid, rental in rented.executors.items()}

    rentals_at_start, rentals_at_end = by_uuid(rented_data), by_uuid(rented_at_end)
    for miner_hotkey, results in job_results.items():
        for index, held in enumerate(results):
            if not _held_for_renter_gpu_reset(held):
                continue
            previous = _take_idle_pass(miner_hotkey, held, job_block)
            in_redis = previous is None
            if in_redis:
                previous = await _last_scored_result(redis_service, miner_hotkey, held, job_block)
            if previous is None:
                continue
            # a GPU-reset or power-tier hold carries a rented node only, on its start snapshot as before this change:
            # its own scrape ran inside that rental. A no-task-time or rented-during-check cycle needs the end read
            needs_end_read = held.failure_reason_code in CYCLE_END_READ_REASONS
            if rentals_at_start is None or (rentals_at_end is None and needs_end_read):
                # a rental state that cannot be read carries nothing; one the cycle saw change still drops the kept result
                seen_rented = held.failure_reason_code == RentalVerificationMessages.RENTED_SINCE_THE_SNAPSHOT.reason
                if in_redis and seen_rented:
                    await _spend_kept_result(redis_service, miner_hotkey, held)
                continue
            info = held.executor_info
            uuid = str(info.uuid).lower()
            # owned as the checks own it: the uuid is the miner's word, so another miner's rental on another endpoint
            # under it is not this node's (review at a8afce8)
            rental_at_start = rental_owned_by(rentals_at_start.get(uuid), miner_hotkey, info.address, info.port)
            rental_at_end = (
                rental_at_start
                if rentals_at_end is None
                else rental_owned_by(rentals_at_end.get(uuid), miner_hotkey, info.address, info.port)
            )
            if not rental_state_unchanged(held, previous, rental_at_start, rental_at_end):
                if in_redis:
                    await _spend_kept_result(redis_service, miner_hotkey, held)
                continue
            no_task_time = held.failure_reason_code in NO_TASK_TIME_REASONS
            rented_now = rental_at_end is not None
            if not no_task_time and not rented_now:
                continue
            rented_gpu_count = rental_gpu_count(rental_at_end) if rented_now else 0
            if no_task_time and in_redis and not await _spend_kept_result(redis_service, miner_hotkey, held):
                continue
            results[index] = previous.model_copy(
                update={
                    "executor_info": held.executor_info,
                    "job_batch_id": held.job_batch_id,
                    "carried_for_renter_gpu_reset": True,
                    **{field: getattr(held, field) for field in _FRESH_PAY_FIELDS},
                    "is_provider_banned": _banned_now(
                        rented_data, miner_hotkey, miner_coldkeys.get(miner_hotkey), previous, held
                    ),
                    "filler_revenue_per_gpu_hour": _filler_revenue_now(rented_data, previous),
                    "filler_revenue_usd_per_hour": (
                        rented_data.get_filler_revenue_usd_per_hour(
                            BASE_GPU_MAP.get(previous.gpu_model or ""), settings.FILLER_REVENUE_MIN_GPU_HOURS
                        )
                        if rented_data
                        else previous.filler_revenue_usd_per_hour
                    ),
                    "rented_gpu_count": rented_gpu_count or previous.rented_gpu_count,
                }
            )
            logger.info(
                _m(
                    "Score carried: the task had no time left for a check, the last scored result stands"
                    if no_task_time
                    else "Score carried: the renter is resetting the node's GPUs, the last scored result stands",
                    extra={
                        "miner_hotkey": miner_hotkey,
                        "executor_uuid": held.executor_info.uuid,
                        "job_batch_id": held.job_batch_id,
                        "carried_from_job_batch_id": previous.job_batch_id,
                        "carried_score": previous.score,
                        "gpu_model": previous.gpu_model,
                        "gpu_count": previous.gpu_count,
                    },
                )
            )


def _banned_now(
    rented_data: RentedExecutorsResponse | None,
    miner_hotkey: str,
    miner_coldkey: str | None,
    previous: JobResult,
    held: JobResult,
) -> bool:
    """BannedProviderCheck never ran on the held cycle: look the ban up as it does, on the carried GPUs."""
    if rented_data is None:
        return held.is_provider_banned or previous.is_provider_banned
    gpu_details = ((previous.spec or {}).get("gpu") or {}).get("details") or []
    return held.is_provider_banned or rented_data.is_provider_banned(
        miner_hotkey=miner_hotkey,
        miner_coldkey=miner_coldkey,
        gpu_uuids=[detail["uuid"] for detail in gpu_details if detail.get("uuid")],
        executor_uuid=str(held.executor_info.uuid),
    )


def _filler_revenue_now(rented_data: RentedExecutorsResponse | None, previous: JobResult) -> float | None:
    """This cycle's filler average for the carried GPU configuration (the held scrape may list none)."""
    if rented_data is None:
        return previous.filler_revenue_per_gpu_hour
    return rented_data.get_filler_revenue_per_gpu_hour(
        BASE_GPU_MAP.get(previous.gpu_model or ""),
        previous.gpu_count,
        settings.get_filler_revenue_min_gpu_hours(previous.is_spot),
    )


async def _spend_kept_result(redis_service: RedisService, miner_hotkey: str, held: JobResult) -> bool:
    """Drops the node's kept result. A cycle the task had no time for carries the last pass once: the next such
    cycle, with no pass between, is withheld. The node's own checks spend the task's time, so a node that keeps
    running out of it never keeps its score (review). A kept result invalidated by a rental change is dropped here
    too. False when it could not be deleted: the held result is withheld then, and since a withheld cycle vouches for
    nothing, the row left behind is not eligible in any later cycle either (`_last_scored_result`)."""
    try:
        await redis_service.delete(last_scored_result_key(miner_hotkey, held.executor_info.uuid))
        return True
    except Exception as exc:
        logger.warning(
            _m(
                "Kept result could not be dropped; the held verdict is withheld",
                extra={"miner_hotkey": miner_hotkey, "executor_uuid": held.executor_info.uuid, "error": str(exc)},
            )
        )
        return False


# DAH-5933: each node's idle pass of the last cycle, in this process only and never in Redis: taking or dropping one
# cannot fail, and a restart starts with none (the next held cycle is withheld). (result, job block it was scored on)
_idle_passes: dict[tuple[str, str], tuple[JobResult, int]] = {}


def _take_idle_pass(miner_hotkey: str, held: JobResult, job_block: int) -> JobResult | None:
    """The node's idle pass kept by the cycle right before this one, or None. Taken on read, so whatever this cycle
    decides, no later cycle finds it; only a pass this cycle scores is kept for the next."""
    kept = _idle_passes.pop((miner_hotkey, held.executor_info.uuid), None)
    if kept is None or kept[1] < job_block - settings.BLOCKS_FOR_JOB:
        return None
    return kept[0]


async def _last_scored_result(
    redis_service: RedisService, miner_hotkey: str, held: JobResult, job_block: int
) -> JobResult | None:
    """The node's kept rented result (an idle one is `_take_idle_pass`'s), or None: none, unreadable, or not vouched
    for by the cycle right before this one (the held result is then withheld, as before). Eligibility is a fact of
    the store, not of a process: every cycle that keeps the result eligible writes its job block with the cycle's
    results (`remember_rented_scored_results`). A cycle that dropped it, withheld the node, or wrote nothing at all
    (a Redis outage, a restart) leaves no vouch, so a row such a cycle could not delete is never read as eligible
    again. The read takes the vouch it finds, so a cycle that shares the job block of the one before it cannot
    leave that cycle's vouch behind."""
    try:
        # the read spends the vouch: only this cycle's own write can vouch for the next one, so a cycle that drops the
        # result (on any job block) and cannot delete it leaves nothing a later cycle accepts
        raw, vouch = await redis_service.read_and_take(
            last_scored_result_key(miner_hotkey, held.executor_info.uuid),
            last_scored_result_vouch_key(miner_hotkey, held.executor_info.uuid),
        )
        if not raw:
            return None
        # a cycle an operator asks for runs on the job block of the cycle before it, so the vouch may be this block too
        if vouch is None or int(vouch) < job_block - settings.BLOCKS_FOR_JOB:
            logger.info(
                _m(
                    "Last scored result not vouched for by the previous cycle; the held verdict is withheld",
                    extra={"miner_hotkey": miner_hotkey, "executor_uuid": held.executor_info.uuid, "vouch": vouch},
                )
            )
            return None
        return JobResult.model_validate_json(raw)
    except Exception as exc:
        logger.warning(
            _m(
                "Last scored result unreadable; the held verdict is withheld",
                extra={"miner_hotkey": miner_hotkey, "executor_uuid": held.executor_info.uuid, "error": str(exc)},
            )
        )
        return None


async def remember_rented_scored_results(
    job_results: dict[str, list[JobResult]], redis_service: RedisService, *, job_block: int
) -> None:
    """DAH-4658: keep each node's scored result of this cycle for `carry_renter_gpu_reset_results`.

    Any other standing result drops the kept one, so only the cycle right before a reset is carried.
    A carried result is not kept again, so the TTL bounds how long one result can stand in. A Redis
    error only logs: the next cycle writes again.

    A rented pass is kept in Redis, as on main, and in the same write each node whose kept rented result stays
    eligible (a new pass, or a carried one) is vouched for at `job_block`; a node withheld this cycle is left out of
    `job_results` and gets no vouch (`_last_scored_result`). DAH-5933: an idle pass is kept in this process only
    (`_take_idle_pass`), and the cycle's idle passes replace the last cycle's, so a node withheld this cycle has none.
    """
    writes = RedisWrites()
    idle_passes: dict[tuple[str, str], tuple[JobResult, int]] = {}
    for miner_hotkey, results in job_results.items():
        for result in results:
            vouch = last_scored_result_vouch_key(miner_hotkey, result.executor_info.uuid)
            if result.carried_for_renter_gpu_reset:
                if result.is_rented:
                    writes.set(vouch, str(job_block), ex=LAST_SCORED_RESULT_TTL_SECONDS)
                continue
            key = last_scored_result_key(miner_hotkey, result.executor_info.uuid)
            if result.is_rented and result.is_successful:
                writes.set(key, result.model_dump_json(), ex=LAST_SCORED_RESULT_TTL_SECONDS)
                writes.set(vouch, str(job_block), ex=LAST_SCORED_RESULT_TTL_SECONDS)
                continue
            writes.delete(key)
            writes.delete(vouch)
            if result.is_successful:
                # a copy as it stands now, as the rented pass's Redis copy is: scoring later sets flags on `result`
                idle_passes[(miner_hotkey, result.executor_info.uuid)] = (result.model_copy(deep=True), job_block)
    _idle_passes.clear()
    _idle_passes.update(idle_passes)
    try:
        await redis_service.write_atomically(writes)
    except Exception as exc:
        logger.error(_m("Could not keep the cycle's scored results", extra={"error": str(exc)}))
