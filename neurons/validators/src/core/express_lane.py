"""DAH-2958: verify and publish never-validated executors ahead of the 15-min cycle.

A new node is published only by the fleet-wide scored cycle, so it waits for the cycle boundary
(BLOCKS_FOR_JOB = 75 blocks, uniform 0–15 min) and then for the slowest miner in the fleet
(publish_machine_specs runs after asyncio.wait over every miner: +2–9 min). Node add → AVAILABLE
was p50 21.4 / p90 63.5 min over 175 onboardings, floor 8.2 min (RECOVERY report, 6 Sep 2026).

This lane runs beside Validator.sync() in the same process. Every tick it reads the portal's bulk
executor snapshot, picks the executors assigned to this validator that registered after the first
cycle since start began and that no cycle has published yet, and runs the SAME pipeline the cycle
runs — same job files, digests and image snapshot, same checks, as a first pass (DAH-3011) — on
each one, alone, then publishes the result spec-only (scored_at stays None, so the backend creates
the executor row but writes no incentive ledger row) under that cycle's job_batch_id, so its
prod_executors row lands on the cycle's time. The next scored cycle overwrites it as today.

The portal snapshot lists a self-hosted miner's node only after its first publish, so every
SELF_HOSTED_LIST_SECONDS the lane also asks each serving miner that is not routed through the
central miner for its executor list (the miner's POST /executors, a DB read) and adds the nodes
that appeared since its first answer to the snapshot.
"""

import asyncio
import logging
import os
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import bittensor
from clients.validator_portal_api import PortalExecutor, ValidatorPortalAPI
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.executor_image_policy import ExpectedImageSnapshot
from services.miner_service import EXPRESS_LANE, MinerService
from services.redis_service import EXPRESS_REDIS_SLOT
from services.task_service import JobResult

from core.config import settings
from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

# One line per express verification; its registration_to_publish_s is the deploy metric.
EXPRESS_PUBLISHED_EVENT = "[express] Executor verified and published ahead of the cycle"
# Same line for a first pass that ran to a failed result (published so the provider sees why).
EXPRESS_FAILED_EVENT = "[express] Executor failed its first check, result published ahead of the cycle"
# A restart drops the in-memory queue and an express check in flight. Nodes registered this long
# before the first cycle since start and still not in the validated set get the lane again.
RESTART_LOOKBACK = timedelta(minutes=30)
# The miner did not return the executor (its portal snapshot is not refreshed yet, or the node is
# unreachable): try again later, a bounded number of times, then leave it to the normal cycle.
MAX_ATTEMPTS = 3
RETRY_SECONDS = 120
# The one deferral the validation fast path shortens (settings.EXPRESS_LANE_MINER_SNAPSHOT_RETRY_SECONDS):
# the central miner serves executors from a portal snapshot it refreshes every 30 s, so a node this
# validator's snapshot already lists is often one refresh away on the miner's side. Every other
# deferral keeps RETRY_SECONDS.
MINER_DID_NOT_RETURN_EXECUTOR = "miner did not return the executor"
# A first pass that failed is published (the provider sees why) and, with
# EXPRESS_LANE_FAILED_FIRST_PASS_RETRY_ENABLED, run again after this long instead of waiting for
# the next cycle (0–15 min to its boundary, then the wave): a node still pulling its images or
# drawing a cold bandwidth sample often passes a minute later. Same pipeline, same verdict rules.
FIRST_PASS_FAILED = "first pass failed"
FIRST_PASS_FAILED_RETRY_SECONDS = 60
FIRST_PASS_MAX_RUNS = 3
SESSION_LOST_IN_PORT_CHECK = "the validator's SSH session closed during the port check"
# A lost session is the validator's, not the node's, so it spends none of MAX_ATTEMPTS (which the
# miner's snapshot misses may already have used). Its own bound keeps a node whose session always
# drops from staying queued, and hidden from the wave, for good: ~20 min at RETRY_SECONDS.
SESSION_LOST_MAX_RUNS = 10
# Self-hosted miners' executor lists: how often, how many miners at once, and the longest a miner
# that keeps not answering is skipped (one miss skips one interval, doubling per further miss).
# Each ask is bounded by REST_EXECUTOR_LIST_TIMEOUT (miner_service).
SELF_HOSTED_LIST_SECONDS = 60
SELF_HOSTED_LIST_CONCURRENCY = 8
SELF_HOSTED_LIST_FAILURE_BACKOFF_SECONDS = 600
# job_batch_id format the backend parses into the prod_executors row's time.
JOB_BATCH_ID_FORMAT = "%Y-%m-%d %H:%M:%S"
# DAH-5716: a requested recheck of an already-validated executor must end before the next wave
# lists its miner (the wave leaves out what this lane is running). The next cycle starts no
# earlier than BLOCKS_FOR_JOB blocks after this one; a recheck starts only with at least
# RECHECK_MIN_SECONDS left, and is cut off RECHECK_MARGIN_SECONDS before that start.
SECONDS_PER_BLOCK = 12
RECHECK_MIN_SECONDS = 300
RECHECK_MARGIN_SECONDS = 60
# A request stays queued until its run publishes, for at most this many runs.
RECHECK_MAX_RUNS = 2


def _cycle_batch_id_or_clock(cycle_batch_id: str) -> str:
    """The cycle's id, or the clock when the cycle could not read its block time
    (SubtensorClient.get_time_from_block gives "Unknown"): the backend's write of a spec whose id
    does not parse fails before it lists the node, and a node checked ahead of the cycle must
    still be published."""
    try:
        datetime.strptime(cycle_batch_id, JOB_BATCH_ID_FORMAT)
    except (TypeError, ValueError):
        return datetime.now(UTC).strftime(JOB_BATCH_ID_FORMAT)
    return cycle_batch_id


@dataclass
class CycleInputs:
    """What the cycle prepared for its wave. The express lane reuses it verbatim, so an express
    verification is the cycle's pipeline with the cycle's job files, digests and image snapshot."""

    encrypted_files: MinerJobEnryptedFiles
    default_image_digests: dict[str, str]
    executor_image_snapshot: ExpectedImageSnapshot | None
    # The cycle's own job_batch_id. The backend parses it into the prod_executors row's time, and
    # the fleet charts group rows by that time, so an express publish carries it: with an id of
    # its own the node plots as a 1-2 node point between two fleet points.
    job_batch_id: str
    # When the first cycle since this process started began. That cycle asked every serving miner
    # for its executors, so an executor registered before it is long-known even when its miner
    # failed or was offline that cycle and it never reached the validated set; only executors
    # registered after this instant can be new to this validator.
    fleet_known_since: datetime
    # When this cycle's inputs were made (monotonic), the earliest base for the next cycle's start.
    started_at: float = field(default_factory=time.monotonic)
    # DAH-6013: executor id -> end (UTC) of the provider maintenance window the backend still holds it for. The
    # lane rechecks each node once its window has ended instead of leaving it to the next cycle.
    maintenance_ends: dict[str, datetime] = field(default_factory=dict)


@dataclass
class _Pending:
    executor: PortalExecutor
    miner_hotkey: str
    first_seen_at: datetime
    # first_seen_at replaced by the stamp in Redis, which outlives a validator restart
    first_seen_restored: bool = False
    attempts: int = 0
    not_before: float = 0.0  # monotonic
    # First passes that ran to a failed result; counted apart from attempts, which the miner's
    # snapshot misses may already have spent.
    failed_runs: int = 0
    # Runs whose validator session closed during the port check (no verdict).
    lost_sessions: int = 0
    # DAH-5716: set for a recheck of an already-validated executor; monotonic end of its run.
    recheck_deadline: float | None = None
    # Taken only because it was requested: never queued in _pending, so the wave never leaves it out.
    recheck_only: bool = False
    # Picked this tick because a recheck was requested; its run spends the request's budget.
    requested: bool = False
    # The request generation this run started from; ending it leaves a newer request queued (review 4510532999).
    recheck_generation: int | None = None


class _LaneOwnedNodes:
    """What the wave leaves to the lane (MinerService.express_queued): the lane's queue, minus a
    node whose failed-first-pass retry is due. The lane launches nothing for a miner the wave has
    not listed yet, and each cycle start marks every miner so, so a due retry could wait cycle
    after cycle while the wave kept leaving the node to the lane (DAH-6185); the wave runs it.
    A live view of the queue, never a copy."""

    def __init__(self, pending: dict[str, _Pending]):
        self._pending: dict[str, _Pending] = pending

    def _is_lane_owned(self, executor_id: str, now: float) -> bool:
        pending = self._pending.get(executor_id)
        return pending is not None and not (pending.failed_runs and pending.not_before <= now)

    def __contains__(self, executor_id: object) -> bool:
        return isinstance(executor_id, str) and self._is_lane_owned(executor_id, time.monotonic())

    def __iter__(self) -> Iterator[str]:
        now = time.monotonic()
        return iter([e for e in self._pending if self._is_lane_owned(e, now)])


class ExpressLane:
    def __init__(
        self,
        *,
        miner_service: MinerService,
        redis_service,
        backend_client,
        subtensor_client,
        cycle_inputs: Callable[[], CycleInputs | None],
        portal_api=ValidatorPortalAPI,
    ):
        self.miner_service = miner_service
        self.redis_service = redis_service
        self.backend_client = backend_client
        self.subtensor_client = subtensor_client
        # None until the first cycle since start has completed: that cycle publishes every
        # executor of every miner that answered and produces the job files, so nothing runs
        # before it.
        self.cycle_inputs = cycle_inputs
        self.portal_api = portal_api
        self._pending: dict[str, _Pending] = {}
        # The wave leaves what this lane has queued to it (MinerService._claim_for_cycle), through
        # a view of this dict, never rebound, so the wave always reads the current queue.
        self.miner_service.express_queued = _LaneOwnedNodes(self._pending)
        self._tasks: set[asyncio.Task] = set()
        self._my_hotkey: str | None = None
        # Given up on this process's watch (MAX_ATTEMPTS without the miner returning them); the
        # normal cycle owns them from here. In memory on purpose: a restart may try again.
        self._left_to_cycle: set[str] = set()
        # Job-files directory -> running verifications reading it. The validator keeps these
        # directories when it prepares the next cycle's files (FileEncryptService).
        self._directories_in_use: Counter[str] = Counter()
        # Self-hosted miner hotkey -> {executor id: when this lane first saw it listed}. None for
        # what the miner's first answer listed: known before this lane looked, the cycle's.
        self._self_hosted: dict[str, dict[str, datetime | None]] = {}
        self._self_hosted_backoff: dict[str, float] = {}  # miner hotkey -> monotonic
        self._self_hosted_misses: Counter[str] = Counter()  # miner hotkey -> misses in a row
        self._next_self_hosted_listing = 0.0  # monotonic
        self._listing: asyncio.Task | None = None
        # DAH-5716: runs started per queued recheck request.
        # shortcut: in memory, so a restart restarts the budget; the request's TTL still bounds it.
        self._recheck_runs: Counter[str] = Counter()
        # DAH-6013: (executor id, window end) already queued for a recheck, so a window is queued once.
        self._maintenance_end_rechecks: set[tuple[str, datetime]] = set()
        self._rechecks_in_flight: dict[str, str] = {}  # executor id -> miner hotkey
        # Recheck-only runs kept between ticks, so a deferral and the lost-session count outlive one tick.
        self._recheck_pending: dict[str, _Pending] = {}

    def directories_in_use(self) -> set[str]:
        """Job-files directories a running express verification still reads."""
        return {path for path, count in self._directories_in_use.items() if count > 0}

    async def run(self, should_exit: Callable[[], bool]) -> None:
        logger.info(
            _m(
                "[express] Express lane started",
                extra=get_extra_info(
                    {
                        "tick_seconds": settings.express_lane_tick_seconds(),
                        "fast_path": settings.VALIDATION_FAST_PATH_ENABLED,
                        "max_in_flight": settings.EXPRESS_LANE_MAX_IN_FLIGHT,
                        "max_in_flight_per_miner": settings.EXPRESS_LANE_MAX_IN_FLIGHT_PER_MINER,
                    }
                ),
            )
        )
        while not should_exit():
            try:
                await self.tick()
            except Exception as exc:
                logger.error(
                    _m("[express] Tick failed", extra=get_extra_info({"error": str(exc)})),
                    exc_info=True,
                )
            await asyncio.sleep(settings.express_lane_tick_seconds())

    def _hotkey(self) -> str:
        if self._my_hotkey is None:
            self._my_hotkey = settings.get_bittensor_wallet().get_hotkey().ss58_address
        return self._my_hotkey

    async def tick(self) -> int:
        """One pass: discover, select under the caps, launch. Returns how many were launched."""
        inputs = self.cycle_inputs()
        if inputs is None:
            return 0

        self._start_self_hosted_listing()
        snapshot = await self.portal_api.get_all_executors()
        if snapshot is None:
            return 0
        snapshot = self._with_self_hosted(snapshot)

        validated = await self.redis_service.get_validated_executors()
        await self._request_rechecks_at_maintenance_end(inputs.maintenance_ends, snapshot)
        rechecks = await self.redis_service.get_executor_recheck_requests()
        recheck_deadline = (
            inputs.started_at
            + settings.BLOCKS_FOR_JOB * SECONDS_PER_BLOCK
            - RECHECK_MARGIN_SECONDS
        )
        if recheck_deadline - time.monotonic() < RECHECK_MIN_SECONDS:
            rechecks = set()  # left in Redis: they run once the next wave has listed their miners
        chosen = self._select(snapshot, validated, inputs.fleet_known_since, rechecks)
        await self._restore_first_seen()
        for pending in chosen:
            pending.requested = pending.executor.id in rechecks
            if pending.requested:
                pending.recheck_generation = await self.redis_service.get_executor_recheck_generation(
                    pending.executor.id
                )
            pending.recheck_deadline = (
                recheck_deadline if pending.requested and pending.executor.id in validated else None
            )
        if not chosen:
            return 0
        return await self._launch(chosen)

    async def _restore_first_seen(self) -> None:
        """Newly queued executors take their first sighting from Redis (written now when there is none),
        so a restart neither resets first_seen_to_publish_s nor the logged discovery wait."""
        new = [p for p in self._pending.values() if not p.first_seen_restored]
        if not new:
            return
        stored = await self.redis_service.express_first_seen(
            [p.executor.id for p in new], datetime.now(UTC).timestamp()
        )
        for pending in new:
            pending.first_seen_restored = True
            if pending.executor.id in stored:
                pending.first_seen_at = datetime.fromtimestamp(stored[pending.executor.id], UTC)

    async def _request_rechecks_at_maintenance_end(
        self, maintenance_ends: dict[str, datetime], snapshot: dict[str, list[PortalExecutor]]
    ) -> None:
        """DAH-6013: queue a recheck of each node whose maintenance window has ended. The backend keeps such a node
        off the market until a pass that started after the window's end; the cycle's own pass may be up to a cycle
        away. Queued through the same Redis set as a provider's request, so a request that cannot run before the
        next wave waits there."""
        now = datetime.now(UTC)
        my_hotkey = self._hotkey()
        mine = {e.id for executors in snapshot.values() for e in executors if e.validator_hotkey == my_hotkey}
        current = {(executor_id, ends_at) for executor_id, ends_at in maintenance_ends.items() if executor_id in mine}
        self._maintenance_end_rechecks &= current
        for executor_id, ends_at in current - self._maintenance_end_rechecks:
            if ends_at <= now:
                await self.redis_service.request_executor_recheck(executor_id)
                self._maintenance_end_rechecks.add((executor_id, ends_at))

    def _start_self_hosted_listing(self) -> None:
        """Every SELF_HOSTED_LIST_SECONDS, in the background so a slow miner never holds a tick."""
        if (self._listing is not None and not self._listing.done()) or (
            time.monotonic() < self._next_self_hosted_listing
        ):
            return
        self._next_self_hosted_listing = time.monotonic() + SELF_HOSTED_LIST_SECONDS
        self._listing = asyncio.create_task(self._list_self_hosted_miners())

    async def _list_self_hosted_miners(self) -> None:
        try:
            central = self.subtensor_client.hotkeys_routed_via_central_miner
            now = time.monotonic()
            miners = [
                miner
                for miner in await self.subtensor_client.get_miners()
                if miner.axon_info.is_serving
                and miner.hotkey not in central
                and self._self_hosted_backoff.get(miner.hotkey, 0.0) <= now
            ]
            limit = asyncio.Semaphore(SELF_HOSTED_LIST_CONCURRENCY)

            async def list_one(miner: bittensor.NeuronInfo) -> None:
                async with limit:
                    executor_ids = await self.miner_service.list_executor_ids(miner)
                if executor_ids is None:
                    self._self_hosted_misses[miner.hotkey] += 1
                    # From the listing's start, so a one-interval skip ends at the next listing.
                    self._self_hosted_backoff[miner.hotkey] = now + min(
                        SELF_HOSTED_LIST_SECONDS
                        * 2 ** (self._self_hosted_misses[miner.hotkey] - 1),
                        SELF_HOSTED_LIST_FAILURE_BACKOFF_SECONDS,
                    )
                    return
                self._self_hosted_misses.pop(miner.hotkey, None)
                known = self._self_hosted.get(miner.hotkey)
                seen_at = datetime.now(UTC) if known is not None else None
                self._self_hosted[miner.hotkey] = {
                    executor_id: (known or {}).get(executor_id, seen_at)
                    for executor_id in executor_ids
                }

            await asyncio.gather(*(list_one(miner) for miner in miners))
        except Exception as exc:
            logger.error(
                _m(
                    "[express] Self-hosted miner listing failed",
                    extra=get_extra_info({"error": str(exc)}),
                ),
                exc_info=True,
            )

    def _with_self_hosted(
        self, snapshot: dict[str, list[PortalExecutor]]
    ) -> dict[str, list[PortalExecutor]]:
        """The snapshot plus the self-hosted miners' listed nodes it lacks, registered (as far as
        _select goes) when this lane first saw them; None keeps the cycle's ones out."""
        merged = dict(snapshot)
        my_hotkey = self._hotkey()
        for miner_hotkey, listed in self._self_hosted.items():
            in_snapshot = {executor.id for executor in snapshot.get(miner_hotkey, [])}
            merged[miner_hotkey] = [
                *snapshot.get(miner_hotkey, []),
                *(
                    PortalExecutor(id=executor_id, validator_hotkey=my_hotkey, created_at=seen_at)
                    for executor_id, seen_at in listed.items()
                    if executor_id not in in_snapshot
                ),
            ]
        return merged

    def _select(
        self,
        snapshot: dict[str, list[PortalExecutor]],
        validated: set[str],
        fleet_known_since: datetime,
        rechecks: set[str] = frozenset(),
    ) -> list[_Pending]:
        """Track this validator's never-validated executors; return the ones to launch now.

        An executor in rechecks (DAH-5716: its provider asked for a check) is taken even when
        validated before. Oldest registration first, under the global in-flight cap and, for a
        self-hosted miner, the per-miner cap.
        """
        in_flight = self.miner_service.in_flight
        my_hotkey = self._hotkey()
        now = time.monotonic()
        now_wall = datetime.now(UTC)

        present: set[str] = set()
        candidates: list[_Pending] = []
        for miner_hotkey, executors in snapshot.items():
            for executor in executors:
                if executor.validator_hotkey != my_hotkey:
                    continue
                new_to_lane = not (
                    executor.id in validated
                    or executor.id in self._left_to_cycle
                    # Registered well before the first cycle since start (or with no registration
                    # time to go by; RESTART_LOOKBACK covers what a restart cut off): long-known to this validator even if never published since
                    # the flag went on — its miner failed or was offline. The cycle owns it.
                    or executor.created_at is None
                    or executor.created_at < fleet_known_since - RESTART_LOOKBACK
                )
                if not new_to_lane and executor.id not in rechecks:
                    continue
                present.add(executor.id)
                pending = self._pending.get(executor.id) or self._recheck_pending.get(executor.id)
                if pending is None:
                    pending = _Pending(
                        executor=executor,
                        miner_hotkey=miner_hotkey,
                        first_seen_at=now_wall,
                        recheck_only=not new_to_lane,
                    )
                    if new_to_lane:
                        self._pending[executor.id] = pending
                    else:
                        self._recheck_pending[executor.id] = pending
                if executor.id in in_flight or pending.not_before > now:
                    continue
                # The wave has not received this miner's list yet (a cycle started seconds ago):
                # wait a tick, no attempt spent. See MinerService.miners_awaiting_wave_list.
                if miner_hotkey in self.miner_service.miners_awaiting_wave_list:
                    continue
                candidates.append(pending)

        # Removed from the portal (or validated by the cycle) before this lane got to it.
        for executor_id in [e for e in self._pending if e not in present and e not in in_flight]:
            del self._pending[executor_id]
        for executor_id in [e for e in self._recheck_pending if e not in present and e not in in_flight]:
            del self._recheck_pending[executor_id]

        # Oldest registration first; then the two caps that bound a registration flood.
        candidates.sort(key=lambda p: p.executor.created_at or p.first_seen_at)
        express_in_flight = [e for e, lane in in_flight.items() if lane == EXPRESS_LANE]
        room = settings.EXPRESS_LANE_MAX_IN_FLIGHT - len(express_in_flight)
        per_miner: Counter[str] = Counter(
            self._pending[e].miner_hotkey for e in express_in_flight if e in self._pending
        )
        per_miner.update(self._rechecks_in_flight.values())
        # The per-miner cap guards a self-hosted miner's one process. A hotkey routed through the
        # central miner can carry many providers' nodes on a shared service; only the global cap
        # bounds it (5 of the 10 longest first-pass waits on 5 Nov sat behind its 2 slots).
        central = self.subtensor_client.hotkeys_routed_via_central_miner
        chosen: list[_Pending] = []
        for pending in candidates:
            if room <= 0:
                break
            if (
                pending.miner_hotkey not in central
                and per_miner[pending.miner_hotkey] >= settings.EXPRESS_LANE_MAX_IN_FLIGHT_PER_MINER
            ):
                continue
            chosen.append(pending)
            per_miner[pending.miner_hotkey] += 1
            room -= 1
        return chosen

    async def _launch(self, chosen: list[_Pending]) -> int:
        """Start one verification per chosen executor on the cycle's current job files.

        Returns how many were started.
        """
        in_flight = self.miner_service.in_flight
        miners = {miner.hotkey: miner for miner in await self.subtensor_client.get_miners()}
        rented_data = await self.backend_client.get_all_rented_executors()
        if rented_data is None:
            logger.error(
                _m(
                    "[express] Failed to fetch rented executors, skipping this tick",
                    extra=get_extra_info({}),
                )
            )
            return 0

        # Read the inputs again, after the last await: a cycle that started during the awaits
        # above replaced them and removed the earlier job-files directory (nothing held it yet).
        # The validator's prep is synchronous and there is no await between here and the holds
        # below, so the directory these inputs name is the current one and each hold keeps it
        # until its verification ends.
        inputs = self.cycle_inputs()
        if inputs is None:
            return 0
        directory = inputs.encrypted_files.tmp_directory
        if not os.path.isdir(directory):
            # Something outside the validator removed the current cycle's files. A verification
            # without them fails as if the node had; skip the tick instead.
            logger.warning(
                _m(
                    "[express] Job files directory is missing, skipping this tick",
                    extra=get_extra_info({"directory": directory}),
                )
            )
            return 0

        launched = 0
        for pending in chosen:
            if pending.executor.id in in_flight:
                # The wave accepted it during the awaits above; it publishes it at the wave's end.
                continue
            if pending.miner_hotkey in self.miner_service.miners_awaiting_wave_list:
                # A cycle started during the awaits above and its wave has not listed this miner.
                continue
            miner = miners.get(pending.miner_hotkey)
            if miner is None:
                pending.attempts += 1
                self._defer(pending, "miner is not among the serving opted-in miners")
                continue
            in_flight[pending.executor.id] = EXPRESS_LANE
            if pending.recheck_only:
                self._rechecks_in_flight[pending.executor.id] = pending.miner_hotkey
            if pending.requested:
                self._recheck_runs[pending.executor.id] += 1
            self._directories_in_use[directory] += 1
            task = asyncio.create_task(self._verify(pending, miner, inputs, rented_data))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            launched += 1
        return launched

    async def _verify(
        self,
        pending: _Pending,
        miner: bittensor.NeuronInfo,
        inputs: CycleInputs,
        rented_data: RentedExecutorsResponse,
    ) -> None:
        # this task's Redis commands (and its children's) use the express slot, not the wave's queue
        EXPRESS_REDIS_SLOT.set(True)
        executor_id = pending.executor.id
        directory = inputs.encrypted_files.tmp_directory
        # Counted against the node once the outcome is the node's (below); a verification the
        # validator itself spoiled is retried without spending one of MAX_ATTEMPTS.
        attempt = pending.attempts + 1
        started = time.monotonic()
        check_started_at = datetime.now(UTC)
        extra: dict[str, object] = {
            "executor_uuid": executor_id,
            "miner_hotkey": miner.hotkey,
            "attempt": attempt,
        }
        try:
            payload = MinerJobRequestPayload(
                # The cycle whose job files this run uses, even when the next cycle starts
                # before it publishes: that cycle's wave may publish the node under its own id.
                job_batch_id=_cycle_batch_id_or_clock(inputs.job_batch_id),
                miner_hotkey=miner.hotkey,
                miner_coldkey=miner.coldkey,
                miner_address=miner.axon_info.ip,
                miner_port=miner.axon_info.port,
            )
            job = await asyncio.wait_for(
                self.miner_service.request_job_to_miner(
                    payload=payload,
                    encrypted_files=inputs.encrypted_files,
                    rented_data=rented_data,
                    default_docker_image_digests=inputs.default_image_digests,
                    executor_image_snapshot=inputs.executor_image_snapshot,
                    executor_id=executor_id,
                    # Never scored, so the pipeline right-sizes its probes (DAH-3011).
                    first_pass=True,
                ),
                timeout=settings.JOB_TIME_OUT
                if pending.recheck_deadline is None
                else min(settings.JOB_TIME_OUT, max(0.0, pending.recheck_deadline - started)),
            )
            if not os.path.isdir(directory):
                # The hold keeps this directory across cycles, so it went away by another hand.
                # The pipeline reports a missing upload (UploadFilesCheck, the scrape's binary
                # fallback) as the node's failure; the node did nothing. Discard, retry later.
                logger.warning(
                    _m(
                        "[express] Job files were removed during the verification, result discarded",
                        extra=get_extra_info({**extra, "directory": directory}),
                    )
                )
                pending.not_before = time.monotonic() + RETRY_SECONDS
                return
            pending.attempts = attempt
            # Only this executor's own result: a miner-level failure comes back under the
            # failed-miner sentinel uuid and a manual rental elsewhere on the miner is not ours.
            results = [
                result
                for result in (job or {}).get("results", [])
                if result.executor_info.uuid == executor_id
            ]
            if not results:
                self._defer(pending, MINER_DID_NOT_RETURN_EXECUTOR)
                return
            for result in results:
                # published under the cycle's id; the backend reads this against a maintenance window's end
                result.check_started_at = check_started_at
            await self._publish(pending, miner, results, extra, started, payload.job_batch_id)
        except Exception as exc:
            pending.attempts = attempt
            logger.error(
                _m(
                    "[express] Verification failed before a result could be published",
                    extra=get_extra_info({**extra, "error": str(exc)}),
                ),
                exc_info=True,
            )
            self._defer(pending, str(exc))
        finally:
            # The budget's last run ended, whatever its outcome: so does the request. Before the
            # in-flight entry goes, so no tick starts a further run in between.
            if pending.requested and self._recheck_runs.get(executor_id, 0) >= RECHECK_MAX_RUNS:
                with suppress(Exception):
                    await self._end_recheck(pending)
            if self.miner_service.in_flight.get(executor_id) == EXPRESS_LANE:
                del self.miner_service.in_flight[executor_id]
            self._rechecks_in_flight.pop(executor_id, None)
            self._directories_in_use[directory] -= 1
            if self._directories_in_use[directory] <= 0:
                del self._directories_in_use[directory]

    async def _publish(
        self,
        pending: _Pending,
        miner: bittensor.NeuronInfo,
        results: list[JobResult],
        extra: dict[str, object],
        started: float,
        job_batch_id: str,
    ) -> None:
        """Publish the executor's own result spec-only, record it as validated, log the metric.

        Published as the pipeline produced it — no incentive, scored_at None — so the backend
        creates/updates the executor row (AVAILABLE when the score is positive) but writes no
        incentive_per_validator_cycle row: is_provider_emission_cycle_eligible needs scored_at.
        The next scored cycle overwrites the row as today.
        """
        executor_id = pending.executor.id
        result = results[0]
        if not (result.score > 0 or result.job_score > 0) or result.session_lost_in_port_check:
            # An earlier pass in this batch must not keep the fallback off (review 4558893416).
            self.miner_service.express_published.pop(executor_id.lower(), None)
            self.miner_service.express_published_miner.pop(executor_id.lower(), None)
        if result.session_lost_in_port_check:
            # The run has no verdict: its findings still go out, and it is retried without spending one of
            # MAX_ATTEMPTS (bounded by SESSION_LOST_MAX_RUNS), neither a failed first pass nor validated.
            if result.availability_errors or result.pod_states:
                await self.miner_service.publish_machine_specs(
                    results, miner.hotkey, miner.coldkey, is_whole_miner_batch=False
                )
            pending.attempts -= 1  # _verify counted this run
            pending.lost_sessions += 1
            if pending.requested:
                # review 4491025356: a lost session spends none of the recheck's runs; its own bound ends it
                self._recheck_runs[executor_id] -= 1
                if pending.lost_sessions >= SESSION_LOST_MAX_RUNS:
                    with suppress(Exception):
                        await self._end_recheck(pending)
            self._defer(pending, SESSION_LOST_IN_PORT_CHECK)
            return
        passed = result.score > 0 or result.job_score > 0
        should_run_first_pass_again = (
            not passed
            and settings.EXPRESS_LANE_FAILED_FIRST_PASS_RETRY_ENABLED
            and pending.failed_runs + 1 < FIRST_PASS_MAX_RUNS
        )
        # One node under the cycle's id: never the miner's batch for that id, the wave's is. Its list is whole, so
        # the outage restore starts from it (review 4510532986: a recheck that cleared a hold).
        await self.miner_service.publish_machine_specs(
            results, miner.hotkey, miner.coldkey, is_whole_miner_batch=False, remember_availability=True
        )
        if passed:
            # Only a pass has the pod check billing needs; a failed or lost run leaves the cycle's
            # result-missing fallback in place (review 4558671509).
            self.miner_service.express_published[executor_id.lower()] = job_batch_id
            self.miner_service.express_published_miner[executor_id.lower()] = miner.hotkey
        pending.recheck_deadline = None
        if not should_run_first_pass_again:
            # A failed first pass that runs again keeps the request: the next tick selects it.
            with suppress(Exception):
                await self._end_recheck(pending)
        if should_run_first_pass_again:
            # Not recorded as validated, so the lane runs it again; a wave that lists its miner
            # once the retry is due takes it instead (_LaneOwnedNodes) and its publish records it.
            pending.failed_runs += 1
            pending.not_before = time.monotonic() + FIRST_PASS_FAILED_RETRY_SECONDS
            self._log_published(pending, result, passed, extra, started)
            logger.info(
                _m(
                    "[express] Executor not verified yet, will retry",
                    extra=get_extra_info(
                        {
                            **extra,
                            "reason": FIRST_PASS_FAILED,
                            "failed_runs": pending.failed_runs,
                            "retry_seconds": FIRST_PASS_FAILED_RETRY_SECONDS,
                        }
                    ),
                )
            )
            return
        if not passed:
            # DAH-5723: attempts used up, the cycle owns it. Not recorded as validated, or
            # publish_new_executors_at_pass would skip its wave pass and it would wait for the
            # cycle's end (6-7 min); the cycle's publish records it.
            self._left_to_cycle.add(executor_id)
            self._pending.pop(executor_id, None)
            self._log_published(pending, result, passed, extra, started)
            return
        try:
            await self.redis_service.mark_executors_validated([executor_id])
        except Exception as exc:
            # The result is published; a Redis blip must not run the node again. The cycle
            # owns it from here (its next seed records it), like a node that used up its
            # attempts.
            self._left_to_cycle.add(executor_id)
            logger.error(
                _m(
                    "[express] Published, but could not record the executor as validated; left to the cycle",
                    extra=get_extra_info({**extra, "error": str(exc)}),
                ),
            )
        self._pending.pop(executor_id, None)
        self._log_published(pending, result, passed, extra, started)

    async def _end_recheck(self, pending: _Pending) -> None:
        if not pending.requested:
            return  # review 4523252174: this run selected no request, so one that arrived during it stays
        executor_id = pending.executor.id
        # Redis first: local state goes only once the request is gone or a newer one is confirmed (review 4523252183)
        await self.redis_service.pop_executor_recheck_request(executor_id, pending.recheck_generation)
        self._recheck_runs.pop(executor_id, None)
        self._recheck_pending.pop(executor_id, None)

    def _log_published(
        self,
        pending: _Pending,
        result: JobResult,
        passed: bool,
        extra: dict[str, object],
        started: float,
    ) -> None:
        published_at = datetime.now(UTC)
        registered_at = pending.executor.created_at
        logger.info(
            _m(
                EXPRESS_PUBLISHED_EVENT if passed else EXPRESS_FAILED_EVENT,
                extra=get_extra_info(
                    {
                        **extra,
                        "outcome": "passed" if passed else "failed",
                        "score": result.score,
                        "log_status": result.log_status,
                        "registered_at": registered_at.isoformat() if registered_at else None,
                        "first_seen_at": pending.first_seen_at.isoformat(),
                        "published_at": published_at.isoformat(),
                        "registration_to_publish_s": round(
                            (published_at - registered_at).total_seconds(), 1
                        )
                        if registered_at
                        else None,
                        "first_seen_to_publish_s": round(
                            (published_at - pending.first_seen_at).total_seconds(), 1
                        ),
                        "verification_s": round(time.monotonic() - started, 1),
                    }
                ),
            )
        )

    def _defer(self, pending: _Pending, reason: str) -> None:
        """Try again after retry_seconds_for(reason), or after max_attempts_for(reason) asks leave
        the executor to the cycle.

        The caller has already counted the attempt.
        """
        extra = {
            "executor_uuid": pending.executor.id,
            "miner_hotkey": pending.miner_hotkey,
            "attempt": pending.attempts,
            "reason": reason,
        }
        if (
            pending.lost_sessions >= SESSION_LOST_MAX_RUNS
            if reason == SESSION_LOST_IN_PORT_CHECK
            else pending.attempts >= max_attempts_for(reason)
        ):
            self._left_to_cycle.add(pending.executor.id)
            self._pending.pop(pending.executor.id, None)
            logger.warning(
                _m("[express] Executor left to the normal cycle", extra=get_extra_info(extra))
            )
            return
        retry_seconds = retry_seconds_for(reason)
        pending.not_before = time.monotonic() + retry_seconds
        logger.info(
            _m(
                "[express] Executor not verified yet, will retry",
                extra=get_extra_info({**extra, "retry_seconds": retry_seconds}),
            )
        )


def retry_seconds_for(reason: str) -> int:
    """How long a deferred executor waits before the lane asks its miner again.

    Validation fast path: a miner that did not list the node yet is asked again once its portal
    snapshot has had time to refresh (EXPRESS_LANE_MINER_SNAPSHOT_RETRY_SECONDS, default 35 s >
    the central miner's 30-s TTL) instead of after RETRY_SECONDS, and max_attempts_for gives that
    reason more asks so the window the lane covers stays at least the serial one (the miner serves
    a stale snapshot while its portal refresh fails, and a blip longer than the window sends the
    node to the wave). Every other reason, and the flag off, keep RETRY_SECONDS and MAX_ATTEMPTS.
    """
    if settings.VALIDATION_FAST_PATH_ENABLED and reason == MINER_DID_NOT_RETURN_EXECUTOR:
        return settings.EXPRESS_LANE_MINER_SNAPSHOT_RETRY_SECONDS
    return RETRY_SECONDS


def max_attempts_for(reason: str) -> int:
    """How many asks a deferred executor gets before the lane leaves it to the cycle (see retry_seconds_for)."""
    if settings.VALIDATION_FAST_PATH_ENABLED and reason == MINER_DID_NOT_RETURN_EXECUTOR:
        return max(MAX_ATTEMPTS, settings.EXPRESS_LANE_MINER_SNAPSHOT_MAX_ATTEMPTS)
    return MAX_ATTEMPTS
