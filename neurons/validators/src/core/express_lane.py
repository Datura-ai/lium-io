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
the executor row but writes no incentive ledger row). The next scored cycle overwrites it as
today. Off by default (settings.EXPRESS_LANE_ENABLED).
"""

import asyncio
import logging
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from clients.validator_portal_api import PortalExecutor, ValidatorPortalAPI
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload
from services.executor_image_policy import ExpectedImageSnapshot
from services.miner_service import EXPRESS_LANE, MinerService

from core.config import settings
from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

# One line per express verification; its registration_to_publish_s is the deploy metric.
EXPRESS_PUBLISHED_EVENT = "[express] Executor verified and published ahead of the cycle"
# The miner did not return the executor (its portal snapshot is not refreshed yet, or the node is
# unreachable): try again later, a bounded number of times, then leave it to the normal cycle.
MAX_ATTEMPTS = 3
RETRY_SECONDS = 120
# job_batch_id format the backend parses into the prod_executors row's time.
JOB_BATCH_ID_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class CycleInputs:
    """What the cycle prepared for its wave. The express lane reuses it verbatim, so an express
    verification is the cycle's pipeline with the cycle's job files, digests and image snapshot."""

    encrypted_files: MinerJobEnryptedFiles
    default_image_digests: dict[str, str]
    executor_image_snapshot: ExpectedImageSnapshot | None
    # When the first cycle since this process started began. That cycle asked every serving miner
    # for its executors, so an executor registered before it is long-known even when its miner
    # failed or was offline that cycle and it never reached the validated set; only executors
    # registered after this instant can be new to this validator.
    fleet_known_since: datetime


@dataclass
class _Pending:
    executor: PortalExecutor
    miner_hotkey: str
    first_seen_at: datetime
    attempts: int = 0
    not_before: float = 0.0  # monotonic


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
        self._tasks: set[asyncio.Task] = set()
        self._my_hotkey: str | None = None
        # Given up on this process's watch (MAX_ATTEMPTS without the miner returning them); the
        # normal cycle owns them from here. In memory on purpose: a restart may try again.
        self._left_to_cycle: set[str] = set()
        # Job-files directory -> running verifications reading it. The validator keeps these
        # directories when it prepares the next cycle's files (FileEncryptService).
        self._directories_in_use: Counter[str] = Counter()

    def directories_in_use(self) -> set[str]:
        """Job-files directories a running express verification still reads."""
        return {path for path, count in self._directories_in_use.items() if count > 0}

    async def run(self, should_exit: Callable[[], bool]) -> None:
        logger.info(
            _m(
                "[express] Express lane started",
                extra=get_extra_info(
                    {
                        "tick_seconds": settings.EXPRESS_LANE_TICK_SECONDS,
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
            await asyncio.sleep(settings.EXPRESS_LANE_TICK_SECONDS)

    def _hotkey(self) -> str:
        if self._my_hotkey is None:
            self._my_hotkey = settings.get_bittensor_wallet().get_hotkey().ss58_address
        return self._my_hotkey

    async def tick(self) -> int:
        """One pass: discover, select under the caps, launch. Returns how many were launched."""
        inputs = self.cycle_inputs()
        if inputs is None:
            return 0
        fleet_known_since = inputs.fleet_known_since

        snapshot = await self.portal_api.get_all_executors()
        if snapshot is None:
            return 0

        validated = await self.redis_service.get_validated_executors()
        in_flight = self.miner_service.in_flight
        my_hotkey = self._hotkey()
        now = time.monotonic()
        now_wall = datetime.now(UTC)

        present: set[str] = set()
        candidates: list[_Pending] = []
        for miner_hotkey, executors in snapshot.items():
            for executor in executors:
                if (
                    executor.validator_hotkey != my_hotkey
                    or executor.id in validated
                    or executor.id in self._left_to_cycle
                    # Registered before the first cycle since start (or with no registration
                    # time to go by): long-known to this validator even if never published since
                    # the flag went on — its miner failed or was offline. The cycle owns it.
                    or executor.created_at is None
                    or executor.created_at < fleet_known_since
                ):
                    continue
                present.add(executor.id)
                pending = self._pending.get(executor.id)
                if pending is None:
                    pending = self._pending[executor.id] = _Pending(
                        executor=executor, miner_hotkey=miner_hotkey, first_seen_at=now_wall
                    )
                if executor.id in in_flight or pending.not_before > now:
                    continue
                candidates.append(pending)

        # Removed from the portal (or validated by the cycle) before this lane got to it.
        for executor_id in [e for e in self._pending if e not in present and e not in in_flight]:
            del self._pending[executor_id]

        if not candidates:
            return 0

        # Oldest registration first; then the two caps that bound a registration flood.
        candidates.sort(key=lambda p: p.executor.created_at or p.first_seen_at)
        express_in_flight = [e for e, lane in in_flight.items() if lane == EXPRESS_LANE]
        room = settings.EXPRESS_LANE_MAX_IN_FLIGHT - len(express_in_flight)
        per_miner: Counter[str] = Counter(
            self._pending[e].miner_hotkey for e in express_in_flight if e in self._pending
        )
        chosen: list[_Pending] = []
        for pending in candidates:
            if room <= 0:
                break
            if per_miner[pending.miner_hotkey] >= settings.EXPRESS_LANE_MAX_IN_FLIGHT_PER_MINER:
                continue
            chosen.append(pending)
            per_miner[pending.miner_hotkey] += 1
            room -= 1
        if not chosen:
            return 0

        miners = {miner.hotkey: miner for miner in await self.subtensor_client.get_miners()}
        rented_data = await self.backend_client.get_all_rented_executors()
        if rented_data is None:
            logger.error(
                _m("[express] Failed to fetch rented executors, skipping this tick", extra=get_extra_info({}))
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
            miner = miners.get(pending.miner_hotkey)
            if miner is None:
                pending.attempts += 1
                self._defer(pending, "miner is not among the serving opted-in miners")
                continue
            in_flight[pending.executor.id] = EXPRESS_LANE
            self._directories_in_use[directory] += 1
            task = asyncio.create_task(self._verify(pending, miner, inputs, rented_data))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            launched += 1
        return launched

    async def _verify(self, pending: _Pending, miner, inputs: CycleInputs, rented_data) -> None:
        executor_id = pending.executor.id
        directory = inputs.encrypted_files.tmp_directory
        # Counted against the node once the outcome is the node's (below); a verification the
        # validator itself spoiled is retried without spending one of MAX_ATTEMPTS.
        attempt = pending.attempts + 1
        started_wall = datetime.now(UTC)
        started = time.monotonic()
        extra = {
            "executor_uuid": executor_id,
            "miner_hotkey": miner.hotkey,
            "attempt": attempt,
        }
        try:
            payload = MinerJobRequestPayload(
                job_batch_id=started_wall.strftime(JOB_BATCH_ID_FORMAT),
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
                    # Never scored, so the pipeline may right-size its probes (DAH-3011) — it
                    # does only with FIRST_PASS_FAST_PATH_ENABLED on.
                    first_pass=True,
                ),
                timeout=settings.JOB_TIME_OUT,
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
                self._defer(pending, "miner did not return the executor")
                return

            # Published as the pipeline produced it — no incentive, scored_at None — so the
            # backend creates/updates the executor row (AVAILABLE when the score is positive)
            # but writes no incentive_per_validator_cycle row: is_provider_emission_cycle_eligible
            # needs scored_at. The next scored cycle overwrites the row as today.
            await self.miner_service.publish_machine_specs(results, miner.hotkey, miner.coldkey)
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

            published_at = datetime.now(UTC)
            result = results[0]
            registered_at = pending.executor.created_at
            logger.info(
                _m(
                    EXPRESS_PUBLISHED_EVENT,
                    extra=get_extra_info(
                        {
                            **extra,
                            "outcome": "passed" if (result.score > 0 or result.job_score > 0) else "failed",
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
            if self.miner_service.in_flight.get(executor_id) == EXPRESS_LANE:
                del self.miner_service.in_flight[executor_id]
            self._directories_in_use[directory] -= 1
            if self._directories_in_use[directory] <= 0:
                del self._directories_in_use[directory]

    def _defer(self, pending: _Pending, reason: str) -> None:
        """Try again after RETRY_SECONDS, or after MAX_ATTEMPTS leave the executor to the cycle.

        The caller has already counted the attempt.
        """
        extra = {
            "executor_uuid": pending.executor.id,
            "miner_hotkey": pending.miner_hotkey,
            "attempt": pending.attempts,
            "reason": reason,
        }
        if pending.attempts >= MAX_ATTEMPTS:
            self._left_to_cycle.add(pending.executor.id)
            self._pending.pop(pending.executor.id, None)
            logger.warning(_m("[express] Executor left to the normal cycle", extra=get_extra_info(extra)))
            return
        pending.not_before = time.monotonic() + RETRY_SECONDS
        logger.info(_m("[express] Executor not verified yet, will retry", extra=get_extra_info(extra)))
