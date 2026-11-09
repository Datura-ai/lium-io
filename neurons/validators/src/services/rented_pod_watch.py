"""DAH-5215: a rented pod whose SSH stops answering shows unreachable within about two minutes.

The cycle probes every rented pod once (services/pod_ssh_probe.py), so a host that drops between
two cycles leaves its pods RUNNING for up to 15 min (renter feedback F-1640: SSH dead at 17:56Z,
caught by the 18:03Z cycle). This watch runs beside Validator.sync() and, every
RENTED_POD_WATCH_INTERVAL_SECONDS, runs the same ``probe_rented_pods`` on the backend's light rented
list. Only a pod that has sent a banner since this process started counts misses (a pod without sshd
never alarms). A RUNNING pod that misses RENTED_POD_WATCH_MISSES probes in a row (``refused`` or
``timeout``) is reported ``unreachable`` (since = its first missed probe) every tick while it stays
down, and ``reachable`` once when its banner comes back. A ``no_banner`` is re-probed once with
RENTED_POD_WATCH_NO_BANNER_RETRY_TIMEOUT_SECONDS first and needs RENTED_POD_WATCH_NO_BANNER_MISSES;
the other outcomes are counted and reported while those retries run, and a retry that no longer fits
in the tick waits for the next one.
A tick whose fleet is not ok is our own outage and changes nothing.

Observe only: nothing here touches a score, a verdict, a cycle result or the per-cycle probe. State
is in memory; after a restart the first tick sends ``reachable`` for every answering pod, so a marker
set before the restart is cleared.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from clients.backend_client import BackendRejected
from protocol.vc_protocol.validator_requests import PodSshObservation, PodSshResult
from services.pod_ssh_probe import POD_STATUS_RUNNING, probe_rented_pods

from core.config import settings
from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

UNREACHABLE = "unreachable"
REACHABLE = "reachable"


# all of a tick's reports must land within this, so a slow backend never stretches the minute
REPORTS_DEADLINE_SECONDS = 30
# kept free at the end of a tick for the reports of the no_banner retries
RETRY_REPORTS_RESERVE_SECONDS = 5
# reports in flight at once, below the backend client's 10 connections per host, so a report's
# request timeout starts when it is sent, never while it queues for a connection
REPORT_CONCURRENCY = 5


@dataclass
class _PodState:
    # a banner arrived since this process started; misses count only after it
    answered: bool = False
    misses: int = 0
    first_miss_at: datetime | None = None
    # a reachable is owed: an unreachable was posted (or it is the pod's first banner), none landed
    posted: bool = False
    # an unreachable was posted and no banner since: repost on every miss, whatever its threshold
    announced: bool = False
    # when its no_banner was last re-probed: retries left out of a tick go first in the next one
    retried_at: datetime | None = None
    # when a report of it last landed: reports cut at a deadline go first in the next tick
    reported_at: datetime | None = None


def _ok(task: asyncio.Task) -> bool:
    return not task.cancelled() and task.exception() is None and bool(task.result())


def _landed(state: _PodState, reachable: bool, task: asyncio.Task) -> None:
    if _ok(task):
        state.reported_at = datetime.now(UTC)
        state.posted = state.posted and not reachable


class RentedPodWatch:
    def __init__(self, *, backend_client, probe=probe_rented_pods):
        self.backend_client = backend_client
        self.probe = probe
        self._pods: dict[str, _PodState] = {}
        self._route_missing_logged = False
        self._send_slots = asyncio.Semaphore(REPORT_CONCURRENCY)

    async def run(self, should_exit: Callable[[], bool]) -> None:
        loop = asyncio.get_running_loop()
        interval = settings.RENTED_POD_WATCH_INTERVAL_SECONDS
        while not should_exit():
            started = loop.time()
            try:
                # the whole tick (list, probes, reports) fits in one interval; the next starts from this start
                await asyncio.wait_for(self.tick(), timeout=interval)
            except Exception as exc:
                logger.error(
                    _m("[pod-watch] Tick failed", extra=get_extra_info({"error": repr(exc)})),
                    exc_info=True,
                )
            await asyncio.sleep(max(0.0, started + interval - loop.time()))

    async def tick(self) -> None:
        started = asyncio.get_running_loop().time()
        rented = await self.backend_client.get_rented_pod_targets()
        if rented is None:
            return  # no list is no news: keep every counter
        listed = {
            pod.pod_id: pod.status
            for executor in rented.executors.values()
            for pod in executor.pods
        }
        running = {pod_id for pod_id, status in listed.items() if status == POD_STATUS_RUNNING}
        for pod_id in [p for p in self._pods if p not in running]:
            if pod_id in listed:
                # a reboot is a new container, but an owed reachable must still follow it
                self._pods[pod_id] = _PodState(posted=self._pods[pod_id].posted)
            else:
                del self._pods[pod_id]

        probed_at = datetime.now(UTC)
        observations = await self.probe(
            rented,
            timeout=settings.RENTED_POD_SSH_PROBE_TIMEOUT_SECONDS,
            concurrency=settings.RENTED_POD_SSH_PROBE_CONCURRENCY,
            job_batch_id="pod-watch",
        )
        watched = [
            observation
            for node in observations.values()
            for observation in node
            if observation.pod_id in running
        ]
        if not all(observation.fleet_ok for observation in watched):
            return  # our own outage: says nothing about the pods
        loop = asyncio.get_running_loop()
        # the tick's budget: run() cancels a tick that outlives the interval, with all its results
        ends = started + settings.RENTED_POD_WATCH_INTERVAL_SECONDS
        retry = {
            o.pod_id
            for o in watched
            if o.result is PodSshResult.NO_BANNER and self._pods.get(o.pod_id, _PodState()).answered
        }
        # the short outcomes are counted and reported while the long no_banner retries run
        no_banner = [o for o in watched if o.pod_id in retry]
        retrying = asyncio.ensure_future(
            self._retry_no_banner(rented, no_banner, ends, probed_at)
        )
        try:
            short = [o for o in watched if o.pod_id not in retry]
            reports, misses = self._observe(short, probed_at)
            failed = await self._send(reports)
            retried = await retrying
        finally:
            retrying.cancel()
        late, late_misses = self._observe(retried, probed_at)
        failed += await self._send(late, max(0.0, ends - loop.time()))
        logger.info(
            _m(
                "[pod-watch] rented pod SSH watch tick",
                extra=get_extra_info(
                    {
                        "pods": len(watched),
                        "misses": misses + late_misses,
                        "posted": len(reports) + len(late),
                        "failed": failed,
                        "retry_deferred": len(retry) - len(retried),
                    }
                ),
            )
        )

    def _observe(self, observations: list[PodSshObservation], probed_at: datetime):
        """Counts each observation into its pod's state; returns the reports owed and the misses."""
        reports = []  # (state, is it a reachable, report coroutine)
        misses = 0
        for observation in observations:
            state = self._pods.setdefault(observation.pod_id, _PodState())
            if observation.result is PodSshResult.BANNER:
                state.posted = state.posted or not state.answered
                state.answered, state.misses, state.first_miss_at = True, 0, None
                state.announced = False
                if state.posted:
                    reports.append(
                        (state, True, self._post(observation, REACHABLE, probed_at, probed_at))
                    )
                continue
            if not state.answered:
                continue  # never answered SSH (no sshd, an HTTP-only pod): not ours to alarm on
            misses += 1
            state.misses += 1
            state.first_miss_at = state.first_miss_at or probed_at
            needed = (
                settings.RENTED_POD_WATCH_NO_BANNER_MISSES
                if observation.result is PodSshResult.NO_BANNER
                else settings.RENTED_POD_WATCH_MISSES
            )
            if state.announced or state.misses >= needed:
                state.announced = state.posted = True
                reports.append(
                    (
                        state,
                        False,
                        self._post(observation, UNREACHABLE, state.first_miss_at, probed_at),
                    )
                )
        return reports, misses

    async def _retry_no_banner(
        self, rented, first: list[PodSshObservation], ends: float, probed_at: datetime
    ) -> list[PodSshObservation]:
        """Re-probes each no_banner once with the long timeout, one probe batch at a time, while a
        whole batch still fits before ``ends`` less RETRY_REPORTS_RESERVE_SECONDS. A pod left out
        changes nothing this tick and goes first next tick (least recently retried first)."""
        loop = asyncio.get_running_loop()
        timeout = settings.RENTED_POD_WATCH_NO_BANNER_RETRY_TIMEOUT_SECONDS
        batch_size = settings.RENTED_POD_SSH_PROBE_CONCURRENCY
        never = datetime.min.replace(tzinfo=UTC)
        first.sort(key=lambda o: self._pods[o.pod_id].retried_at or never)
        retried: list[PodSshObservation] = []
        for i in range(0, len(first), batch_size):
            if loop.time() + timeout > ends - RETRY_REPORTS_RESERVE_SECONDS:
                break
            batch = first[i : i + batch_size]
            ids = {o.pod_id for o in batch}
            subset = rented.model_copy(
                update={
                    "executors": {
                        uuid: executor.model_copy(
                            update={"pods": [p for p in executor.pods if p.pod_id in ids]}
                        )
                        for uuid, executor in rented.executors.items()
                        if any(p.pod_id in ids for p in executor.pods)
                    }
                }
            )
            again = await self.probe(
                subset, timeout=timeout, concurrency=batch_size, job_batch_id="pod-watch-retry"
            )
            second = {o.pod_id: o for node in again.values() for o in node}
            for o in batch:
                self._pods[o.pod_id].retried_at = probed_at
                retried.append(second.get(o.pod_id, o))
        return retried

    async def _send(self, reports, deadline: float = REPORTS_DEADLINE_SECONDS) -> int:
        """Posts the reports, ``REPORT_CONCURRENCY`` at a time and least recently landed first,
        within ``deadline`` seconds; returns how many did not land. A reachable that did not land
        stays owed and goes again next tick."""
        if not reports:
            return 0
        never = datetime.min.replace(tzinfo=UTC)
        reports = sorted(reports, key=lambda report: report[0].reported_at or never)
        tasks = []
        for state, reachable, coro in reports:
            task = asyncio.ensure_future(coro)
            # recorded as each lands, so a tick cancelled at its own deadline keeps what got through
            task.add_done_callback(functools.partial(_landed, state, reachable))
            tasks.append(task)
        try:
            done, _ = await asyncio.wait(tasks, timeout=min(deadline, REPORTS_DEADLINE_SECONDS))
        finally:
            for task in tasks:
                task.cancel()  # no-op on the finished ones; also when the tick deadline cancels us
        return sum(not (task in done and _ok(task)) for task in tasks)

    async def _post(
        self, observation: PodSshObservation, state: str, since: datetime, observed_at: datetime
    ) -> bool:
        """True when the backend answered; a missing route (older backend, 404) counts as answered."""
        down = state == UNREACHABLE
        try:
            async with self._send_slots:
                response = await self.backend_client.report_pod_ssh_reachability(
                    observation.pod_id,
                    state=state,
                    since=since.isoformat(),
                    result=observation.result.value if down else None,
                    errno=observation.errno if down else None,
                    observed_at=observed_at.isoformat(),
                    answered_before=True if down else None,  # misses count only after a banner
                )
        except BackendRejected as exc:
            if exc.status != 404:
                return False  # only 404 means "no route"; any other rejection is sent again
            if not self._route_missing_logged:
                self._route_missing_logged = True
                logger.info(
                    _m(
                        "[pod-watch] backend has no ssh-reachability route yet; reports are dropped",
                        extra=get_extra_info({"error": str(exc)}),
                    )
                )
            return True
        return response is not None
