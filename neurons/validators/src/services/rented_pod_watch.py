"""DAH-5215: a rented pod whose SSH stops answering shows unreachable within about two minutes.

The cycle probes every rented pod once (services/pod_ssh_probe.py), so a host that drops between
two cycles leaves its pods RUNNING for up to 15 min (renter feedback F-1640: SSH dead at 17:56Z,
caught by the 18:03Z cycle). This watch runs beside Validator.sync() and, every
RENTED_POD_WATCH_INTERVAL_SECONDS, runs the same ``probe_rented_pods`` on the backend's light rented
list. Only a pod that has sent a banner since this process started counts misses (a pod without sshd
never alarms). A RUNNING pod that misses RENTED_POD_WATCH_MISSES probes in a row (``refused`` or
``timeout``) is reported ``unreachable`` (since = its first missed probe) every tick while it stays
down, and ``reachable`` once when its banner comes back. A ``no_banner`` is re-probed once with
RENTED_POD_WATCH_NO_BANNER_RETRY_TIMEOUT_SECONDS first and needs RENTED_POD_WATCH_NO_BANNER_MISSES.
A tick whose fleet is not ok is our own outage and changes nothing.

Observe only: nothing here touches a score, a verdict, a cycle result or the per-cycle probe. State
is in memory; after a restart the first tick sends ``reachable`` for every answering pod, so a marker
set before the restart is cleared.
"""

from __future__ import annotations

import asyncio
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


class RentedPodWatch:
    def __init__(self, *, backend_client, probe=probe_rented_pods):
        self.backend_client = backend_client
        self.probe = probe
        self._pods: dict[str, _PodState] = {}
        self._route_missing_logged = False

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
        watched = await self._retry_no_banner(rented, watched)

        reports = []  # (state, report coroutine)
        misses = 0
        for observation in watched:
            state = self._pods.setdefault(observation.pod_id, _PodState())
            if observation.result is PodSshResult.BANNER:
                state.posted = state.posted or not state.answered
                state.answered, state.misses, state.first_miss_at = True, 0, None
                state.announced = False
                if state.posted:
                    reports.append(
                        (state, self._post(observation, REACHABLE, probed_at, probed_at))
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
                    (None, self._post(observation, UNREACHABLE, state.first_miss_at, probed_at))
                )

        failed = await self._send(reports)
        logger.info(
            _m(
                "[pod-watch] rented pod SSH watch tick",
                extra=get_extra_info(
                    {
                        "pods": len(watched),
                        "misses": misses,
                        "posted": len(reports),
                        "failed": failed,
                    }
                ),
            )
        )

    async def _retry_no_banner(
        self, rented, watched: list[PodSshObservation]
    ) -> list[PodSshObservation]:
        """A no_banner from a pod that answered before gets one more probe with the long timeout."""
        retry = {
            o.pod_id
            for o in watched
            if o.result is PodSshResult.NO_BANNER and self._pods.get(o.pod_id, _PodState()).answered
        }
        if not retry:
            return watched
        subset = rented.model_copy(
            update={
                "executors": {
                    uuid: executor.model_copy(
                        update={"pods": [p for p in executor.pods if p.pod_id in retry]}
                    )
                    for uuid, executor in rented.executors.items()
                    if any(p.pod_id in retry for p in executor.pods)
                }
            }
        )
        again = await self.probe(
            subset,
            timeout=settings.RENTED_POD_WATCH_NO_BANNER_RETRY_TIMEOUT_SECONDS,
            concurrency=settings.RENTED_POD_SSH_PROBE_CONCURRENCY,
            job_batch_id="pod-watch-retry",
        )
        second = {o.pod_id: o for node in again.values() for o in node}
        return [second.get(o.pod_id, o) if o.pod_id in retry else o for o in watched]

    async def _send(self, reports) -> int:
        """Posts every report at once, within REPORTS_DEADLINE_SECONDS; returns how many did not land.
        A reachable that did not land stays owed and goes again next tick."""
        if not reports:
            return 0
        tasks = [asyncio.ensure_future(coro) for _, coro in reports]
        try:
            done, _ = await asyncio.wait(tasks, timeout=REPORTS_DEADLINE_SECONDS)
        finally:
            for task in tasks:
                task.cancel()  # no-op on the finished ones; also when the tick deadline cancels us
        failed = 0
        for (state, _), task in zip(reports, tasks, strict=True):
            ok = (
                task in done and not task.cancelled() and task.exception() is None and task.result()
            )
            failed += not ok
            if ok and state is not None:
                state.posted = False
        return failed

    async def _post(
        self, observation: PodSshObservation, state: str, since: datetime, observed_at: datetime
    ) -> bool:
        """True when the backend answered; a missing route (older backend, 404) counts as answered."""
        down = state == UNREACHABLE
        try:
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
