"""DAH-5215: a rented pod whose SSH stops answering shows unreachable within about two minutes.

The cycle probes every rented pod once (services/pod_ssh_probe.py), so a host that drops between
two cycles leaves its pods RUNNING for up to 15 min (renter feedback F-1640: SSH dead at 17:56Z,
caught by the 18:03Z cycle). This watch runs beside Validator.sync() and, every
RENTED_POD_WATCH_INTERVAL_SECONDS, runs the same ``probe_rented_pods`` on the backend's rented list.
A RUNNING pod that misses RENTED_POD_WATCH_MISSES probes in a row is reported
``unreachable`` (since = its first missed probe) every tick while it stays down, and ``reachable``
once when its banner comes back. A tick whose fleet is not ok is our own outage and changes nothing.

Observe only: nothing here touches a score, a verdict, a cycle result or the per-cycle probe. State
is in memory on purpose; a restart starts counting again.
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


@dataclass
class _PodState:
    misses: int = 0
    first_miss_at: datetime | None = None
    # an unreachable was posted and no reachable has been delivered since
    posted: bool = False


class RentedPodWatch:
    def __init__(self, *, backend_client, probe=probe_rented_pods):
        self.backend_client = backend_client
        self.probe = probe
        self._pods: dict[str, _PodState] = {}
        self._route_missing_logged = False

    async def run(self, should_exit: Callable[[], bool]) -> None:
        while not should_exit():
            try:
                await self.tick()
            except Exception as exc:
                logger.error(
                    _m("[pod-watch] Tick failed", extra=get_extra_info({"error": str(exc)})),
                    exc_info=True,
                )
            await asyncio.sleep(settings.RENTED_POD_WATCH_INTERVAL_SECONDS)

    async def tick(self) -> None:
        rented = await self.backend_client.get_rented_executors_now()
        if rented is None:
            return  # no list is no news: keep every counter
        running = {
            pod.pod_id
            for executor in rented.executors.values()
            for pod in executor.pods
            if pod.status == POD_STATUS_RUNNING
        }
        for pod_id in [p for p in self._pods if p not in running]:
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

        misses = posted = failed = 0
        for observation in watched:
            state = self._pods.setdefault(observation.pod_id, _PodState())
            if observation.result is PodSshResult.BANNER:
                if state.posted:
                    ok = await self._post(observation, REACHABLE, datetime.now(UTC))
                    posted += 1
                    failed += not ok
                    if not ok:
                        state.misses, state.first_miss_at = 0, None
                        continue  # the reachable did not land: send it again next tick
                del self._pods[observation.pod_id]
                continue
            misses += 1
            state.misses += 1
            state.first_miss_at = state.first_miss_at or probed_at
            if state.misses >= settings.RENTED_POD_WATCH_MISSES:
                ok = await self._post(observation, UNREACHABLE, state.first_miss_at)
                state.posted = True
                posted += 1
                failed += not ok

        logger.info(
            _m(
                "[pod-watch] rented pod SSH watch tick",
                extra=get_extra_info(
                    {"pods": len(watched), "misses": misses, "posted": posted, "failed": failed}
                ),
            )
        )

    async def _post(self, observation: PodSshObservation, state: str, since: datetime) -> bool:
        """True when the backend answered; a missing route (older backend) counts as answered."""
        down = state == UNREACHABLE
        try:
            response = await self.backend_client.report_pod_ssh_reachability(
                observation.pod_id,
                state=state,
                since=since.isoformat(),
                result=observation.result.value if down else None,
                errno=observation.errno if down else None,
            )
        except BackendRejected as exc:
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
