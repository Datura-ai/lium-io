"""DAH-5215: the between-cycles watch of rented pods' SSH (services/rented_pod_watch.py).

A fake probe and a fake backend client drive the per-pod state machine tick by tick.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest
import services.rented_pod_watch as watch_module
from clients.backend_client import BackendRejected
from protocol.vc_protocol.compute_requests import (
    PodSshReachabilityResponse,
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
    RentedPodTargets,
)
from protocol.vc_protocol.validator_requests import PodSshObservation, PodSshResult
from services.rented_pod_watch import RentedPodWatch

from core.config import settings

UUID = "11111111-1111-1111-1111-111111111111"


def _rented(**statuses: str) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            UUID: RentedExecutor(
                miner_hotkey="hk",
                executor_ip_address="198.51.100.1",
                executor_ip_port="9001",
                pods=[
                    RentedPod(pod_id=pod_id, container_name=f"c-{pod_id}", ssh_port=40100, status=status)
                    for pod_id, status in statuses.items()
                ],
            )
        },
    )


class FakeBackend:
    def __init__(self, rented: RentedExecutorsResponse, *, reject: int | None = None):
        self.rented = rented
        self.reject = reject
        self.hang = False
        self.posts: list[dict] = []

    async def get_rented_pod_targets(self):
        return self.rented

    async def report_pod_ssh_reachability(self, pod_id, **fields):
        self.posts.append({"pod_id": pod_id, **fields})
        if self.hang:
            await asyncio.sleep(10)
        if self.reject:
            raise BackendRejected(self.reject)
        return PodSshReachabilityResponse(recorded=True)


class FakeProbe:
    """Answers each tick with the results queued per pod; the long no_banner retry reads `retry`."""

    def __init__(self):
        self.results: dict[str, PodSshResult] = {}
        self.retry: dict[str, PodSshResult] = {}
        self.retry_timeouts: list[float] = []
        self.fleet_ok = True

    async def __call__(self, rented, *, timeout, concurrency, job_batch_id):
        asked = {pod.pod_id for executor in rented.executors.values() for pod in executor.pods}
        results = self.results
        if job_batch_id == "pod-watch-retry":
            self.retry_timeouts.append(timeout)
            results = {pod_id: self.retry.get(pod_id, result) for pod_id, result in results.items()}
        return {
            UUID: [
                PodSshObservation(
                    pod_id=pod_id,
                    result=result,
                    errno=111 if result is PodSshResult.REFUSED else None,
                    fleet_ok=self.fleet_ok,
                )
                for pod_id, result in results.items()
                if pod_id in asked
            ]
        }


def _watch(backend, probe) -> RentedPodWatch:
    return RentedPodWatch(backend_client=backend, probe=probe)


async def _answered(backend, probe, *pods) -> RentedPodWatch:
    """A watch past its first tick on which every pod sent a banner (the restart reachables dropped)."""
    watch = _watch(backend, probe)
    probe.results = {pod: PodSshResult.BANNER for pod in pods}
    await watch.tick()
    backend.posts.clear()
    return watch


def _states(backend) -> list[str]:
    return [p["state"] for p in backend.posts]


@pytest.mark.asyncio
async def test_two_misses_post_unreachable_since_the_first_miss_and_repost_while_down():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = await _answered(backend, probe, "p1")
    probe.results = {"p1": PodSshResult.REFUSED}

    await watch.tick()
    assert backend.posts == []
    first_miss_at = watch._pods["p1"].first_miss_at

    await watch.tick()
    await watch.tick()

    assert _states(backend) == ["unreachable", "unreachable"]
    for post in backend.posts:
        assert datetime.fromisoformat(post["since"]) == first_miss_at
        assert post["result"] == "refused" and post["errno"] == 111
        assert post["answered_before"] is True
        assert datetime.fromisoformat(post["observed_at"]) >= first_miss_at


@pytest.mark.asyncio
async def test_banner_after_unreachable_posts_reachable_once_then_resets():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = await _answered(backend, probe, "p1")
    probe.results = {"p1": PodSshResult.TIMEOUT}
    await watch.tick()
    await watch.tick()

    probe.results = {"p1": PodSshResult.BANNER}
    await watch.tick()
    await watch.tick()

    assert _states(backend) == ["unreachable", "reachable"]
    reachable = backend.posts[1]
    assert reachable["result"] is None and reachable["errno"] is None and reachable["answered_before"] is None
    assert reachable["observed_at"]
    assert watch._pods["p1"].answered and not watch._pods["p1"].posted and watch._pods["p1"].misses == 0


@pytest.mark.asyncio
async def test_a_banner_between_misses_resets_the_count():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = await _answered(backend, probe, "p1")
    for result in (PodSshResult.REFUSED, PodSshResult.BANNER, PodSshResult.REFUSED):
        probe.results = {"p1": result}
        await watch.tick()

    assert backend.posts == []


@pytest.mark.asyncio
async def test_fleet_not_ok_changes_nothing():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = await _answered(backend, probe, "p1")
    probe.results = {"p1": PodSshResult.TIMEOUT}
    await watch.tick()
    first_miss_at = watch._pods["p1"].first_miss_at

    probe.fleet_ok = False
    await watch.tick()
    await watch.tick()

    assert backend.posts == []
    assert watch._pods["p1"].misses == 1
    assert watch._pods["p1"].first_miss_at == first_miss_at


@pytest.mark.asyncio
async def test_non_running_pods_are_ignored_and_pods_that_leave_are_forgotten():
    backend, probe = FakeBackend(_rented(p1="RUNNING", p2="REBOOT_PENDING")), FakeProbe()
    watch = await _answered(backend, probe, "p1", "p2")
    probe.results = {"p1": PodSshResult.REFUSED, "p2": PodSshResult.REFUSED}
    await watch.tick()
    await watch.tick()

    assert [p["pod_id"] for p in backend.posts] == ["p1"]

    backend.rented = _rented(p2="REBOOT_PENDING")
    await watch.tick()
    assert watch._pods == {}


@pytest.mark.asyncio
async def test_a_reboot_keeps_the_owed_reachable():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = await _answered(backend, probe, "p1")
    probe.results = {"p1": PodSshResult.REFUSED}
    await watch.tick()
    await watch.tick()

    backend.rented = _rented(p1="REBOOT_PENDING")
    await watch.tick()
    backend.rented = _rented(p1="RUNNING")
    probe.results = {"p1": PodSshResult.BANNER}
    await watch.tick()

    assert _states(backend) == ["unreachable", "reachable"]


@pytest.mark.asyncio
async def test_an_older_backend_404_is_tolerated():
    backend, probe = FakeBackend(_rented(p1="RUNNING"), reject=404), FakeProbe()
    watch = await _answered(backend, probe, "p1")
    probe.results = {"p1": PodSshResult.REFUSED}

    for _ in range(4):
        await watch.tick()

    assert _states(backend) == ["unreachable"] * 3
    assert watch._route_missing_logged


# review ask 3: only 404 means "no route"
@pytest.mark.asyncio
async def test_a_rejected_reachable_other_than_404_is_sent_again():
    backend, probe = FakeBackend(_rented(p1="RUNNING"), reject=422), FakeProbe()
    watch = _watch(backend, probe)
    probe.results = {"p1": PodSshResult.BANNER}

    await watch.tick()
    await watch.tick()

    assert _states(backend) == ["reachable", "reachable"]
    assert watch._pods["p1"].posted and not watch._route_missing_logged


# review ask 4: the first tick after a restart clears any marker the backend still holds
@pytest.mark.asyncio
async def test_first_tick_sends_reachable_for_every_answering_pod_once():
    backend, probe = FakeBackend(_rented(p1="RUNNING", p2="RUNNING")), FakeProbe()
    watch = _watch(backend, probe)
    probe.results = {"p1": PodSshResult.BANNER, "p2": PodSshResult.TIMEOUT}

    await watch.tick()
    await watch.tick()

    assert [(p["pod_id"], p["state"]) for p in backend.posts] == [("p1", "reachable")]


# review ask 5: a pod that never sent a banner (no sshd) is never reported
@pytest.mark.asyncio
async def test_a_pod_that_never_answered_is_never_reported():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = _watch(backend, probe)
    probe.results = {"p1": PodSshResult.REFUSED}

    for _ in range(5):
        await watch.tick()

    assert backend.posts == []
    assert not watch._pods["p1"].answered and watch._pods["p1"].misses == 0


# review ask 6: no_banner retries once with the long timeout, then needs the long run of misses
@pytest.mark.asyncio
async def test_no_banner_that_answers_the_long_retry_is_not_a_miss():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = await _answered(backend, probe, "p1")
    probe.results, probe.retry = {"p1": PodSshResult.NO_BANNER}, {"p1": PodSshResult.BANNER}

    for _ in range(settings.RENTED_POD_WATCH_NO_BANNER_MISSES + 1):
        await watch.tick()

    assert backend.posts == [] and watch._pods["p1"].misses == 0
    assert probe.retry_timeouts[0] == settings.RENTED_POD_WATCH_NO_BANNER_RETRY_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_no_banner_needs_its_own_longer_run_of_misses():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = await _answered(backend, probe, "p1")
    probe.results = {"p1": PodSshResult.NO_BANNER}

    for _ in range(settings.RENTED_POD_WATCH_NO_BANNER_MISSES - 1):
        await watch.tick()
    assert backend.posts == []

    await watch.tick()
    assert _states(backend) == ["unreachable"]
    assert backend.posts[0]["result"] == "no_banner" and backend.posts[0]["answered_before"] is True
    assert settings.RENTED_POD_WATCH_NO_BANNER_MISSES > settings.RENTED_POD_WATCH_MISSES


# review ask 2: the reports go out together and a slow backend cannot hold the tick
@pytest.mark.asyncio
async def test_slow_reports_are_cut_at_the_deadline_and_stay_owed(monkeypatch):
    monkeypatch.setattr(watch_module, "REPORTS_DEADLINE_SECONDS", 0.05)
    backend, probe = FakeBackend(_rented(p1="RUNNING", p2="RUNNING")), FakeProbe()
    backend.hang = True
    watch = _watch(backend, probe)
    probe.results = {"p1": PodSshResult.BANNER, "p2": PodSshResult.BANNER}

    await asyncio.wait_for(watch.tick(), timeout=1)

    assert len(backend.posts) == 2  # both started before either finished
    assert watch._pods["p1"].posted and watch._pods["p2"].posted


@pytest.mark.asyncio
async def test_the_real_client_posts_the_report_and_asks_for_the_404_raise(monkeypatch):
    # the fakes above stand in for BackendClient, so this pins the real call path to _request
    from clients.backend_client import BackendClient

    seen = {}

    async def fake_request(self, method, path, response_model, **kwargs):
        seen.update(method=method, path=path, **kwargs)
        return response_model(recorded=True)

    monkeypatch.setattr(BackendClient, "_request", fake_request)
    client = object.__new__(BackendClient)

    answer = await client.report_pod_ssh_reachability(
        "pod-1",
        state="unreachable",
        since="2026-10-26T17:56:00+00:00",
        result="timeout",
        errno=None,
        observed_at="2026-10-26T17:58:00+00:00",
        answered_before=True,
    )

    assert answer == PodSshReachabilityResponse(recorded=True)
    assert seen["method"] == "POST"
    assert seen["path"] == "/internal/pods/pod-1/ssh-reachability"
    assert seen["raise_on_4xx"] is True
    assert seen["json_data"] == {
        "state": "unreachable",
        "since": "2026-10-26T17:56:00+00:00",
        "result": "timeout",
        "errno": None,
        "probe_result": "timeout",
        "answered_before": True,
        "observed_at": "2026-10-26T17:58:00+00:00",
        "source": "watch",
    }

# review comment 4469024024: the whole tick is bounded and the next one is scheduled from its start
@pytest.mark.asyncio
async def test_run_bounds_the_whole_tick_and_schedules_from_its_start(monkeypatch):
    monkeypatch.setattr(settings, "RENTED_POD_WATCH_INTERVAL_SECONDS", 0.2)
    watch = _watch(FakeBackend(_rented()), FakeProbe())
    loop = asyncio.get_running_loop()
    starts = []

    async def slow_tick():
        starts.append(loop.time())
        await asyncio.sleep(0.15 if len(starts) == 1 else 10)  # a tick in time, then one that hangs

    watch.tick = slow_tick
    await asyncio.wait_for(watch.run(lambda: len(starts) >= 3), timeout=2)

    assert len(starts) == 3
    assert starts[1] - starts[0] < 0.3  # 0.2 from the first start, not 0.15 + 0.2 after it
    assert starts[2] - starts[1] < 0.3  # the hung tick was cut at the interval


# review ask 1: the light target list, shaped as the rented list the probe takes
@pytest.mark.asyncio
async def test_the_light_list_becomes_probe_targets_and_a_4xx_falls_back(monkeypatch):
    from clients.backend_client import BackendClient

    calls = []
    full = _rented(p9="RUNNING")

    async def fake_request(self, method, path, response_model, **kwargs):
        calls.append(path)
        if self.light is None:
            raise BackendRejected(422)
        return response_model.model_validate(self.light)

    async def fake_full(self):
        calls.append("full")
        return full

    monkeypatch.setattr(BackendClient, "_request", fake_request)
    monkeypatch.setattr(BackendClient, "get_rented_executors_now", fake_full)
    client = object.__new__(BackendClient)

    client.light = [
        {
            "pod_id": "p1",
            "ip": "198.51.100.7",
            "miner_hotkey": "hk1",
            "port": 40022,
            "status": "RUNNING",
        }
    ]
    rented = await client.get_rented_pod_targets()
    assert calls == ["/internal/executors/rented?light=true"]
    (pod,) = rented.executors["p1"].pods
    assert rented.executors["p1"].executor_ip_address == "198.51.100.7"
    assert rented.executors["p1"].miner_hotkey == "hk1"
    assert (pod.pod_id, pod.ssh_port, pod.status) == ("p1", 40022, "RUNNING")

    client.light = full.model_dump()  # a backend that ignores the parameter answers the full list
    assert await client.get_rented_pod_targets() == full

    calls.clear()
    client.light = None
    assert await client.get_rented_pod_targets() == full
    assert calls == ["/internal/executors/rented?light=true", "full"]
    assert RentedPodTargets.model_validate([]).root == []


@pytest.mark.asyncio
async def test_a_pod_down_on_the_first_tick_after_a_restart_still_gets_its_reachable():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = _watch(backend, probe)
    probe.results = {"p1": PodSshResult.REFUSED}
    await watch.tick()

    probe.results = {"p1": PodSshResult.BANNER}
    await watch.tick()
    await watch.tick()

    assert _states(backend) == ["reachable"]


@pytest.mark.asyncio
async def test_an_announced_outage_keeps_reposting_when_refused_turns_into_no_banner():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = await _answered(backend, probe, "p1")
    probe.results = {"p1": PodSshResult.REFUSED}
    await watch.tick()
    await watch.tick()

    probe.results = {"p1": PodSshResult.NO_BANNER}
    await watch.tick()
    probe.results = {"p1": PodSshResult.BANNER}
    await watch.tick()
    probe.results = {"p1": PodSshResult.NO_BANNER}
    await watch.tick()

    assert _states(backend) == ["unreachable", "unreachable", "reachable"]


class SlowRetryProbe(FakeProbe):
    """The long no_banner retry takes its whole timeout; records the pods each batch dialled."""

    def __init__(self):
        super().__init__()
        self.retry_batches: list[set[str]] = []

    async def __call__(self, rented, *, timeout, concurrency, job_batch_id):
        if job_batch_id == "pod-watch-retry":
            self.retry_batches.append({p.pod_id for e in rented.executors.values() for p in e.pods})
            await asyncio.sleep(timeout)
        return await super().__call__(
            rented, timeout=timeout, concurrency=concurrency, job_batch_id=job_batch_id
        )


@pytest.mark.asyncio
async def test_retries_that_outgrow_the_tick_keep_the_short_outcomes_and_go_first_next_tick(
    monkeypatch,
):
    monkeypatch.setattr(settings, "RENTED_POD_WATCH_INTERVAL_SECONDS", 1)
    monkeypatch.setattr(settings, "RENTED_POD_WATCH_NO_BANNER_RETRY_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(settings, "RENTED_POD_SSH_PROBE_CONCURRENCY", 2)
    monkeypatch.setattr(watch_module, "RETRY_REPORTS_RESERVE_SECONDS", 0.1)
    slow = [f"n{i}" for i in range(10)]
    backend = FakeBackend(_rented(r1="RUNNING", **{p: "RUNNING" for p in slow}))
    probe = SlowRetryProbe()
    watch = await _answered(backend, probe, "r1", *slow)
    probe.results = {"r1": PodSshResult.REFUSED, **{p: PodSshResult.NO_BANNER for p in slow}}

    started = asyncio.get_running_loop().time()
    await watch.tick()
    took = asyncio.get_running_loop().time() - started

    assert took < 1
    assert watch._pods["r1"].misses == 1  # counted although the retries did not all fit
    first = set().union(*probe.retry_batches)
    assert 0 < len(first) < len(slow)
    # a pod left out of the retries changes nothing this tick
    assert all(watch._pods[p].misses == (p in first) for p in slow)

    probe.retry_batches.clear()
    await watch.tick()
    assert probe.retry_batches[0] <= set(slow) - first
