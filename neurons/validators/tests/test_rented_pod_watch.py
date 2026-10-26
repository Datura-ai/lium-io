"""DAH-5215: the between-cycles watch of rented pods' SSH (services/rented_pod_watch.py).

A fake probe and a fake backend client drive the per-pod state machine tick by tick.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from clients.backend_client import BackendRejected
from protocol.vc_protocol.compute_requests import (
    PodSshReachabilityResponse,
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from protocol.vc_protocol.validator_requests import PodSshObservation, PodSshResult
from services.rented_pod_watch import RentedPodWatch

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
    def __init__(self, rented: RentedExecutorsResponse, *, reject: bool = False):
        self.rented = rented
        self.reject = reject
        self.posts: list[dict] = []

    async def get_rented_executors_now(self):
        return self.rented

    async def report_pod_ssh_reachability(self, pod_id, *, state, since, result, errno):
        self.posts.append(
            {"pod_id": pod_id, "state": state, "since": since, "result": result, "errno": errno}
        )
        if self.reject:
            raise BackendRejected(404)
        return PodSshReachabilityResponse(recorded=True)


class FakeProbe:
    """Answers each tick with the results queued per pod; fleet_ok as set."""

    def __init__(self):
        self.results: dict[str, PodSshResult] = {}
        self.fleet_ok = True

    async def __call__(self, rented, *, timeout, concurrency, job_batch_id):
        return {
            UUID: [
                PodSshObservation(
                    pod_id=pod_id,
                    result=result,
                    errno=111 if result is PodSshResult.REFUSED else None,
                    fleet_ok=self.fleet_ok,
                )
                for pod_id, result in self.results.items()
            ]
        }


def _watch(backend, probe) -> RentedPodWatch:
    return RentedPodWatch(backend_client=backend, probe=probe)


@pytest.mark.asyncio
async def test_two_misses_post_unreachable_since_the_first_miss_and_repost_while_down():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = _watch(backend, probe)
    probe.results = {"p1": PodSshResult.REFUSED}

    await watch.tick()
    assert backend.posts == []
    first_miss_at = watch._pods["p1"].first_miss_at

    await watch.tick()
    await watch.tick()

    assert [p["state"] for p in backend.posts] == ["unreachable", "unreachable"]
    for post in backend.posts:
        assert datetime.fromisoformat(post["since"]) == first_miss_at
        assert post["result"] == "refused" and post["errno"] == 111


@pytest.mark.asyncio
async def test_banner_after_unreachable_posts_reachable_once_then_resets():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = _watch(backend, probe)
    probe.results = {"p1": PodSshResult.TIMEOUT}
    await watch.tick()
    await watch.tick()

    probe.results = {"p1": PodSshResult.BANNER}
    await watch.tick()
    await watch.tick()

    assert [p["state"] for p in backend.posts] == ["unreachable", "reachable"]
    assert backend.posts[1]["result"] is None and backend.posts[1]["errno"] is None
    assert "p1" not in watch._pods


@pytest.mark.asyncio
async def test_a_banner_between_misses_resets_the_count():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = _watch(backend, probe)
    for result in (PodSshResult.NO_BANNER, PodSshResult.BANNER, PodSshResult.NO_BANNER):
        probe.results = {"p1": result}
        await watch.tick()

    assert backend.posts == []


@pytest.mark.asyncio
async def test_fleet_not_ok_changes_nothing():
    backend, probe = FakeBackend(_rented(p1="RUNNING")), FakeProbe()
    watch = _watch(backend, probe)
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
    watch = _watch(backend, probe)
    probe.results = {"p1": PodSshResult.REFUSED, "p2": PodSshResult.REFUSED}
    await watch.tick()
    await watch.tick()

    assert [p["pod_id"] for p in backend.posts] == ["p1"]

    backend.rented = _rented(p2="REBOOT_PENDING")
    await watch.tick()
    assert watch._pods == {}


@pytest.mark.asyncio
async def test_an_older_backend_404_is_tolerated():
    backend, probe = FakeBackend(_rented(p1="RUNNING"), reject=True), FakeProbe()
    watch = _watch(backend, probe)
    probe.results = {"p1": PodSshResult.REFUSED}

    for _ in range(4):
        await watch.tick()

    assert [p["state"] for p in backend.posts] == ["unreachable"] * 3
    assert watch._route_missing_logged
