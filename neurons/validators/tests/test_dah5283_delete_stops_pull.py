"""DAH-5283: a delete that lands while its create is still pulling stops the pull.

The delete only flagged the create, and the create read the flag after the pull, so a slow pull kept
streaming up to its 1 h deadline after the renter deleted the pod.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import FailedContainerRequest
from services.docker_service import DockerService, inflight_creates
from test_deploy_optimizations import _docker_client, _patch_happy, _payload, _run, _ssh_client


@pytest.fixture
def svc():
    service = DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())
    service._read_host_pid_max = AsyncMock(return_value=4_194_304)
    return service


@pytest.mark.asyncio
async def test_a_delete_mid_pull_stops_the_pull_and_closes_the_docker_client(svc, monkeypatch):
    payload = _payload()
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=1))
    docker = _docker_client(svc)
    pull_started, pull_cancelled = asyncio.Event(), asyncio.Event()

    async def endless_pull(*, image):
        pull_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pull_cancelled.set()
            raise

    docker.pull = endless_pull
    docker.aclose = AsyncMock()

    async def tracked_create():
        with inflight_creates.track(payload.pod_id):
            return await _run(svc, payload)

    create = asyncio.create_task(tracked_create())
    await asyncio.wait_for(pull_started.wait(), 10)
    assert inflight_creates.cancel(payload.pod_id)
    result = await asyncio.wait_for(create, 10)

    assert isinstance(result, FailedContainerRequest)
    assert pull_cancelled.is_set()
    docker.aclose.assert_awaited()
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_pull_without_a_delete_runs_to_the_end(svc, monkeypatch):
    payload = _payload()
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=1))

    with inflight_creates.track(payload.pod_id):
        result = await _run(svc, payload)

    assert not isinstance(result, FailedContainerRequest)
    assert _docker_client(svc).pulled_images == [payload.docker_image]
    svc._run_rental_docker_create_with_port_retry.assert_awaited()


@pytest.mark.asyncio
async def test_wait_cancelled_returns_only_after_the_cancel():
    with inflight_creates.track("pod-5283"):
        waiter = asyncio.create_task(inflight_creates.wait_cancelled("pod-5283"))
        await asyncio.sleep(0)
        assert not waiter.done()
        inflight_creates.cancel("pod-5283")
        await asyncio.wait_for(waiter, 1)
