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


@pytest.mark.asyncio
async def test_a_delete_mid_custom_build_stops_the_build_before_its_export_and_tears_the_dind_down(svc, monkeypatch):
    """A deleted pod's build loaded its image onto the host after the delete and filled the disk under the
    renter's retry on the same node."""
    from services.docker_service import _CreateCancelledByDelete
    from test_custom_dockerfile_build import _base_payload, _make_dind_ssh

    ssh_client = _make_dind_ssh()
    build_started, build_cancelled, seen = asyncio.Event(), asyncio.Event(), []

    async def esl(**kwargs):
        seen.append(kwargs["command"])
        if "docker build" in kwargs["command"]:
            build_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                build_cancelled.set()
                raise
        return True, ""

    monkeypatch.setattr(svc, "execute_and_stream_logs", esl)
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    monkeypatch.setattr(svc, "cleanup_failed_container_creation", AsyncMock())
    payload = _base_payload(dockerfile_content="FROM alpine\n")

    async def tracked_build():
        with inflight_creates.track(payload.pod_id):
            return await svc._custom_build_image(
                ssh_client=ssh_client, payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id}
            )

    build = asyncio.create_task(tracked_build())
    await asyncio.wait_for(build_started.wait(), 10)
    assert inflight_creates.cancel(payload.pod_id)
    with pytest.raises(_CreateCancelledByDelete):
        await asyncio.wait_for(build, 10)

    assert build_cancelled.is_set()
    assert not any("docker save" in c for c in seen)
    assert any("rm -f" in c and "lium-dind-build-" in c for c in ssh_client.calls)
