"""DAH-5283: a delete that lands while its create is still pulling stops the pull.

The delete only flagged the create, and the create read the flag after the pull, so a slow pull kept
streaming up to its 1 h deadline after the renter deleted the pod.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import ContainerDeleteRequest, FailedContainerRequest
from services.docker_service import DockerService, executor_key, inflight_creates
from test_deploy_optimizations import _docker_client, _patch_happy, _payload, _run, _ssh_client


@pytest.fixture
def svc():
    redis_service = Mock(get=AsyncMock(return_value=None), mget=AsyncMock(side_effect=lambda keys: [None] * len(keys)), set=AsyncMock(), write_atomically=AsyncMock(), get_rented_machine=AsyncMock(return_value=None), rented_machines_synced=AsyncMock(return_value=True))
    service = DockerService(ssh_service=Mock(), redis_service=redis_service, attestation_service=Mock())
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
        with inflight_creates.track(payload.pod_id, executor_key(payload)):
            return await _run(svc, payload)

    create = asyncio.create_task(tracked_create())
    await asyncio.wait_for(pull_started.wait(), 10)
    assert inflight_creates.cancel(payload.pod_id, executor_key(payload))
    result = await asyncio.wait_for(create, 10)

    assert isinstance(result, FailedContainerRequest)
    assert pull_cancelled.is_set()
    docker.aclose.assert_awaited()
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_pull_without_a_delete_runs_to_the_end(svc, monkeypatch):
    payload = _payload()
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=1))

    with inflight_creates.track(payload.pod_id, executor_key(payload)):
        result = await _run(svc, payload)

    assert not isinstance(result, FailedContainerRequest)
    assert _docker_client(svc).pulled_images == [payload.docker_image]
    svc._run_rental_docker_create_with_port_retry.assert_awaited()


@pytest.mark.asyncio
async def test_wait_cancelled_returns_only_after_the_cancel():
    with inflight_creates.track("pod-5283", ("miner", "exec")):
        waiter = asyncio.create_task(inflight_creates.wait_cancelled("pod-5283"))
        await asyncio.sleep(0)
        assert not waiter.done()
        inflight_creates.cancel("pod-5283", ("miner", "exec"))
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
        with inflight_creates.track(payload.pod_id, executor_key(payload)):
            return await svc._custom_build_image(
                ssh_client=ssh_client, payload=payload, log_tag="t", default_extra={"pod_id": payload.pod_id}
            )

    build = asyncio.create_task(tracked_build())
    await asyncio.wait_for(build_started.wait(), 10)
    assert inflight_creates.cancel(payload.pod_id, executor_key(payload))
    with pytest.raises(_CreateCancelledByDelete):
        await asyncio.wait_for(build, 10)

    assert build_cancelled.is_set()
    assert not any("docker save" in c for c in seen)
    assert any("rm -f" in c and "lium-dind-build-" in c for c in ssh_client.calls)


@pytest.mark.asyncio
async def test_a_step_failing_after_the_delete_is_the_renters_cancel_and_holds_no_node(svc, monkeypatch):
    """DAH-6410: a step that ends in an error after the renter's delete flagged the create (a long node
    wait, then a failure) is the renter's cancel, not the provider's failure."""
    payload = _payload()
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=1))
    hold = AsyncMock()
    monkeypatch.setattr(svc, "hold_node_until_probe_passes", hold)

    async def deleted_then_failed(*args, **kwargs):
        inflight_creates.cancel(payload.pod_id, executor_key(payload))
        raise RuntimeError("volume plugin refused")

    svc.create_local_volume = AsyncMock(side_effect=deleted_then_failed)

    with inflight_creates.track(payload.pod_id, executor_key(payload)):
        result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "cancelled_by_delete"
    assert hold.await_args.args[1] == "cancelled_by_delete"
    from services.docker_service import is_host_fault_create_failure

    assert not is_host_fault_create_failure(hold.await_args.args[1], hold.await_args.kwargs["error"])


@pytest.mark.asyncio
async def test_a_delete_while_dockerd_runs_the_create_removes_the_container_once_it_exists(svc, monkeypatch):
    """lium-io#2406 review: a delete acknowledged while dockerd still ran `containers/create` must not leave
    that container behind. The Docker thread cannot be cancelled, so the create waits for dockerd's answer
    and then removes the container and its volume; a cleanup run before the container existed removed
    nothing and the container then started holding the pod volume. The delete itself does not wait for
    dockerd: the create's run step removes its own container, so the delete answers while it is pending."""
    payload = _payload()
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=1))
    events: list[str] = []
    run_started, released = asyncio.Event(), asyncio.Event()

    async def dockerd():
        await released.wait()
        events.append("created")
        return "cid"

    dockerd_task: list[asyncio.Task] = []

    async def blocked_create(**kwargs):
        run_started.set()
        dockerd_task.append(asyncio.ensure_future(dockerd()))
        # like the Docker thread: cancelling the caller does not stop dockerd's create
        return await asyncio.shield(dockerd_task[0])

    async def cleanup(**kwargs):
        events.append("cleanup")
        return True

    svc._run_rental_docker_create_with_port_retry = blocked_create
    monkeypatch.setattr(svc, "cleanup_failed_container_creation", cleanup)

    async def tracked_create():
        with inflight_creates.track(payload.pod_id, executor_key(payload)):
            return await _run(svc, payload)

    create = asyncio.create_task(tracked_create())
    await asyncio.wait_for(run_started.wait(), 10)
    delete = ContainerDeleteRequest(
        miner_hotkey=payload.miner_hotkey,
        executor_id=payload.executor_id,
        pod_id=payload.pod_id,
        container_name=svc.get_container_name(payload),
    )
    # the delete's own wait for the create, well inside its 30 s bound, while dockerd still runs the create
    assert await asyncio.wait_for(svc._stop_create_before_delete(delete, Mock()), 5) is None
    assert inflight_creates.is_cancelled(payload.pod_id, executor_key(payload)) and events == []
    released.set()
    result = await asyncio.wait_for(create, 10)
    await asyncio.wait_for(dockerd_task[0], 10)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "cancelled_by_delete"
    assert events[0] == "created" and "cleanup" in events


@pytest.mark.asyncio
async def test_a_delete_during_the_resume_wait_keeps_waiting_and_never_starts_dockerds_container(svc, monkeypatch):
    """lium-io#2406 review: the resume after a lost `containers/create` answer waits while dockerd holds the
    name (409, inspect 404). A delete during that wait must not end it: the first create registers later,
    and is adopted but never started, so the create's cleanup removes it and its volume."""
    import requests
    import services.rental_docker_sdk as sdk
    from docker.errors import APIError
    from services.docker_service import _CreateCancelledByDelete
    from services.rental_docker_sdk import RentalDockerSdkClient
    from test_docker_transport_retry import _FakeAdapter, _FakeApiClient, _not_found

    payload = _payload()
    _patch_happy(svc, monkeypatch, _ssh_client(inspect_exit=1))
    monkeypatch.setattr(sdk, "_INFLIGHT_CREATE_POLL_SECONDS", 0)
    with inflight_creates.track(payload.pod_id, executor_key(payload)):
        await _run(svc, payload)
        # the run spec the create hands the SDK, with its own `before_create`
        spec = svc._run_rental_docker_create_with_port_retry.await_args.kwargs["run_spec"]
        held = APIError(
            "Conflict",
            response=Mock(status_code=409, url="containers/create", reason="Conflict"),
            explanation=f'Conflict. The container name "/{spec.name}" is already in use by container "abc".',
        )
        api = _FakeApiClient(adapter=_FakeAdapter())
        api.create_container_errors = [requests.exceptions.ReadTimeout("Read timed out. (read timeout=60)"), held, held]
        inspects = iter([_not_found(), "delete"])

        def inspect_container(name):
            api.calls.append("inspect_container")
            step = next(inspects, None)
            if step == "delete":  # the renter deletes while dockerd still runs the first create
                assert inflight_creates.cancel(payload.pod_id, executor_key(payload))
                raise _not_found()
            if step is not None:
                raise step
            return {"Id": "first", "Config": {"Image": spec.image}}

        api.inspect_container = inspect_container
        with pytest.raises(_CreateCancelledByDelete):
            await RentalDockerSdkClient(api).run_container(spec)
    assert api.calls[-1] == "inspect_container" and "start" not in api.calls
    assert api.calls.count("create_container") == 3
