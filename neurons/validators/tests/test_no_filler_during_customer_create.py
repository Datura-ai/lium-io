"""A filler create stands down at `docker run` while this connector creates a customer's container on
the same executor: the customer's sweep would remove the filler, or, if the executor's create lock
lapsed and the sweep already ran, the filler would run beside the renter.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from unittest.mock import AsyncMock, Mock

import pytest
import services.docker_service as ds_module
from payload_models.payloads import (
    ContainerCreated,
    FailedContainerErrorCodes,
    FailedContainerRequest,
    WorkloadKind,
)
from services.docker_service import DockerService, customer_creates
from services.miner_service import MinerService
from services.rental_docker_sdk import RENTAL_NETWORK_OPTIONS, RentalDockerSdkClient
from test_deploy_optimizations import _docker_client, _patch_happy, _payload, _run, _ssh_client


@pytest.fixture
def svc() -> DockerService:
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


@pytest.fixture
def miner_service() -> MinerService:
    return MinerService(ssh_service=Mock(), task_service=Mock(), redis_service=Mock(), attestation_service=Mock())


@pytest.mark.asyncio
async def test_filler_create_is_refused_before_docker_run_while_a_customer_create_runs_on_its_executor(
    svc: DockerService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_happy(svc, monkeypatch, _ssh_client())
    filler = _payload(workload_kind=WorkloadKind.FILLER)
    customer = _payload(executor_id=filler.executor_id)

    with customer_creates.track(customer):
        result = await _run(svc, filler)

    assert isinstance(result, FailedContainerRequest)
    assert result.error_code == FailedContainerErrorCodes.RentingInProgress
    assert result.failure_step == "customer_create_in_flight"
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


@pytest.mark.parametrize(
    ("same_miner", "same_executor_id", "refused"),
    [(True, True, True), (False, True, False), (True, False, False)],
    ids=["same_executor", "same_executor_id_of_another_miner", "another_executor_of_the_miner"],
)
@pytest.mark.asyncio
async def test_customer_create_blocks_a_filler_only_on_its_own_executor(
    svc: DockerService, monkeypatch: pytest.MonkeyPatch, same_miner: bool, same_executor_id: bool, refused: bool
) -> None:
    _patch_happy(svc, monkeypatch, _ssh_client())
    filler = _payload(workload_kind=WorkloadKind.FILLER)
    customer = _payload(
        miner_hotkey=filler.miner_hotkey if same_miner else "other-miner",
        executor_id=filler.executor_id if same_executor_id else _payload().executor_id,
    )

    with customer_creates.track(customer):
        result = await _run(svc, filler)

    assert isinstance(result, FailedContainerRequest if refused else ContainerCreated)


@pytest.mark.parametrize(
    ("registers_during", "docker_calls", "removed_container_id"),
    [
        ("stream_log", [], None),
        ("create_host_config", ["create_host_config"], None),
        ("create_container", ["create_host_config", "create_container"], "filler-container"),
        ("start", ["create_host_config", "create_container"], None),
    ],
    ids=[
        "last_await_before_docker_run",
        "docker_thread_before_create",
        "daemon_creating_the_container",
        "customer_sweep_between_create_and_start",
    ],
)
@pytest.mark.asyncio
async def test_customer_create_registering_until_the_filler_container_exists_refuses_the_filler(
    svc: DockerService,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    registers_during: str,
    docker_calls: list[str],
    removed_container_id: str | None,
) -> None:
    _patch_happy(svc, monkeypatch, _ssh_client())
    monkeypatch.delattr(svc, "_run_rental_docker_create_with_port_retry")
    docker_api = Mock(inspect_network=Mock(return_value={"Driver": "bridge", "Options": RENTAL_NETWORK_OPTIONS}))
    _docker_client(svc).run_container = RentalDockerSdkClient(docker_api).run_container
    steps = Mock(cleanup=AsyncMock(return_value=False), restore_power=AsyncMock())
    monkeypatch.setattr(svc, "cleanup_failed_container_creation", steps.cleanup)
    monkeypatch.setattr(ds_module, "restore_filler_pod_gpu_power_limits", steps.restore_power)
    filler = _payload(workload_kind=WorkloadKind.FILLER)
    customer_registration = contextlib.ExitStack()

    def register_customer(*_args, **_kwargs) -> dict:
        customer_registration.enter_context(customer_creates.track(_payload(executor_id=filler.executor_id)))
        if registers_during == "start":
            raise RuntimeError("No such container: filler-container")
        return {"Id": "filler-container"}

    stream_log_blocked, stream_log_released = asyncio.Event(), asyncio.Event()

    async def stream_log_blocking_at_docker_run(text: str, *_args) -> None:
        if text == "Creating docker container" and registers_during == "stream_log":
            stream_log_blocked.set()
            await stream_log_released.wait()

    monkeypatch.setattr(svc, "stream_log", stream_log_blocking_at_docker_run)
    if registers_during != "stream_log":
        getattr(docker_api, registers_during).side_effect = register_customer

    with customer_registration:
        filler_create = asyncio.create_task(_run(svc, filler))
        if registers_during == "stream_log":
            await asyncio.wait_for(stream_log_blocked.wait(), timeout=5)
            register_customer()
            stream_log_released.set()
        result = await filler_create

    assert result.error_code == FailedContainerErrorCodes.RentingInProgress
    assert result.failure_step == "customer_create_in_flight"
    assert [name for name in ("create_host_config", "create_container") if getattr(docker_api, name).called] == docker_calls
    # DAH-2356: the PEARL cap is lifted only once the filler container is removed
    assert [call[0] for call in steps.mock_calls] == ["cleanup", "restore_power"]
    assert steps.cleanup.await_args.kwargs["container_id"] == removed_container_id
    # an expected outcome, not a validator fault (DAH-3593); a failed `docker start` is still logged as one
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR and "docker" in r.name]
    assert bool(errors) == (registers_during == "start")


async def _route_that_succeeds(_payload) -> str:
    return "created"


async def _route_that_fails(_payload) -> str:
    raise RuntimeError("create failed")


@pytest.mark.parametrize("route", [_route_that_succeeds, _route_that_fails], ids=["success", "failure"])
@pytest.mark.asyncio
async def test_customer_create_is_tracked_only_while_it_runs(miner_service: MinerService, route) -> None:
    customer = _payload()
    seen_while_running: list[bool] = []

    async def tracked_route(routed):
        seen_while_running.append(customer_creates.is_running(routed.miner_hotkey, routed.executor_id))
        return await route(routed)

    miner_service._route_container = tracked_route

    with pytest.raises(RuntimeError) if route is _route_that_fails else contextlib.nullcontext():
        await miner_service.handle_container(customer)

    assert seen_while_running == [True]
    assert customer_creates.is_running(customer.miner_hotkey, customer.executor_id) is False


@pytest.mark.asyncio
async def test_cancelled_customer_create_is_untracked(miner_service: MinerService) -> None:
    customer = _payload()
    route_entered = asyncio.Event()

    async def hanging_route(_routed):
        route_entered.set()
        await asyncio.Event().wait()

    miner_service._route_container = hanging_route
    create_task = asyncio.create_task(miner_service.handle_container(customer))
    await route_entered.wait()
    assert customer_creates.is_running(customer.miner_hotkey, customer.executor_id) is True

    create_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await create_task

    assert customer_creates.is_running(customer.miner_hotkey, customer.executor_id) is False


@pytest.mark.asyncio
async def test_filler_create_does_not_count_as_a_customer_create(miner_service: MinerService) -> None:
    # Otherwise every filler would refuse itself at `docker run`.
    filler = _payload(workload_kind=WorkloadKind.FILLER)
    seen_while_running: list[bool] = []

    async def tracked_route(routed):
        seen_while_running.append(customer_creates.is_running(routed.miner_hotkey, routed.executor_id))
        return "created"

    miner_service._route_container = tracked_route

    await miner_service.handle_container(filler)

    assert seen_while_running == [False]

