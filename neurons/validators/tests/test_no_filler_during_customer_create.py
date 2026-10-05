"""A filler create stands down at `docker run` while this connector creates a customer's container on
the same executor: the customer's create has already listed and removed the node's fillers, so a
filler started now would run beside the renter.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, Mock

import pytest
import services.docker_service as ds_module
from payload_models.payloads import (
    ContainerCreated,
    FailedContainerErrorCodes,
    FailedContainerRequest,
    WorkloadKind,
)
from services.docker_service import DockerService, _FillerRefusedForCustomerCreate, customer_creates
from services.miner_service import MinerService
from test_deploy_optimizations import _patch_happy, _payload, _run, _ssh_client


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


@pytest.mark.asyncio
async def test_customer_create_on_another_executor_does_not_block_a_filler(
    svc: DockerService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_happy(svc, monkeypatch, _ssh_client())
    filler = _payload(workload_kind=WorkloadKind.FILLER)
    customer_elsewhere = _payload()

    with customer_creates.track(customer_elsewhere):
        result = await _run(svc, filler)

    assert isinstance(result, ContainerCreated)
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_refused_filler_gets_its_gpu_power_back(svc: DockerService, monkeypatch: pytest.MonkeyPatch) -> None:
    # The PEARL cap is applied before `docker run`; the customer coming in must not inherit it.
    restore_power_limits = AsyncMock()
    monkeypatch.setattr(ds_module, "restore_filler_pod_gpu_power_limits", restore_power_limits)
    filler = _payload(workload_kind=WorkloadKind.FILLER)

    with customer_creates.track(_payload(executor_id=filler.executor_id)):
        with pytest.raises(_FillerRefusedForCustomerCreate):
            await svc._refuse_filler_during_customer_create(Mock(), filler, {})

    restore_power_limits.assert_awaited_once()


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
        seen_while_running.append(customer_creates.is_running(routed.executor_id))
        return await route(routed)

    miner_service._route_container = tracked_route

    with pytest.raises(RuntimeError) if route is _route_that_fails else contextlib.nullcontext():
        await miner_service.handle_container(customer)

    assert seen_while_running == [True]
    assert customer_creates.is_running(customer.executor_id) is False


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
    assert customer_creates.is_running(customer.executor_id) is True

    create_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await create_task

    assert customer_creates.is_running(customer.executor_id) is False


@pytest.mark.asyncio
async def test_filler_create_does_not_count_as_a_customer_create(miner_service: MinerService) -> None:
    # Otherwise every filler would refuse itself at `docker run`.
    filler = _payload(workload_kind=WorkloadKind.FILLER)
    seen_while_running: list[bool] = []

    async def tracked_route(routed):
        seen_while_running.append(customer_creates.is_running(routed.executor_id))
        return "created"

    miner_service._route_container = tracked_route

    await miner_service.handle_container(filler)

    assert seen_while_running == [False]

