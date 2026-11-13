"""A filler create stands down at `docker run` while this connector creates a customer's container on
the same executor, or did since the filler create started: the customer's sweep would remove the filler, or, if the executor's create lock
lapsed and the sweep already ran, the filler would run beside the renter.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from payload_models.payloads import (
    WorkloadKind,
)
from docker.errors import APIError

from services import docker_service
from services.docker_service import customer_creates
from services.rental_docker_sdk import RentalDockerOperationError
from test_deploy_optimizations import _payload


@contextlib.contextmanager
def _customer_create_that_reached_create_container(customer):
    with customer_creates.track(customer):
        customer_creates.reached_create_container(customer)
        yield


def test_filler_is_refused_by_a_customer_create_ending_after_its_start_not_by_older_ones() -> None:
    customer = _payload()
    filler = _payload(workload_kind=WorkloadKind.FILLER, executor_id=customer.executor_id)
    with _customer_create_that_reached_create_container(customer):
        pass
    customer_running = contextlib.ExitStack()
    customer_running.enter_context(_customer_create_that_reached_create_container(customer))

    with customer_creates.track(filler):
        customer_running.close()
        refused_after_customer_ended = customer_creates.ran_since_filler_started(filler)
    with customer_creates.track(filler):
        refused_after_older_customer_creates = customer_creates.ran_since_filler_started(filler)
        with _customer_create_that_reached_create_container(customer):
            pass
        with customer_creates.track(filler.model_copy()):
            pass
        refused_beside_a_retry_of_its_pod = customer_creates.ran_since_filler_started(filler)

    assert refused_after_customer_ended is True
    assert refused_after_older_customer_creates is False
    assert refused_beside_a_retry_of_its_pod is True
    assert id(filler) not in customer_creates._ended_seen_by_filler_create


@pytest.mark.asyncio
async def test_customer_create_registered_during_an_in_flight_prune_waits_for_it() -> None:
    customer = _payload()
    pruning = contextlib.ExitStack()
    may_prune = pruning.enter_context(
        customer_creates.prune_unless_customer_create(customer.miner_hotkey, customer.executor_id)
    )

    with customer_creates.track(customer):
        waiting = asyncio.ensure_future(customer_creates.wait_for_prunes(customer, poll_s=0.01))
        await asyncio.sleep(0.05)
        waited_while_pruning = not waiting.done()
        pruning.close()
        await asyncio.wait_for(waiting, 1)
        with customer_creates.prune_unless_customer_create(
            customer.miner_hotkey, customer.executor_id
        ) as prune_beside_customer:
            pass

    assert may_prune
    assert waited_while_pruning
    assert not prune_beside_customer


@pytest.mark.asyncio
async def test_customer_create_reports_a_prune_still_running_at_the_deadline() -> None:
    customer = _payload()
    with customer_creates.prune_unless_customer_create(customer.miner_hotkey, customer.executor_id):
        with customer_creates.track(customer):
            cleared = await customer_creates.wait_for_prunes(customer, timeout_s=0.03, poll_s=0.01)

    assert cleared is False


class _Prune:
    def __init__(self, error: BaseException | None = None):
        self.release = asyncio.Event()
        self.error = error

    async def prune_images(self):
        await self.release.wait()
        if self.error is not None:
            raise self.error


@pytest.mark.asyncio
async def test_a_cancelled_delete_keeps_the_prune_registered_until_docker_returns(monkeypatch) -> None:
    customer = _payload()
    docker = _Prune()
    assert customer_creates.begin_prune(customer.miner_hotkey, customer.executor_id)
    prune = asyncio.shield(
        docker_service._prune_images_then_release(docker, {}, customer.miner_hotkey, customer.executor_id)
    )
    delete = asyncio.ensure_future(prune)
    await asyncio.sleep(0.01)
    delete.cancel()

    with customer_creates.track(customer):
        held_after_cancel = not await customer_creates.wait_for_prunes(customer, timeout_s=0.03, poll_s=0.01)
        docker.release.set()
        cleared = await customer_creates.wait_for_prunes(customer, timeout_s=1, poll_s=0.01)

    assert held_after_cancel
    assert cleared is True


@pytest.mark.parametrize(
    ("cause", "fenced"),
    [(TimeoutError("read timed out"), True), (APIError("409 conflict: prune already running"), False)],
)
@pytest.mark.asyncio
async def test_a_prune_of_unknown_outcome_keeps_the_executor_fenced(monkeypatch, cause, fenced) -> None:
    monkeypatch.setattr(docker_service, "PRUNE_UNKNOWN_COMPLETION_FENCE_SECONDS", 0.2)
    customer = _payload()
    docker = _Prune(error=RentalDockerOperationError("Docker SDK prune images failed"))
    docker.error.__cause__ = cause
    docker.release.set()
    assert customer_creates.begin_prune(customer.miner_hotkey, customer.executor_id)

    with pytest.raises(RentalDockerOperationError):
        await docker_service._prune_images_then_release(docker, {}, customer.miner_hotkey, customer.executor_id)
    with customer_creates.track(customer):
        fenced_now = not await customer_creates.wait_for_prunes(customer, timeout_s=0.03, poll_s=0.01)
        cleared_later = await customer_creates.wait_for_prunes(customer, timeout_s=1, poll_s=0.01)

    assert fenced_now is fenced
    assert cleared_later is True
