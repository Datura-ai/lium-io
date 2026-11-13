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
from services.docker_service import confirm_no_prune_runs, customer_creates
from services.rental_docker_sdk import RentalDockerOperationError
from test_deploy_optimizations import _payload


class _CreateLocks:
    """Redis's per-executor create lock, one for every connector process (a renter's create holds it)."""

    def __init__(self):
        self.held = False

    def executor_create_lock(self, executor_id):
        return _CreateLock(self)


class _CreateLock:
    def __init__(self, locks: _CreateLocks):
        self.locks = locks
        self.mine = False

    async def acquire(self, blocking=True):
        if self.locks.held:
            return False
        self.locks.held = self.mine = True
        return True

    async def release(self):
        if self.mine:
            self.locks.held = self.mine = False


_FREE = _CreateLocks()  # no create holds the executor: every prune may run


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
        docker_service._prune_images_then_release(docker, {}, customer.miner_hotkey, customer.executor_id, _FREE)
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


class _Prunes:
    """dockerd answering one prune call after another: an exception to raise, or None for a completed prune."""

    def __init__(self, *outcomes: BaseException | None):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def prune_images(self):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if outcome is not None:
            raise outcome


def _prune_failure(cause: BaseException) -> RentalDockerOperationError:
    error = RentalDockerOperationError("Docker SDK prune images failed")
    error.__cause__ = cause
    return error


_LOST_REPLY = TimeoutError("read timed out")
_PRUNE_RUNNING = APIError("409 Client Error: Conflict (a prune operation is already running)")


async def _fenced_after(docker, customer) -> bool:
    customer_creates = docker_service.customer_creates  # a test may swap in a fresh connector's registry
    assert customer_creates.begin_prune(customer.miner_hotkey, customer.executor_id)
    with contextlib.suppress(RentalDockerOperationError):
        await docker_service._prune_images_then_release(docker, {}, customer.miner_hotkey, customer.executor_id, _FREE)
    return customer_creates.is_fenced(customer.miner_hotkey, customer.executor_id)


@pytest.mark.parametrize(
    ("outcomes", "fenced"),
    [
        # lio#2226 review: a lost reply, then dockerd still pruning and a lost reply again: no time lifts the fence
        ((_prune_failure(_LOST_REPLY), _prune_failure(_PRUNE_RUNNING), _prune_failure(_LOST_REPLY)), True),
        ((_prune_failure(_LOST_REPLY), _prune_failure(_PRUNE_RUNNING), None), False),  # a completed prune proves none runs
        ((_prune_failure(APIError("500 Server Error: disk I/O")),), False),  # dockerd answered: this prune is over
    ],
)
@pytest.mark.asyncio
async def test_only_a_completed_prune_lifts_the_fence_of_one_that_may_still_run(monkeypatch, outcomes, fenced) -> None:
    monkeypatch.setattr(docker_service, "PRUNE_CONFIRM_ATTEMPTS", 2)
    monkeypatch.setattr(docker_service, "PRUNE_CONFIRM_INTERVAL_SECONDS", 0)
    customer = _payload()
    docker = _Prunes(*outcomes)

    assert await _fenced_after(docker, customer) is fenced
    assert docker.calls == len(outcomes)

    # a later delete's prune that completes lifts a fence still standing
    assert await _fenced_after(_Prunes(None), customer) is False


@pytest.mark.asyncio
async def test_an_older_prunes_late_success_does_not_lift_a_newer_prunes_fence(monkeypatch) -> None:
    # lio#2226 review: prune A finished on the host but its reply is late; prune B started after it, its reply was
    # lost and dockerd kept saying a prune runs. A's reply must not clear the fence B left.
    monkeypatch.setattr(docker_service, "PRUNE_CONFIRM_ATTEMPTS", 1)
    monkeypatch.setattr(docker_service, "PRUNE_CONFIRM_INTERVAL_SECONDS", 0)
    customer = _payload()
    late = _Prune()
    assert customer_creates.begin_prune(customer.miner_hotkey, customer.executor_id)
    prune_a = asyncio.ensure_future(
        # its create lock lapsed past the TTL while it ran, so B could take the lock and prune
        docker_service._prune_images_then_release(late, {}, customer.miner_hotkey, customer.executor_id, _CreateLocks())
    )
    await asyncio.sleep(0.01)

    lost_then_running = _Prunes(_prune_failure(_LOST_REPLY), _prune_failure(_PRUNE_RUNNING))
    assert await _fenced_after(lost_then_running, customer)
    late.release.set()
    await prune_a

    assert customer_creates.is_fenced(customer.miner_hotkey, customer.executor_id)
    # a prune started after the fence that completes still lifts it
    assert await _fenced_after(_Prunes(None), customer) is False


def _fresh_connector(monkeypatch) -> docker_service._CustomerCreateRegistry:
    """A replaced connector: a new process, so a new registry that knows nothing of the old one's prunes."""
    registry = docker_service._CustomerCreateRegistry()
    monkeypatch.setattr(docker_service, "customer_creates", registry)
    return registry


@pytest.mark.asyncio
async def test_a_replaced_connector_proves_no_prune_runs_before_a_customer_pull(monkeypatch) -> None:
    # lio#2226 review: the old connector's prune lost its reply and may still run; its replacement must not pull
    monkeypatch.setattr(docker_service, "PRUNE_CONFIRM_ATTEMPTS", 1)
    monkeypatch.setattr(docker_service, "PRUNE_CONFIRM_INTERVAL_SECONDS", 0)
    registry = _fresh_connector(monkeypatch)
    customer = _payload()
    assert registry.is_fenced(customer.miner_hotkey, customer.executor_id)

    still_pruning = _Prunes(_prune_failure(_PRUNE_RUNNING), _prune_failure(_PRUNE_RUNNING))
    with pytest.raises(docker_service._PruneMayStillRun):
        await confirm_no_prune_runs(still_pruning, {}, customer.miner_hotkey, customer.executor_id)
    assert still_pruning.calls == 2

    # once dockerd takes a prune and completes it, none is left: the create goes on, and the next one prunes nothing
    done = _Prunes(None)
    await confirm_no_prune_runs(done, {}, customer.miner_hotkey, customer.executor_id)
    await confirm_no_prune_runs(done, {}, customer.miner_hotkey, customer.executor_id)
    assert done.calls == 1
    assert not registry.is_fenced(customer.miner_hotkey, customer.executor_id)


@pytest.mark.asyncio
async def test_a_waiting_customer_create_lifts_the_fence_with_its_own_prune(monkeypatch) -> None:
    # lio#2226 review: while the create runs no delete may prune, so the create runs the confirming prune itself
    monkeypatch.setattr(docker_service, "PRUNE_CONFIRM_ATTEMPTS", 1)
    monkeypatch.setattr(docker_service, "PRUNE_CONFIRM_INTERVAL_SECONDS", 0)
    registry = _fresh_connector(monkeypatch)
    customer = _payload()
    assert await _fenced_after(_Prunes(_prune_failure(_LOST_REPLY), _prune_failure(_LOST_REPLY)), customer)

    with registry.track(customer):
        assert not registry.begin_prune(customer.miner_hotkey, customer.executor_id)
        assert await registry.wait_for_prunes(customer, timeout_s=0.03, poll_s=0.01)
        await confirm_no_prune_runs(_Prunes(None), {}, customer.miner_hotkey, customer.executor_id)

    assert not registry.is_fenced(customer.miner_hotkey, customer.executor_id)


@pytest.mark.asyncio
async def test_an_old_connectors_prune_retry_does_not_run_beside_a_successors_pull(monkeypatch) -> None:
    # lio#2226 review 4562793819: the old connector's prune lost its reply and waits to retry; meanwhile its successor
    # confirms no prune runs and admits a renter's create, which holds the executor's create lock through its pull
    monkeypatch.setattr(docker_service, "PRUNE_CONFIRM_ATTEMPTS", 1)
    monkeypatch.setattr(docker_service, "PRUNE_CONFIRM_INTERVAL_SECONDS", 0.05)
    customer = _payload()
    locks = _CreateLocks()
    old = _Prunes(_prune_failure(_LOST_REPLY), None)
    assert customer_creates.begin_prune(customer.miner_hotkey, customer.executor_id)
    old_delete = asyncio.ensure_future(
        docker_service._prune_images_then_release(old, {}, customer.miner_hotkey, customer.executor_id, locks)
    )
    await asyncio.sleep(0.01)  # the old prune's reply is lost; it waits to retry

    successor_create = locks.executor_create_lock(customer.executor_id)
    assert await successor_create.acquire(blocking=False)
    await confirm_no_prune_runs(_Prunes(None), {}, customer.miner_hotkey, customer.executor_id)
    await old_delete  # its retry comes while the successor's pull holds the lock

    assert old.calls == 1  # the retry pruned nothing
    await successor_create.release()
