"""The backend's RecheckExecutorRequest: a rent failed on a known node for the host's reasons, the
backend hid the node and asks for its checks now.

Flag off: the connector logs the request and queues nothing, the express lane never reads the queue
and the wave runs every node's own pipeline as before. Flag on: the connector queues the request in
Redis (one entry per node), the validator's express lane runs the node's full pipeline on the
cycle's inputs with the rental probe's interval stamp dropped, and publishes the result spec-only;
at most RECHECK_MAX_IN_FLIGHT at once; a node the wave or another lane holds waits in the queue; a
request older than RECHECK_REQUEST_MAX_AGE_SECONDS is dropped; a wave that reaches a node under a
recheck takes the recheck's result as the node's own for the cycle.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from clients.compute_client import ComputeClient
from core.express_lane import EXPRESS_PUBLISHED_EVENT, RECHECK_PUBLISHED_EVENT
from core.validator import Validator
from services.miner_service import CYCLE_DONE, CYCLE_LANE, EXPRESS_LANE, RECHECK_LANE
from services.redis_service import RECHECK_REQUESTS_HASH, RedisService
from services.task.checks import rental_probe
import test_express_lane
from test_express_lane import _Harness, _job_result, _Neuron, _portal_executor, _request

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "lium_protocol") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "lium_protocol"))

from lium_protocol.recorded import recorded  # noqa: E402

MINER = "miner-a"

# the express lane's fixtures, shared
job_files_root = test_express_lane.job_files_root
rest_miner_service = test_express_lane.rest_miner_service
wallet = test_express_lane.wallet


def _recorded_recheck_message() -> dict:
    return next(
        entry["message"]
        for entry in recorded("backend_to_validator")
        if entry["expect"] == "RecheckExecutorRequest"
    )


def _redis_service(server: FakeServer | None = None) -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis(server=server) if server is not None else FakeRedis()
    service.lock = asyncio.Lock()
    return service


def _client(redis_service: RedisService) -> ComputeClient:
    client = ComputeClient.__new__(ComputeClient)
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.miner_service = MagicMock(
        redis_service=redis_service, request_validation_cycle_now=AsyncMock()
    )
    client.lock = asyncio.Lock()
    return client


def _request_for(executor_id: str, *, age_seconds: float = 5.0, miner: str = MINER) -> dict:
    return {
        "executor_id": executor_id,
        "miner_hotkey": miner,
        "reason": "CREATION_FAILED_HOST_FAULT",
        "pod_id": str(uuid4()),
        "requested_at": time.time() - age_seconds,
    }


def _recheck_harness(monkeypatch, *, express: bool = False, miners=(MINER,)) -> _Harness:
    harness = _Harness(monkeypatch, {}, [_Neuron(m) for m in miners])
    monkeypatch.setattr(harness.settings, "EXPRESS_LANE_ENABLED", express)
    monkeypatch.setattr(harness.settings, "RECHECK_ON_REQUEST_ENABLED", True)
    harness.miner_service.recheck_outcomes = {}
    return harness


# --- the connector: parse and queue ----------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_the_connector_queues_nothing(monkeypatch, caplog):
    from core.config import settings

    monkeypatch.setattr(settings, "RECHECK_ON_REQUEST_ENABLED", False)
    redis_service = _redis_service()
    client = _client(redis_service)
    stamp = f"rental_probe_ok:{_recorded_recheck_message()['executor_id']}"
    await redis_service.redis.set(stamp, str(time.time()))

    with caplog.at_level(logging.INFO):
        await client.handle_message(json.dumps(_recorded_recheck_message()))

    assert await redis_service.get_recheck_requests() == {}
    assert await redis_service.redis.get(stamp) is not None
    assert "[recheck] Request ignored, RECHECK_ON_REQUEST_ENABLED is off" in caplog.text
    assert "Invalid message received from backend" not in caplog.text


@pytest.mark.asyncio
async def test_the_backend_message_is_queued_for_the_validator_process(monkeypatch):
    """Two clients on one store: the connector writes, the validator's lane reads."""
    from core.config import settings

    monkeypatch.setattr(settings, "RECHECK_ON_REQUEST_ENABLED", True)
    server = FakeServer()
    client = _client(_redis_service(server))
    message = _recorded_recheck_message()
    stamp = f"rental_probe_ok:{message['executor_id']}"
    await _redis_service(server).redis.set(stamp, str(time.time()))

    await client.handle_message(json.dumps(message))
    await client.handle_message(json.dumps(message))  # a repeat is one recheck

    queued = await _redis_service(server).get_recheck_requests()
    assert list(queued) == [message["executor_id"]]
    request = queued[message["executor_id"]]
    assert (request["miner_hotkey"], request["pod_id"], request["reason"]) == (
        message["miner_hotkey"],
        message["pod_id"],
        "CREATION_FAILED_HOST_FAULT",
    )
    assert time.time() - request["requested_at"] < 5
    client.miner_service.request_validation_cycle_now.assert_not_awaited()
    # whichever run reaches the node next, the recheck or the wave, probes it
    assert await _redis_service(server).redis.get(stamp) is None


@pytest.mark.asyncio
async def test_an_unreadable_queue_entry_is_dropped():
    redis_service = _redis_service()
    await redis_service.redis.hset(RECHECK_REQUESTS_HASH, "broken", "{not json")
    await redis_service.queue_recheck_request(_request_for("good"))

    assert list(await redis_service.get_recheck_requests()) == ["good"]
    assert await redis_service.redis.hkeys(RECHECK_REQUESTS_HASH) == [b"good"]


# --- the express lane: run, publish, bounds -------------------------------------------------


@pytest.mark.asyncio
async def test_a_requested_node_is_rechecked_now_and_published_spec_only(
    monkeypatch, wallet, caplog
):
    node = str(uuid4())
    harness = _recheck_harness(monkeypatch)
    await harness.redis_service.redis.set(f"rental_probe_ok:{node}", str(time.time()))
    await harness.redis_service.queue_recheck_request(_request_for(node))

    with caplog.at_level(logging.INFO):
        assert await harness.tick_and_settle() == 1

    request = harness.miner_service.request_job_to_miner.await_args.kwargs
    assert request["executor_id"] == node
    assert "first_pass" not in request  # a known node: the full pipeline
    assert request["encrypted_files"] is harness.inputs.encrypted_files
    # the probe runs inside its interval: the pass on record predates the failure
    assert await harness.redis_service.redis.get(f"rental_probe_ok:{node}") is None
    (results, miner_hotkey, _), publish_kwargs = harness.miner_service.publish_machine_specs.await_args
    assert miner_hotkey == MINER and [r.executor_info.uuid for r in results] == [node]
    # the platform credits no uptime for a recheck: the cycle's own report does
    assert publish_kwargs == {"recheck": True}
    assert results[0].scored_at is None and results[0].incentive is None
    # a known node: the lane's new-node bookkeeping is untouched, and the portal is not read
    assert await harness.redis_service.get_validated_executors() == set()
    harness.portal_api.get_all_executors.assert_not_awaited()
    assert await harness.redis_service.get_recheck_requests() == {}
    assert harness.miner_service.in_flight == {} and harness.miner_service.recheck_outcomes == {}
    assert harness.lane.directories_in_use() == set()

    published = [r for r in caplog.records if r.getMessage() == RECHECK_PUBLISHED_EVENT]
    assert len(published) == 1
    extra = published[0].msg.extra
    assert extra["executor_uuid"] == node and extra["outcome"] == "passed"
    assert extra["reason"] == "CREATION_FAILED_HOST_FAULT"
    assert 5 <= extra["request_to_publish_s"] < 30
    assert not [r for r in caplog.records if r.getMessage() == EXPRESS_PUBLISHED_EVENT]


@pytest.mark.asyncio
async def test_a_failing_recheck_is_published_so_the_node_stays_hidden(monkeypatch, wallet, caplog):
    node = str(uuid4())
    harness = _recheck_harness(monkeypatch)
    harness.job_for = lambda payload, executor_id: {
        "results": [_job_result(executor_id, score=0.0)]
    }
    await harness.redis_service.queue_recheck_request(_request_for(node))

    with caplog.at_level(logging.INFO):
        assert await harness.tick_and_settle() == 1

    harness.miner_service.publish_machine_specs.assert_awaited_once()
    published = [r for r in caplog.records if r.getMessage() == RECHECK_PUBLISHED_EVENT]
    assert published[0].msg.extra["outcome"] == "failed"


@pytest.mark.asyncio
async def test_rechecks_are_bounded_and_the_rest_wait_their_turn(monkeypatch, wallet):
    harness = _recheck_harness(monkeypatch)
    monkeypatch.setattr(harness.settings, "RECHECK_MAX_IN_FLIGHT", 4)
    nodes = [str(uuid4()) for _ in range(6)]
    for age, node in zip(range(60, 0, -10), nodes):  # nodes[0] asked first
        await harness.redis_service.queue_recheck_request(_request_for(node, age_seconds=age))
    harness.release.clear()

    assert await harness.lane.tick() == 4
    running = {e for e, lane in harness.miner_service.in_flight.items() if lane == RECHECK_LANE}
    assert running == set(nodes[:4])  # oldest requests first
    assert await harness.lane.tick() == 0  # no room while the four run
    assert set(await harness.redis_service.get_recheck_requests()) == set(nodes[4:])

    harness.release.set()
    await asyncio.gather(*harness.lane._tasks)
    assert await harness.tick_and_settle() == 2
    assert harness.miner_service.publish_machine_specs.await_count == 6
    assert await harness.redis_service.get_recheck_requests() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("holder", [CYCLE_LANE, CYCLE_DONE, EXPRESS_LANE, RECHECK_LANE])
async def test_a_node_another_run_holds_waits_in_the_queue(monkeypatch, wallet, holder):
    node = str(uuid4())
    harness = _recheck_harness(monkeypatch)
    harness.miner_service.in_flight[node] = holder
    await harness.redis_service.queue_recheck_request(_request_for(node))

    assert await harness.tick_and_settle() == 0
    harness.miner_service.request_job_to_miner.assert_not_awaited()
    assert list(await harness.redis_service.get_recheck_requests()) == [node]

    del harness.miner_service.in_flight[node]  # that run ended and was published
    assert await harness.tick_and_settle() == 1
    harness.miner_service.publish_machine_specs.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_expired_request_and_one_for_an_unknown_miner_are_dropped(
    monkeypatch, wallet, caplog
):
    stale, orphan = str(uuid4()), str(uuid4())
    harness = _recheck_harness(monkeypatch)
    await harness.redis_service.queue_recheck_request(_request_for(stale, age_seconds=601))
    await harness.redis_service.queue_recheck_request(_request_for(orphan, miner="gone"))

    with caplog.at_level(logging.INFO):
        assert await harness.tick_and_settle() == 0

    harness.miner_service.request_job_to_miner.assert_not_awaited()
    assert await harness.redis_service.get_recheck_requests() == {}
    assert "[recheck] Request expired before the node was free; dropped" in caplog.text
    assert "[recheck] Miner is not among the serving opted-in miners; dropped" in caplog.text
    assert harness.miner_service.in_flight == {} and harness.miner_service.recheck_outcomes == {}


@pytest.mark.asyncio
async def test_a_node_the_miner_does_not_return_publishes_nothing(monkeypatch, wallet, caplog):
    node = str(uuid4())
    harness = _recheck_harness(monkeypatch)
    harness.job_for = lambda payload, executor_id: {"results": []}
    await harness.redis_service.queue_recheck_request(_request_for(node))

    with caplog.at_level(logging.INFO):
        assert await harness.tick_and_settle() == 1

    harness.miner_service.publish_machine_specs.assert_not_awaited()
    assert "[recheck] Miner did not return the executor; nothing published" in caplog.text
    assert harness.miner_service.in_flight == {} and harness.miner_service.recheck_outcomes == {}
    assert (
        await harness.redis_service.get_recheck_requests() == {}
    )  # taken; the backend's timeout decides


@pytest.mark.asyncio
async def test_a_recheck_that_raises_releases_everything(monkeypatch, wallet):
    node = str(uuid4())
    harness = _recheck_harness(monkeypatch)
    harness.miner_service.request_job_to_miner = AsyncMock(side_effect=RuntimeError("ssh dropped"))
    await harness.redis_service.queue_recheck_request(_request_for(node))

    assert await harness.tick_and_settle() == 1

    harness.miner_service.publish_machine_specs.assert_not_awaited()
    assert harness.miner_service.in_flight == {} and harness.miner_service.recheck_outcomes == {}
    assert harness.lane.directories_in_use() == set()


@pytest.mark.asyncio
async def test_flag_off_the_lane_never_reads_the_queue(monkeypatch, wallet):
    node = str(uuid4())
    harness = _recheck_harness(monkeypatch, express=True)
    monkeypatch.setattr(harness.settings, "RECHECK_ON_REQUEST_ENABLED", False)
    await harness.redis_service.queue_recheck_request(_request_for(node))

    assert await harness.tick_and_settle() == 0

    harness.miner_service.request_job_to_miner.assert_not_awaited()
    assert list(await harness.redis_service.get_recheck_requests()) == [node]


@pytest.mark.asyncio
async def test_rechecks_and_new_nodes_share_a_tick_under_their_own_caps(monkeypatch, wallet):
    rechecked, new_node = str(uuid4()), str(uuid4())
    harness = _recheck_harness(monkeypatch, express=True)
    harness.portal_api.get_all_executors = AsyncMock(
        return_value={MINER: [_portal_executor(new_node)]}
    )
    await harness.redis_service.queue_recheck_request(_request_for(rechecked))

    assert await harness.tick_and_settle() == 2

    asked = {
        c.kwargs["executor_id"]: c.kwargs.get("first_pass")
        for c in harness.miner_service.request_job_to_miner.await_args_list
    }
    assert asked == {rechecked: None, new_node: True}
    assert await harness.redis_service.get_validated_executors() == {new_node}
    # a new node's first report is credited as before; only the recheck carries the marker
    marked = {
        c.args[0][0].executor_info.uuid: c.kwargs.get("recheck", False)
        for c in harness.miner_service.publish_machine_specs.await_args_list
    }
    assert marked == {rechecked: True, new_node: False}


def _validator(harness: _Harness) -> Validator:
    validator = Validator.__new__(Validator)
    validator.redis_service = harness.redis_service
    validator.miner_service = harness.miner_service
    validator.default_extra = {}
    return validator


@pytest.mark.asyncio
@pytest.mark.parametrize("express", [False, True])
async def test_the_cycle_frees_the_waves_nodes_for_a_queued_recheck(monkeypatch, wallet, express):
    """Recheck on, express lane off is the config the recheck flag gives on its own: the wave still
    leaves its nodes CYCLE_DONE, so the cycle must drop them or every queued recheck waits until it
    expires and in_flight keeps every node the wave ever verified."""
    node = str(uuid4())
    harness = _recheck_harness(monkeypatch, express=express)
    harness.miner_service.in_flight[node] = CYCLE_DONE
    await harness.redis_service.queue_recheck_request(_request_for(node))
    assert await harness.tick_and_settle() == 0

    await _validator(harness).release_cycle_claims([node])

    assert harness.miner_service.in_flight == {}
    # only the express lane reads the validated set
    assert await harness.redis_service.get_validated_executors() == ({node} if express else set())
    assert await harness.tick_and_settle() == 1
    assert harness.miner_service.request_job_to_miner.await_args.kwargs["executor_id"] == node
    assert await harness.redis_service.get_recheck_requests() == {}


@pytest.mark.asyncio
async def test_flags_off_the_cycle_close_touches_nothing(monkeypatch, wallet):
    node = str(uuid4())
    harness = _recheck_harness(monkeypatch)
    monkeypatch.setattr(harness.settings, "RECHECK_ON_REQUEST_ENABLED", False)
    harness.miner_service.in_flight[node] = CYCLE_DONE

    await _validator(harness).release_cycle_claims([node])

    assert harness.miner_service.in_flight == {node: CYCLE_DONE}
    assert await harness.redis_service.get_validated_executors() == set()


# --- the wave meets a node under a recheck ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_wave_takes_the_rechecks_result_for_that_node(rest_miner_service, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", False)
    monkeypatch.setattr(settings, "RECHECK_ON_REQUEST_ENABLED", True)
    under_recheck, other = str(uuid4()), str(uuid4())
    rest_miner_service.recheck_outcomes = {
        under_recheck: asyncio.get_running_loop().create_future()
    }
    rest_miner_service.in_flight[under_recheck] = RECHECK_LANE
    rest_miner_service.miner_returns(under_recheck, other)

    wave = asyncio.create_task(_request(rest_miner_service))
    await asyncio.sleep(0.05)
    assert not wave.done()  # waits for the recheck, its other node already ran
    rechecked = _job_result(under_recheck, score=0.0)
    rechecked.job_batch_id = "2026-09-06 16:47:12"
    rest_miner_service.recheck_outcomes[under_recheck].set_result(rechecked)
    job = await wave

    verified = [
        c.kwargs["executor_info"].uuid
        for c in rest_miner_service.task_service.create_task.await_args_list
    ]
    assert verified == [other]  # one pipeline per node
    by_node = {r.executor_info.uuid: r for r in job["results"]}
    assert set(by_node) == {under_recheck, other}
    assert by_node[under_recheck].score == 0.0
    assert by_node[under_recheck].job_batch_id == "2026-09-06 16:40:00"  # the wave's batch
    assert rechecked.job_batch_id == "2026-09-06 16:47:12"  # the published copy is untouched
    assert rest_miner_service.in_flight == {under_recheck: RECHECK_LANE, other: CYCLE_DONE}


@pytest.mark.asyncio
async def test_the_wave_runs_the_node_itself_when_the_recheck_produced_nothing(
    rest_miner_service, monkeypatch
):
    from core.config import settings

    monkeypatch.setattr(settings, "RECHECK_ON_REQUEST_ENABLED", True)
    node = str(uuid4())
    outcome = asyncio.get_running_loop().create_future()
    outcome.set_result(None)
    rest_miner_service.recheck_outcomes = {node: outcome}
    rest_miner_service.miner_returns(node)

    job = await _request(rest_miner_service)

    verified = [
        c.kwargs["executor_info"].uuid
        for c in rest_miner_service.task_service.create_task.call_args_list
    ]
    assert verified == [node]
    assert [r.executor_info.uuid for r in job["results"]] == [node]


@pytest.mark.asyncio
async def test_a_slow_recheck_leaves_the_wave_room_to_run_the_node_itself(
    rest_miner_service, monkeypatch
):
    """The recheck outlasts the wave's wait and then produces nothing: the wave stops waiting at the
    executor's budget minus the room a normal pass needs, runs the node's own pipeline, and leaves
    the recheck running."""
    from core.config import settings
    from services.miner_service import executor_budget_seconds

    monkeypatch.setattr(settings, "RECHECK_ON_REQUEST_ENABLED", True)
    monkeypatch.setattr(settings, "JOB_TIME_OUT", 123)
    monkeypatch.setattr(settings, "RECHECK_WAVE_PIPELINE_ROOM_SECONDS", 2)
    assert executor_budget_seconds() == 3
    node = str(uuid4())
    outcome = asyncio.get_running_loop().create_future()
    rest_miner_service.recheck_outcomes = {node: outcome}
    rest_miner_service.in_flight[node] = RECHECK_LANE
    rest_miner_service.miner_returns(node)

    started = time.monotonic()
    job = await asyncio.wait_for(_request(rest_miner_service), timeout=5)
    waited = time.monotonic() - started

    assert 1 <= waited < 3  # capped at 3 - 2 seconds, inside the 3-second budget
    verified = [
        c.kwargs["executor_info"].uuid
        for c in rest_miner_service.task_service.create_task.call_args_list
    ]
    assert verified == [node]
    assert [r.executor_info.uuid for r in job["results"]] == [node]
    assert job["results"][0].job_batch_id == "2026-09-06 16:40:00"
    assert not outcome.done()  # the recheck is not cancelled by the wave giving up
    outcome.set_result(None)


@pytest.mark.asyncio
async def test_flags_off_the_wave_never_waits_on_a_recheck(rest_miner_service, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", False)
    monkeypatch.setattr(settings, "RECHECK_ON_REQUEST_ENABLED", False)
    node = str(uuid4())
    rest_miner_service.recheck_outcomes = {node: asyncio.get_running_loop().create_future()}
    rest_miner_service.miner_returns(node)

    job = await asyncio.wait_for(_request(rest_miner_service), timeout=5)

    assert [r.executor_info.uuid for r in job["results"]] == [node]
    assert rest_miner_service.in_flight == {}


@pytest.mark.asyncio
async def test_the_rechecks_own_run_never_waits_on_itself(rest_miner_service, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "RECHECK_ON_REQUEST_ENABLED", True)
    node = str(uuid4())
    rest_miner_service.recheck_outcomes = {node: asyncio.get_running_loop().create_future()}
    rest_miner_service.in_flight[node] = RECHECK_LANE
    rest_miner_service.miner_returns(node)

    job = await asyncio.wait_for(_request(rest_miner_service, executor_id=node), timeout=5)

    assert [r.executor_info.uuid for r in job["results"]] == [node]


# --- the rental probe runs inside its interval after the stamp is dropped -----------------------


@pytest.mark.asyncio
async def test_dropping_the_stamp_makes_the_probe_run_inside_its_interval(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "RENTAL_PROBE_INTERVAL_HOURS", 6.0)
    redis_service = _redis_service()
    node = str(uuid4())
    ctx = SimpleNamespace(
        executor=SimpleNamespace(uuid=node),
        state=SimpleNamespace(rented_data=None),
        services=SimpleNamespace(redis=redis_service),
    )
    await redis_service.redis.set(f"rental_probe_ok:{node}", str(time.time() - 60))
    assert (await rental_probe._skip_reason(ctx))[0] == "within interval"

    await rental_probe.forget_last_pass(redis_service, node)

    assert (await rental_probe._skip_reason(ctx))[0] is None
