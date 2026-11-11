"""DAH-2958: the express lane verifies never-validated executors ahead of the 15-min cycle.

Kept cases (DAH-4027 cut the rest): a never-validated executor is verified alone and published
spec-only (no incentive); a verification without its job files is never published as the node's
verdict; the portal auth blob is what signature_auth_miner verifies. The helpers are also
imported by test_rented_executor_not_listed.py.
"""

import asyncio
import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import bittensor
import pytest
import services.file_encrypt_service as file_encrypt_service
from clients.validator_portal_api import PortalExecutor, portal_miner_auth_blob
from core.express_lane import (
    EXPRESS_FAILED_EVENT,
    EXPRESS_PUBLISHED_EVENT,
    CycleInputs,
    ExpressLane,
    _LeftToLane,
    _Pending,
)
from fakeredis.aioredis import FakeRedis
from fixtures.rest_miner_fixtures import VALIDATOR_HOTKEY
from fixtures.rest_miner_fixtures import executor_info as _executor_info
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.miner_service import CYCLE_DONE, MinerService
from services.redis_service import EXPRESS_REDIS_SLOT, RedisService, _CommandLock
from services.task.models import JobResult

pytest_plugins = ["fixtures.incentive_fixtures", "fixtures.rest_miner_fixtures"]

# The first cycle since start began an hour ago; the test executors register after it.
FLEET_KNOWN_SINCE = datetime.now(UTC) - timedelta(hours=1)
# Two consecutive cycles' job_batch_id (block time at the cycle's job block, as Validator.sync()
# derives it); the backend parses it into the prod_executors row's time.
CYCLE_BATCH_ID = "2026-09-22 19:01:12"


@dataclass
class _AxonInfo:
    ip: str = "192.0.2.10"
    port: int = 8091
    is_serving: bool = True


@dataclass
class _Neuron:
    hotkey: str
    coldkey: str = "miner-coldkey"
    axon_info: _AxonInfo = field(default_factory=_AxonInfo)


def _job_result(
    executor_id: str, score: float = 1.0, job_batch_id: str = "2026-09-06 16:40:00"
) -> JobResult:
    return JobResult(
        spec={"gpu": {"count": 1}},
        executor_info=_executor_info(executor_id),
        score=score,
        job_score=score,
        job_batch_id=job_batch_id,
        log_status="info",
        log_text="Validation task completed",
        gpu_model="NVIDIA H100 80GB HBM3",
        gpu_count=1,
    )


def _portal_executor(
    executor_id: str,
    registered_seconds_ago: float | None = 40.0,
    validator: str = VALIDATOR_HOTKEY,
):
    return PortalExecutor(
        id=executor_id,
        validator_hotkey=validator,
        created_at=(
            datetime.now(UTC) - timedelta(seconds=registered_seconds_ago)
            if registered_seconds_ago is not None
            else None
        ),
    )


@pytest.fixture(autouse=True)
def job_files_root(tmp_path, monkeypatch):
    """Every cycle-* directory a test makes lands under pytest's tmp_path (the lane launches only
    on a directory that exists, so the inputs name real ones)."""
    monkeypatch.setattr(file_encrypt_service, "JOB_FILES_ROOT", tmp_path)
    return tmp_path


def _cycle_inputs(
    tmp_directory: str | None = None, job_batch_id: str = CYCLE_BATCH_ID
) -> CycleInputs:
    if tmp_directory is None:
        tmp_directory = tempfile.mkdtemp(prefix="cycle-", dir=file_encrypt_service.JOB_FILES_ROOT)
    return CycleInputs(
        encrypted_files=MinerJobEnryptedFiles(
            encrypt_key="k",
            all_keys={},
            tmp_directory=tmp_directory,
            machine_scrape_file_name="scrape",
            machine_scrape_source="",
        ),
        default_image_digests={},
        executor_image_snapshot=None,
        job_batch_id=job_batch_id,
        fleet_known_since=FLEET_KNOWN_SINCE,
    )


def _redis_service() -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis()
    service.lock = asyncio.Lock()
    return service


@pytest.fixture
def miner_task_result():
    return _job_result


def _payload() -> MinerJobRequestPayload:
    return MinerJobRequestPayload(
        job_batch_id="2026-09-06 16:40:00",
        miner_hotkey="miner-a",
        miner_coldkey="miner-coldkey",
        miner_address="192.0.2.10",
        miner_port=8091,
    )


async def _request(service: MinerService, **kwargs):
    return await service.request_job_to_miner(
        payload=_payload(),
        encrypted_files=_cycle_inputs().encrypted_files,
        rented_data=RentedExecutorsResponse(executors={}),
        default_docker_image_digests={},
        **kwargs,
    )


# --- MinerService: what the miner is asked for, and who holds an executor -----------------


@pytest.mark.asyncio
async def test_the_cycle_asks_for_every_executor(rest_miner_service):
    first, second = str(uuid4()), str(uuid4())
    rest_miner_service.miner_returns(first, second)

    job = await _request(rest_miner_service)

    submit = dict(rest_miner_service.rest_calls)["ssh-pubkey-submit"]
    assert submit["executor_id"] is None
    assert {r.executor_info.uuid for r in job["results"]} == {first, second}
    assert rest_miner_service.in_flight == {first: CYCLE_DONE, second: CYCLE_DONE}


@pytest.mark.asyncio
async def test_the_wave_leaves_a_node_the_express_lane_has_queued_to_it(rest_miner_service):
    """DAH-5601: queued, not yet running (per-miner cap, retry wait), the node still goes express:
    in the wave it waited its turn and ran the full check, 7-15 min from add to listed."""
    queued, known = str(uuid4()), str(uuid4())
    rest_miner_service.miner_returns(queued, known)
    rest_miner_service.express_queued = {queued: object()}

    job = await _request(rest_miner_service)

    assert {r.executor_info.uuid for r in job["results"]} == {known}
    assert queued not in rest_miner_service.in_flight



def _queued(executor_id: str, failed_runs: int = 0, not_before: float = 0.0) -> _Pending:
    return _Pending(
        executor=_portal_executor(executor_id),
        miner_hotkey="miner-a",
        first_seen_at=datetime.now(UTC),
        failed_runs=failed_runs,
        not_before=not_before,
    )


@pytest.mark.asyncio
async def test_the_wave_takes_a_node_whose_failed_first_pass_retry_is_due(rest_miner_service):
    """DAH-6185: the lane launches nothing for a miner the wave has not listed, and each cycle
    start marks every miner so; leaving a due retry to the lane could starve it cycle after
    cycle. The wave runs it; a retry not due yet, or a node never run, stays the lane's."""
    due, waiting, never_run = str(uuid4()), str(uuid4()), str(uuid4())
    rest_miner_service.miner_returns(due, waiting, never_run)
    rest_miner_service.express_queued = _LeftToLane(
        {
            due: _queued(due, failed_runs=1, not_before=time.monotonic() - 1),
            waiting: _queued(waiting, failed_runs=1, not_before=time.monotonic() + 60),
            never_run: _queued(never_run),
        }
    )

    job = await _request(rest_miner_service)

    assert {r.executor_info.uuid for r in job["results"]} == {due}
    assert set(rest_miner_service.express_queued) == {waiting, never_run}


# --- ExpressLane: discovery, caps, publish, retries -----------------------------------------


class _Harness:
    def __init__(
        self,
        monkeypatch,
        snapshot: dict[str, list[PortalExecutor]],
        miners: list[_Neuron],
        miner_service: MinerService | None = None,
    ):
        from core.config import settings

        self.settings = settings
        self.redis_service = _redis_service()
        if miner_service is None:
            miner_service = MinerService.__new__(MinerService)
            miner_service.in_flight = {}
            miner_service.miners_awaiting_wave_list = {}
            miner_service.express_queued = frozenset()
        self.miner_service = miner_service
        self.miner_service.publish_machine_specs = AsyncMock()
        # A self-hosted miner that does not answer its executor list (the portal-snapshot cases).
        self.miner_service.list_executor_ids = AsyncMock(return_value=None)
        self.release = asyncio.Event()
        self.release.set()

        async def request_job_to_miner(payload, executor_id, **_):
            await self.release.wait()
            return self.job_for(payload, executor_id)

        self.miner_service.request_job_to_miner = AsyncMock(side_effect=request_job_to_miner)
        # The pipeline copies the payload's job_batch_id into every JobResult (ResultHandler).
        self.job_for = lambda payload, executor_id: {
            "miner_hotkey": payload.miner_hotkey,
            "miner_coldkey": payload.miner_coldkey,
            "results": [_job_result(executor_id, job_batch_id=payload.job_batch_id)],
        }
        self.portal_api = Mock(get_all_executors=AsyncMock(return_value=snapshot))
        self.inputs: CycleInputs | None = _cycle_inputs()
        self.lane = ExpressLane(
            miner_service=self.miner_service,
            redis_service=self.redis_service,
            backend_client=Mock(
                get_all_rented_executors=AsyncMock(return_value=RentedExecutorsResponse(executors={}))
            ),
            subtensor_client=Mock(
                get_miners=AsyncMock(return_value=miners),
                hotkeys_routed_via_central_miner=frozenset(),
            ),
            cycle_inputs=lambda: self.inputs,
            portal_api=self.portal_api,
        )

    async def tick_and_settle(self) -> int:
        launched = await self.lane.tick()
        if self.lane._listing is not None:
            await self.lane._listing
        if self.lane._tasks:
            await asyncio.gather(*self.lane._tasks)
        return launched


@pytest.mark.asyncio
async def test_nothing_runs_before_the_first_cycle_has_started(monkeypatch, wallet):
    new_node = str(uuid4())
    harness = _Harness(monkeypatch, {"miner-a": [_portal_executor(new_node)]}, [_Neuron("miner-a")])
    harness.inputs = None  # Validator.express_lane_cycle_inputs returns None until the first cycle starts

    assert await harness.tick_and_settle() == 0
    harness.portal_api.get_all_executors.assert_not_awaited()
    harness.miner_service.request_job_to_miner.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_never_validated_executor_is_verified_alone_and_published_spec_only(
    monkeypatch, wallet, caplog
):
    new_node = str(uuid4())
    harness = _Harness(monkeypatch, {"miner-a": [_portal_executor(new_node, registered_seconds_ago=40)]}, [_Neuron("miner-a")])

    with caplog.at_level(logging.INFO):
        assert await harness.tick_and_settle() == 1

    request = harness.miner_service.request_job_to_miner.await_args.kwargs
    assert request["executor_id"] == new_node
    assert request["first_pass"] is True  # never scored: the DAH-3011 fast path may apply
    assert (request["payload"].miner_hotkey, request["payload"].miner_address, request["payload"].miner_port) == (
        "miner-a", "192.0.2.10", 8091
    )
    assert request["encrypted_files"] is harness.inputs.encrypted_files

    (results, miner_hotkey, miner_coldkey), _ = harness.miner_service.publish_machine_specs.await_args
    assert (miner_hotkey, miner_coldkey) == ("miner-a", "miner-coldkey")
    assert [r.executor_info.uuid for r in results] == [new_node]
    # spec-only: no incentive and no scored_at, so the backend writes no ledger row
    assert results[0].incentive is None and results[0].scored_at is None
    assert results[0].incentive_source == "unknown"
    assert await harness.redis_service.get_validated_executors() == {new_node}
    assert harness.miner_service.in_flight == {}

    published = [r for r in caplog.records if r.getMessage() == EXPRESS_PUBLISHED_EVENT]
    assert len(published) == 1
    extra = published[0].msg.extra
    assert extra["executor_uuid"] == new_node and extra["outcome"] == "passed" and extra["attempt"] == 1
    assert 40 <= extra["registration_to_publish_s"] < 60
    assert extra["first_seen_to_publish_s"] >= 0 and extra["verification_s"] >= 0


@pytest.mark.asyncio
async def test_a_restart_keeps_the_first_sighting_from_redis(monkeypatch, wallet, caplog):
    """Regression: first_seen_at lived in memory only, so every validator rollout re-stamped it and
    the logged discovery wait (registration to first sighting) grew by the whole restart."""
    new_node = str(uuid4())
    snapshot = {"miner-a": [_portal_executor(new_node, registered_seconds_ago=40)]}
    before = _Harness(monkeypatch, snapshot, [_Neuron("miner-a")])
    before.release.clear()  # the first process is cut off mid-run, before it publishes
    assert await before.lane.tick() == 1
    first_seen = before.lane._pending[new_node].first_seen_at
    for task in before.lane._tasks:
        task.cancel()
    await asyncio.gather(*before.lane._tasks, return_exceptions=True)

    after = _Harness(monkeypatch, snapshot, [_Neuron("miner-a")])
    after.redis_service = before.redis_service  # the same Redis outlives the pod
    after.lane.redis_service = before.redis_service
    with caplog.at_level(logging.INFO):
        assert await after.tick_and_settle() == 1

    assert after.lane._pending.get(new_node) is None
    published = [r for r in caplog.records if r.getMessage() == EXPRESS_PUBLISHED_EVENT]
    assert published[0].msg.extra["first_seen_at"] == first_seen.isoformat()


# --- Self-hosted miners: their new nodes are not in the portal snapshot --------------------


@pytest.mark.asyncio
async def test_a_self_hosted_miners_new_executor_is_found_by_asking_the_miner(monkeypatch, wallet):
    """A self-hosted miner's node reaches the portal snapshot only after its first publish, so the
    lane asks the miner for its list. What the first answer lists is the cycle's (like a node
    registered before fleet_known_since); a node that appears later is verified ahead of the wave."""
    old, new = str(uuid4()), str(uuid4())
    harness = _Harness(monkeypatch, {}, [_Neuron("miner-self")])
    harness.miner_service.list_executor_ids = AsyncMock(return_value=[old])

    assert await harness.tick_and_settle() == 0
    harness.miner_service.list_executor_ids.return_value = [old, new]
    harness.lane._next_self_hosted_listing = 0.0
    assert await harness.tick_and_settle() == 0  # the listing lands after this tick's selection
    assert await harness.tick_and_settle() == 1

    request = harness.miner_service.request_job_to_miner.await_args.kwargs
    assert request["executor_id"] == new and request["payload"].miner_hotkey == "miner-self"
    assert await harness.redis_service.get_validated_executors() == {new}


@pytest.mark.asyncio
async def test_the_self_hosted_listing_cannot_hammer_miners(monkeypatch, wallet):
    """Only serving miners off the central miner are asked, once per listing interval, and one that
    did not answer is skipped for the backoff."""
    miners = [
        _Neuron("miner-self"),
        _Neuron("miner-central"),
        _Neuron("miner-off", axon_info=_AxonInfo(ip="0.0.0.0", is_serving=False)),
        _Neuron("miner-down"),
    ]
    harness = _Harness(monkeypatch, {}, miners)
    harness.lane.subtensor_client.hotkeys_routed_via_central_miner = frozenset({"miner-central"})
    harness.miner_service.list_executor_ids = AsyncMock(
        side_effect=lambda miner: None if miner.hotkey == "miner-down" else []
    )

    def asked():
        hotkeys = sorted(
            c.args[0].hotkey for c in harness.miner_service.list_executor_ids.await_args_list
        )
        harness.miner_service.list_executor_ids.reset_mock()
        return hotkeys

    await harness.tick_and_settle()
    assert asked() == ["miner-down", "miner-self"]
    await harness.tick_and_settle()
    assert asked() == []  # not due yet
    harness.lane._next_self_hosted_listing = 0.0
    await harness.tick_and_settle()
    assert asked() == ["miner-self"]  # miner-down is backed off


@pytest.mark.asyncio
async def test_a_miner_that_missed_one_listing_is_asked_again_a_minute_later(monkeypatch, wallet):
    """One timeout is not a down miner: it is skipped for SELF_HOSTED_LIST_SECONDS, and the skip
    doubles only while it keeps failing, up to the failure backoff. An answer resets it."""
    harness = _Harness(monkeypatch, {}, [_Neuron("miner-flaky")])
    answers = [None, None, None, None, None, None, [], None]
    harness.miner_service.list_executor_ids = AsyncMock(side_effect=answers)

    skips = []
    for _ in answers:
        harness.lane._next_self_hosted_listing = 0.0
        harness.lane._self_hosted_backoff.pop("miner-flaky", None)
        await harness.tick_and_settle()
        until = harness.lane._self_hosted_backoff.get("miner-flaky")
        skips.append(None if until is None else round(until - time.monotonic(), -1))
    assert skips == [60, 120, 240, 480, 600, 600, None, 60]


@pytest.mark.asyncio
async def test_a_miners_executor_list_is_its_executors_route(wallet):
    """POST /executors on the miner (a DB read; installs no key), signed over our own hotkey as
    verify_simple_validator_signature checks. No answer is None, never an exception."""
    executor_id = str(uuid4())
    service = MinerService.__new__(MinerService)
    service._make_rest_request = AsyncMock(
        return_value=(
            200,
            {
                "validator_hotkey": VALIDATOR_HOTKEY,
                "executors": [{"uuid": executor_id, "address": "198.51.100.7", "port": 8001}],
            },
        )
    )

    assert await service.list_executor_ids(_Neuron("miner-self")) == [executor_id]
    call = service._make_rest_request.await_args.kwargs
    assert (call["method"], call["url"]) == ("POST", "http://192.0.2.10:8091/executors")
    assert call["json_data"] == {"signature": "0x010203", "validator_hotkey": VALIDATOR_HOTKEY}
    assert call["timeout"] <= 10
    wallet.sign.assert_called_with(VALIDATOR_HOTKEY)

    service._make_rest_request = AsyncMock(return_value=(404, {"detail": "Not Found"}))
    assert await service.list_executor_ids(_Neuron("miner-self")) is None
    service._make_rest_request = AsyncMock(side_effect=asyncio.TimeoutError)
    assert await service.list_executor_ids(_Neuron("miner-self")) is None


# --- job_batch_id: the prod_executors row's time ---------------------------------------------


@pytest.mark.asyncio
async def test_every_miner_is_asked_under_the_cycles_block_time_id(
    validator_with_mocks, create_neuron_info
):
    """Every miner is asked under the id the cycle derives from the job block. The last line
    guards the wiring the lane reads: the inputs carry that same id."""
    validator = validator_with_mocks
    validator.subtensor_client.get_miners = AsyncMock(
        return_value=[create_neuron_info(1, "miner-a"), create_neuron_info(2, "miner-b")]
    )
    validator.subtensor_client.get_time_from_block = AsyncMock(return_value=CYCLE_BATCH_ID)

    async def no_results(payload, **_):
        return {"miner_hotkey": payload.miner_hotkey, "miner_coldkey": "c", "results": []}

    validator.miner_service.request_job_to_miner = AsyncMock(side_effect=no_results)
    with (
        patch("core.validator.fetch_default_image_digests", new=AsyncMock(return_value={})),
        patch("core.validator.fetch_executor_image_digest", new=AsyncMock(return_value=None)),
    ):
        await validator.sync()

    asked = {
        c.kwargs["payload"].miner_hotkey: c.kwargs["payload"].job_batch_id
        for c in validator.miner_service.request_job_to_miner.await_args_list
    }
    assert asked == {"miner-a": CYCLE_BATCH_ID, "miner-b": CYCLE_BATCH_ID}
    assert validator.cycle_inputs.job_batch_id == CYCLE_BATCH_ID
    validator.miner_service.start_awaiting_wave_lists.assert_called_once_with(
        CYCLE_BATCH_ID, ["miner-a", "miner-b"]
    )


@pytest.mark.asyncio
async def test_a_result_produced_after_the_job_files_vanished_is_not_the_nodes_verdict(
    monkeypatch, wallet, caplog
):
    """The hold keeps the directory across cycles; if it is removed anyway while the verification
    runs, the pipeline's failure is the validator's. Nothing is published, the executor is not
    marked validated, the attempt is not counted, and the lane retries after the backoff."""
    new_node = str(uuid4())
    harness = _Harness(monkeypatch, {"miner-a": [_portal_executor(new_node)]}, [_Neuron("miner-a")])
    directory = harness.inputs.encrypted_files.tmp_directory

    def failed_job(payload, executor_id, **_):
        return {"miner_hotkey": "miner-a", "miner_coldkey": "c", "results": [_job_result(executor_id, score=0.0)]}

    async def request_job_to_miner(payload, executor_id, **_):
        shutil.rmtree(directory)  # the upload finds no files; the pipeline scores the node 0
        return failed_job(payload, executor_id)

    harness.miner_service.request_job_to_miner = AsyncMock(side_effect=request_job_to_miner)

    with caplog.at_level(logging.INFO):
        assert await harness.tick_and_settle() == 1

    harness.miner_service.publish_machine_specs.assert_not_awaited()
    assert await harness.redis_service.get_validated_executors() == set()
    assert harness.lane.directories_in_use() == set()
    assert harness.miner_service.in_flight == {}
    pending = harness.lane._pending[new_node]
    assert pending.attempts == 0 and pending.not_before > 0
    assert "[express] Job files were removed during the verification, result discarded" in caplog.text
    assert not [r for r in caplog.records if r.getMessage() == EXPRESS_PUBLISHED_EVENT]

    # negative control — the same 0 score with the files in place is the node's verdict
    harness.inputs = _cycle_inputs()
    harness.miner_service.request_job_to_miner = AsyncMock(side_effect=failed_job)
    pending.not_before = 0.0
    assert await harness.tick_and_settle() == 1
    harness.miner_service.publish_machine_specs.assert_awaited_once()
    assert harness.lane._pending[new_node].failed_runs == 1



@pytest.mark.asyncio
async def test_a_failed_first_pass_is_the_waves_once_its_retry_is_due(monkeypatch, wallet):
    """DAH-6185: after a failed first pass the node stays queued to the lane until its retry is
    due; from then a wave that lists its miner takes it (the lane may still be waiting for that
    list), so the retry is never starved."""
    new_node = str(uuid4())
    harness = _Harness(monkeypatch, {"miner-a": [_portal_executor(new_node)]}, [_Neuron("miner-a")])
    monkeypatch.setattr(harness.settings, "EXPRESS_LANE_FAILED_FIRST_PASS_RETRY_ENABLED", True)

    def failed_job(payload, executor_id, **_):
        results = [_job_result(executor_id, score=0.0)]
        return {"miner_hotkey": "miner-a", "miner_coldkey": "c", "results": results}

    harness.miner_service.request_job_to_miner = AsyncMock(side_effect=failed_job)
    assert await harness.tick_and_settle() == 1
    assert harness.lane._pending[new_node].failed_runs == 1
    assert new_node in harness.miner_service.express_queued  # retry not due: the lane's

    harness.lane._pending[new_node].not_before = time.monotonic() - 1
    assert new_node not in harness.miner_service.express_queued  # due: the wave's


@pytest.mark.asyncio
async def test_a_restart_queues_a_recently_added_unvalidated_node_again(monkeypatch, wallet):
    """A restart drops the queue and kills an express check in flight. The new process's first
    cycle starts after the node registered; the node still goes express, queued before the wave
    asks its miner, while a long-known node stays the cycle's."""
    cut_off, long_known = str(uuid4()), str(uuid4())
    snapshot = {
        "miner-a": [
            _portal_executor(cut_off, registered_seconds_ago=180),
            _portal_executor(long_known, registered_seconds_ago=7200),
        ]
    }
    harness = _Harness(monkeypatch, snapshot, [_Neuron("miner-a")])  # a fresh service instance
    harness.inputs.fleet_known_since = datetime.now(UTC)  # the first cycle since start began now
    harness.miner_service.miners_awaiting_wave_list = {"miner-a": CYCLE_BATCH_ID}

    assert await harness.tick_and_settle() == 0  # Validator.sync()'s pass before the wave asks
    assert set(harness.miner_service.express_queued) == {cut_off}

    harness.miner_service.miners_awaiting_wave_list = {}  # the wave has the miner's list
    assert await harness.tick_and_settle() == 1
    assert harness.miner_service.request_job_to_miner.await_args.kwargs["executor_id"] == cut_off
    assert await harness.redis_service.get_validated_executors() == {cut_off}


@pytest.mark.asyncio
async def test_a_failed_first_pass_logs_as_failed(monkeypatch, wallet, caplog):
    new_node = str(uuid4())
    harness = _Harness(monkeypatch, {"miner-a": [_portal_executor(new_node)]}, [_Neuron("miner-a")])
    harness.job_for = lambda payload, executor_id: {
        "miner_hotkey": "miner-a",
        "miner_coldkey": "c",
        "results": [_job_result(executor_id, score=0.0)],
    }

    with caplog.at_level(logging.INFO):
        assert await harness.tick_and_settle() == 1

    harness.miner_service.publish_machine_specs.assert_awaited_once()
    assert not [r for r in caplog.records if r.getMessage() == EXPRESS_PUBLISHED_EVENT]
    (failed,) = [r for r in caplog.records if r.getMessage() == EXPRESS_FAILED_EVENT]
    assert failed.msg.extra["outcome"] == "failed"


@pytest.mark.asyncio
async def test_a_node_that_used_up_its_express_runs_is_left_to_the_cycle_unvalidated(monkeypatch, wallet):
    """DAH-5723: recorded as validated, its wave pass skipped publish_new_executors_at_pass and waited
    for the cycle's end (6-7 min)."""
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_FAILED_FIRST_PASS_RETRY_ENABLED", False)  # one run is the last
    new_node = str(uuid4())
    harness = _Harness(monkeypatch, {"miner-a": [_portal_executor(new_node)]}, [_Neuron("miner-a")])
    harness.job_for = lambda payload, executor_id: {
        "miner_hotkey": "miner-a",
        "miner_coldkey": "c",
        "results": [_job_result(executor_id, score=0.0)],
    }

    assert await harness.tick_and_settle() == 1

    harness.miner_service.publish_machine_specs.assert_awaited_once()
    assert await harness.redis_service.get_validated_executors() == set()
    assert new_node in harness.lane._left_to_cycle and new_node not in harness.lane._pending
    assert await harness.tick_and_settle() == 0  # the lane does not run it again


# --- Redis seed and portal auth -------------------------------------------------------------


def test_the_portal_auth_blob_is_what_signature_auth_miner_verifies():
    """lium-platform apps/portal/backend/src/auth/signature_auth.py builds
    AuthenticationPayload(timestamp=..., miner_hotkey=...) and verifies json.dumps(model_dump(),
    sort_keys=True); the two repos agree only on these bytes."""
    keypair = bittensor.Keypair.create_from_uri("//LiumTestValidator")
    blob = portal_miner_auth_blob(keypair.ss58_address, 1_757_174_400)

    assert blob == json.dumps({"miner_hotkey": keypair.ss58_address, "timestamp": 1_757_174_400}, sort_keys=True)
    assert keypair.verify(blob, f"0x{keypair.sign(blob).hex()}")




@pytest.mark.asyncio
async def test_wave_publishes_a_new_executor_that_passed_without_waiting_for_the_cycle():
    """A wave result is listed as soon as its miner answers when the node is new and passed;
    a validated, failed or rented one waits for the cycle's publish."""
    service = MinerService.__new__(MinerService)
    service.redis_service = _redis_service()
    service.publish_machine_specs = AsyncMock()
    new, known, failed, rented = (str(uuid4()) for _ in range(4))
    await service.redis_service.mark_executors_validated([known])
    rented_result = _job_result(rented)
    rented_result.is_rented = True
    job = {
        "miner_hotkey": "miner-hotkey",
        "miner_coldkey": "miner-coldkey",
        "results": [_job_result(new), _job_result(known), _job_result(failed, score=0), rented_result],
    }

    await service.publish_new_executors_at_pass(job)

    service.publish_machine_specs.assert_awaited_once()
    args, kwargs = service.publish_machine_specs.await_args
    assert [r.executor_info.uuid for r in args[0]] == [new]
    assert args[1:] == ("miner-hotkey", "miner-coldkey")
    assert kwargs == {"is_whole_miner_batch": False}


@pytest.mark.asyncio
async def test_wave_publishes_nothing_at_pass_while_the_validated_set_is_cold():
    """After a fresh box or Redis wipe the validated set is empty and every executor would look
    new; the first cycle publishes them, as the express lane waits for it."""
    service = MinerService.__new__(MinerService)
    service.redis_service = _redis_service()
    service.publish_machine_specs = AsyncMock()
    job = {"miner_hotkey": "miner-hotkey", "miner_coldkey": "miner-coldkey", "results": [_job_result(str(uuid4()))]}

    await service.publish_new_executors_at_pass(job)

    service.publish_machine_specs.assert_not_awaited()


# --- The first pass's Redis commands do not queue behind the wave's ------------------------


@pytest.mark.asyncio
async def test_a_first_pass_redis_read_is_not_blocked_by_the_wave_holding_the_command_lock():
    service = _redis_service()
    service.lock = _CommandLock()
    # single commands no longer lock (DAH-5723); the two-command methods still do
    async with service.lock:  # the wave's slot is held
        wave_read = asyncio.create_task(service.first_uncached_at("wave", 1.0, 60))

        async def first_pass_read():
            EXPRESS_REDIS_SLOT.set(True)
            return await service.first_uncached_at("new", 2.0, 60)

        assert await asyncio.wait_for(asyncio.create_task(first_pass_read()), timeout=1) == 2.0
        await asyncio.sleep(0)
        assert not wave_read.done()  # the wave's own commands still wait their turn
    assert await wave_read == 1.0


@pytest.mark.asyncio
async def test_the_express_lane_runs_its_first_pass_in_the_express_redis_slot(monkeypatch, wallet):
    new_node = str(uuid4())
    harness = _Harness(monkeypatch, {"miner-a": [_portal_executor(new_node, registered_seconds_ago=40)]}, [_Neuron("miner-a")])
    slot_seen = []
    job_for = harness.job_for

    def record_slot(payload, executor_id):
        slot_seen.append(EXPRESS_REDIS_SLOT.get())
        return job_for(payload, executor_id)

    harness.job_for = record_slot

    assert await harness.tick_and_settle() == 1
    assert slot_seen == [True]
    assert EXPRESS_REDIS_SLOT.get() is False  # set in the lane's own task only


# --- DAH-5716: a provider-requested recheck of one node ------------------------------------


@pytest.mark.asyncio
async def test_a_recheck_request_is_queued_and_a_bad_id_is_ignored():
    from types import SimpleNamespace

    from clients.compute_client import ComputeClient
    from payload_models.payloads import RecheckExecutorRequest

    redis_service = _redis_service()
    client = SimpleNamespace(
        logging_extra={}, miner_service=SimpleNamespace(redis_service=redis_service)
    )
    node = str(uuid4())
    for executor_id in (node, "not-a-uuid"):
        req = RecheckExecutorRequest.model_validate_json(
            json.dumps({"message_type": "RecheckExecutorRequest", "executor_id": executor_id})
        )
        await ComputeClient.handle_recheck_executor(client, req)

    assert await redis_service.get_executor_recheck_requests() == {node}
    assert await redis_service.redis.ttl("express_lane:recheck_requests") > 0


@pytest.mark.asyncio
async def test_a_requested_validated_executor_is_rechecked_once_under_the_caps(monkeypatch, wallet):
    nodes = [str(uuid4()) for _ in range(3)]
    old = [_portal_executor(n, registered_seconds_ago=7 * 86400) for n in nodes]
    harness = _Harness(monkeypatch, {"miner-a": old}, [_Neuron("miner-a")])
    monkeypatch.setattr(harness.settings, "EXPRESS_LANE_MAX_IN_FLIGHT", 4)
    monkeypatch.setattr(harness.settings, "EXPRESS_LANE_MAX_IN_FLIGHT_PER_MINER", 2)
    await harness.redis_service.mark_executors_validated(nodes)

    assert await harness.tick_and_settle() == 0  # validated and long-known: not without a request

    for node in nodes:
        await harness.redis_service.request_executor_recheck(node)
    harness.release.clear()
    assert await harness.lane.tick() == 2  # per-miner cap
    picked = set(harness.miner_service.in_flight)
    (waiting,) = set(nodes) - picked
    assert waiting not in harness.lane._pending  # the wave does not leave a waiting recheck out
    harness.release.set()
    await asyncio.gather(*harness.lane._tasks)
    assert harness.miner_service.publish_machine_specs.await_count == 2
    assert await harness.redis_service.get_executor_recheck_requests() == {waiting}

    assert await harness.tick_and_settle() == 1
    assert await harness.redis_service.get_executor_recheck_requests() == set()
    assert await harness.tick_and_settle() == 0  # each request runs once


@pytest.mark.asyncio
async def test_a_central_miner_hotkey_runs_past_the_per_miner_cap_under_the_global_cap(monkeypatch, wallet):
    central = [str(uuid4()) for _ in range(6)]
    self_hosted = [str(uuid4()) for _ in range(3)]
    harness = _Harness(
        monkeypatch,
        {
            "miner-central": [_portal_executor(n) for n in central],
            "miner-a": [_portal_executor(n, registered_seconds_ago=60) for n in self_hosted],
        },
        [_Neuron("miner-central"), _Neuron("miner-a")],
    )
    harness.lane.subtensor_client.hotkeys_routed_via_central_miner = frozenset({"miner-central"})
    monkeypatch.setattr(harness.settings, "EXPRESS_LANE_MAX_IN_FLIGHT", 7)
    monkeypatch.setattr(harness.settings, "EXPRESS_LANE_MAX_IN_FLIGHT_PER_MINER", 2)
    harness.release.clear()

    # The self-hosted nodes are older: 2 of them (its cap), then 5 central ones fill the global 7.
    assert await harness.lane.tick() == 7
    running = set(harness.miner_service.in_flight)
    assert len(running & set(self_hosted)) == 2
    assert len(running & set(central)) == 5
    assert await harness.lane.tick() == 0  # the global cap still holds

    harness.release.set()
    await asyncio.gather(*harness.lane._tasks)
    assert await harness.tick_and_settle() == 2  # the rest: one central, one self-hosted


@pytest.mark.asyncio
async def test_a_node_is_rechecked_once_when_its_maintenance_window_ends(monkeypatch, wallet):
    node, others_node = str(uuid4()), str(uuid4())
    harness = _Harness(
        monkeypatch,
        {
            "miner-a": [
                _portal_executor(node, registered_seconds_ago=7 * 86400),
                _portal_executor(others_node, registered_seconds_ago=7 * 86400, validator="another-validator"),
            ]
        },
        [_Neuron("miner-a")],
    )
    await harness.redis_service.mark_executors_validated([node, others_node])
    ends_at = datetime.now(UTC) + timedelta(minutes=10)
    harness.inputs.maintenance_ends = {node: ends_at, others_node: ends_at}

    assert await harness.tick_and_settle() == 0  # the window is still running

    ends_at = datetime.now(UTC) - timedelta(seconds=1)
    harness.inputs.maintenance_ends = {node: ends_at, others_node: ends_at}
    assert await harness.tick_and_settle() == 1

    (results, _, _), _ = harness.miner_service.publish_machine_specs.await_args
    assert [r.executor_info.uuid for r in results] == [node]
    # still the cycle's id, so no counter moves; the backend reads the recheck's start against the window's end
    assert results[0].job_batch_id == CYCLE_BATCH_ID
    assert results[0].check_started_at > ends_at
    assert await harness.redis_service.get_executor_recheck_requests() == set()
    assert await harness.tick_and_settle() == 0  # one recheck per window


@pytest.mark.asyncio
async def test_a_recheck_never_overlaps_the_next_wave(monkeypatch, wallet):
    import core.express_lane as express_lane

    node = str(uuid4())
    harness = _Harness(
        monkeypatch,
        {"miner-a": [_portal_executor(node, registered_seconds_ago=7 * 86400)]},
        [_Neuron("miner-a")],
    )
    await harness.redis_service.mark_executors_validated([node])
    await harness.redis_service.request_executor_recheck(node)
    period = harness.settings.BLOCKS_FOR_JOB * express_lane.SECONDS_PER_BLOCK

    # Too close to the next cycle: not started, still requested.
    harness.inputs.started_at = time.monotonic() - period + 200
    assert await harness.tick_and_settle() == 0
    assert await harness.redis_service.get_executor_recheck_requests() == {node}

    # Started, but cut off before the next cycle: asked again, nothing published.
    monkeypatch.setattr(express_lane, "RECHECK_MIN_SECONDS", 0)
    harness.inputs.started_at = (
        time.monotonic() - period + express_lane.RECHECK_MARGIN_SECONDS + 0.2
    )
    harness.release.clear()
    assert await harness.tick_and_settle() == 1
    harness.miner_service.publish_machine_specs.assert_not_awaited()
    assert harness.miner_service.in_flight == {}
    assert await harness.redis_service.get_executor_recheck_requests() == {node}


@pytest.mark.asyncio
async def test_a_recheck_survives_a_deferral_and_ends_with_its_run_budget(monkeypatch, wallet):
    import core.express_lane as express_lane

    node = str(uuid4())
    harness = _Harness(
        monkeypatch,
        {"miner-a": [_portal_executor(node, registered_seconds_ago=7 * 86400)]},
        [_Neuron("miner-a")],
    )
    harness.job_for = lambda payload, executor_id: {"results": []}  # the miner did not return it
    await harness.redis_service.mark_executors_validated([node])
    await harness.redis_service.request_executor_recheck(node)

    assert await harness.tick_and_settle() == 1
    assert await harness.redis_service.get_executor_recheck_requests() == {node}
    assert node not in harness.lane._pending  # never queued: the wave keeps scoring it

    for _ in range(express_lane.RECHECK_MAX_RUNS - 1):
        assert await harness.tick_and_settle() == 1
    assert await harness.redis_service.get_executor_recheck_requests() == set()
    assert await harness.tick_and_settle() == 0
    harness.miner_service.publish_machine_specs.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_recheck_that_never_starts_keeps_its_request_and_budget(monkeypatch, wallet):
    node = str(uuid4())
    harness = _Harness(
        monkeypatch,
        {"miner-a": [_portal_executor(node, registered_seconds_ago=7 * 86400)]},
        [_Neuron("miner-a")],
    )
    await harness.redis_service.mark_executors_validated([node])
    await harness.redis_service.request_executor_recheck(node)
    rented = harness.lane.backend_client.get_all_rented_executors
    rented.return_value = None  # the launch gives up before any run

    for _ in range(3):
        assert await harness.tick_and_settle() == 0
    assert await harness.redis_service.get_executor_recheck_requests() == {node}
    assert harness.lane._recheck_runs[node] == 0

    rented.return_value = RentedExecutorsResponse(executors={})
    assert await harness.tick_and_settle() == 1
    assert await harness.redis_service.get_executor_recheck_requests() == set()


@pytest.mark.asyncio
async def test_a_failed_recheck_runs_again_when_the_first_pass_retry_is_on(monkeypatch, wallet):
    node = str(uuid4())
    harness = _Harness(
        monkeypatch,
        {"miner-a": [_portal_executor(node, registered_seconds_ago=7 * 86400)]},
        [_Neuron("miner-a")],
    )
    monkeypatch.setattr(harness.settings, "EXPRESS_LANE_FAILED_FIRST_PASS_RETRY_ENABLED", True)
    harness.job_for = lambda payload, executor_id: {
        "results": [_job_result(executor_id, score=0.0, job_batch_id=payload.job_batch_id)]
    }
    await harness.redis_service.mark_executors_validated([node])
    await harness.redis_service.request_executor_recheck(node)

    assert await harness.tick_and_settle() == 1
    assert await harness.redis_service.get_executor_recheck_requests() == {node}
    assert await harness.tick_and_settle() == 1  # the retry, the budget's last run
    assert harness.miner_service.publish_machine_specs.await_count == 2
    assert await harness.redis_service.get_executor_recheck_requests() == set()
    assert await harness.tick_and_settle() == 0
