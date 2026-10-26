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
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import bittensor
import pytest
import services.file_encrypt_service as file_encrypt_service
from clients.validator_portal_api import PortalExecutor, portal_miner_auth_blob
from core.express_lane import (
    EXPRESS_PUBLISHED_EVENT,
    CycleInputs,
    ExpressLane,
)
from fakeredis.aioredis import FakeRedis
from fixtures.rest_miner_fixtures import VALIDATOR_HOTKEY
from fixtures.rest_miner_fixtures import executor_info as _executor_info
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.miner_service import CYCLE_DONE, MinerService
from services.redis_service import RedisService
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
        self.miner_service = miner_service
        self.miner_service.publish_machine_specs = AsyncMock()
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
            subtensor_client=Mock(get_miners=AsyncMock(return_value=miners)),
            cycle_inputs=lambda: self.inputs,
            portal_api=self.portal_api,
        )

    async def tick_and_settle(self) -> int:
        launched = await self.lane.tick()
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


# --- Redis seed and portal auth -------------------------------------------------------------


def test_the_portal_auth_blob_is_what_signature_auth_miner_verifies():
    """lium-platform apps/portal/backend/src/auth/signature_auth.py builds
    AuthenticationPayload(timestamp=..., miner_hotkey=...) and verifies json.dumps(model_dump(),
    sort_keys=True); the two repos agree only on these bytes."""
    keypair = bittensor.Keypair.create_from_uri("//LiumTestValidator")
    blob = portal_miner_auth_blob(keypair.ss58_address, 1_757_174_400)

    assert blob == json.dumps({"miner_hotkey": keypair.ss58_address, "timestamp": 1_757_174_400}, sort_keys=True)
    assert keypair.verify(blob, f"0x{keypair.sign(blob).hex()}")


