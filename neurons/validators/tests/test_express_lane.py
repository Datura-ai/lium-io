"""DAH-2958: the express lane verifies never-validated executors ahead of the 15-min cycle.

Flag off: request_job_to_miner asks the miner for every executor (executor_id=None) and claims
nothing, exactly as before. Flag on: a portal executor this validator never published is verified
alone with the cycle's own pipeline inputs and published spec-only; the wave and the lane never
run on one executor at the same time; the in-flight caps bound a registration flood; an executor
the miner does not return is retried a bounded number of times and then left to the cycle.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import bittensor
import pytest
from fakeredis.aioredis import FakeRedis

from clients.validator_portal_api import PortalExecutor, ValidatorPortalAPI, portal_miner_auth_blob
from core.express_lane import EXPRESS_PUBLISHED_EVENT, MAX_ATTEMPTS, CycleInputs, ExpressLane
from datura.requests.miner_requests import AcceptSSHKeyRequest, ExecutorSSHInfo
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.miner_service import CYCLE_LANE, EXPRESS_LANE, MinerService
from services.redis_service import EXPRESS_LANE_VALIDATED_SET, RedisService
from services.task.models import JobResult

VALIDATOR_HOTKEY = "validator-hotkey"


@dataclass
class _AxonInfo:
    ip: str = "192.0.2.10"
    port: int = 8091


@dataclass
class _Neuron:
    hotkey: str
    coldkey: str = "miner-coldkey"
    axon_info: _AxonInfo = field(default_factory=_AxonInfo)


def _executor_info(executor_id: str) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=executor_id,
        address="198.51.100.7",
        port=8001,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )


def _job_result(executor_id: str, score: float = 1.0) -> JobResult:
    return JobResult(
        spec={"gpu": {"count": 1}},
        executor_info=_executor_info(executor_id),
        score=score,
        job_score=score,
        job_batch_id="2026-09-06 16:40:00",
        log_status="info",
        log_text="Validation task completed",
        gpu_model="NVIDIA H100 80GB HBM3",
        gpu_count=1,
    )


def _portal_executor(executor_id: str, registered_seconds_ago: float = 40.0, validator: str = VALIDATOR_HOTKEY):
    return PortalExecutor(
        id=executor_id,
        validator_hotkey=validator,
        created_at=datetime.now(UTC) - timedelta(seconds=registered_seconds_ago),
    )


def _cycle_inputs() -> CycleInputs:
    return CycleInputs(
        encrypted_files=MinerJobEnryptedFiles(
            encrypt_key="k",
            all_keys={},
            tmp_directory="/tmp/x",
            machine_scrape_file_name="scrape",
            machine_scrape_source="",
        ),
        default_image_digests={},
        executor_image_snapshot=None,
    )


def _redis_service() -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis()
    service.lock = asyncio.Lock()
    return service


@pytest.fixture
def wallet(mocker):
    my_key = Mock(ss58_address=VALIDATOR_HOTKEY)
    my_key.sign.return_value = b"\x01\x02\x03"
    mocker.patch(
        "core.config.Settings.get_bittensor_wallet",
        return_value=Mock(get_hotkey=Mock(return_value=my_key)),
    )
    return my_key


@pytest.fixture
def rest_miner_service(mocker, wallet, monkeypatch):
    """A MinerService whose REST boundary is mocked: the miner accepts the key and returns the
    executors the test hands it; every executor task resolves to a passing JobResult."""
    from core.config import settings

    monkeypatch.setattr(settings, "USE_REST_API", True)
    ssh_service = mocker.Mock()
    ssh_service.generate_ssh_key.return_value = (b"---PRIV---", b"ssh-ed25519 pub")
    ssh_service.decrypt_payload.return_value = "---DECRYPTED-PRIV---"
    task_service = mocker.Mock()

    async def create_task(miner_info, executor_info, **_):
        return _job_result(executor_info.uuid)

    task_service.create_task = AsyncMock(side_effect=create_task)
    service = MinerService(
        ssh_service=ssh_service,
        task_service=task_service,
        redis_service=mocker.AsyncMock(),
        attestation_service=Mock(maybe_issue_nonce=AsyncMock(return_value=None)),
    )
    mocker.patch("services.miner_service.measure_and_attach", AsyncMock())

    def miner_returns(*executor_ids: str):
        service.rest_calls = []

        async def _make_rest_request(method, url, json_data, headers, timeout, log_extra, operation_name):
            service.rest_calls.append((url.rsplit("/", 1)[-1], json_data))
            if url.endswith("ssh-pubkey-submit"):
                return 200, AcceptSSHKeyRequest(
                    executors=[_executor_info(e) for e in executor_ids]
                ).model_dump(mode="json")
            return 200, {"message_type": "SSHKeyRemoved"}

        service._make_rest_request = _make_rest_request

    service.miner_returns = miner_returns
    return service


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
async def test_flag_off_the_cycle_asks_for_every_executor_and_claims_nothing(rest_miner_service, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", False)
    first, second = str(uuid4()), str(uuid4())
    rest_miner_service.miner_returns(first, second)

    job = await _request(rest_miner_service)

    submit = dict(rest_miner_service.rest_calls)["ssh-pubkey-submit"]
    assert submit["executor_id"] is None
    assert {r.executor_info.uuid for r in job["results"]} == {first, second}
    assert rest_miner_service.in_flight == {}


@pytest.mark.asyncio
async def test_the_express_lane_asks_the_miner_for_one_executor(rest_miner_service, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", True)
    new_node = str(uuid4())
    rest_miner_service.in_flight[new_node] = EXPRESS_LANE
    rest_miner_service.miner_returns(new_node)

    job = await _request(rest_miner_service, executor_id=new_node)

    calls = dict(rest_miner_service.rest_calls)
    assert calls["ssh-pubkey-submit"]["executor_id"] == new_node
    assert calls["ssh-pubkey-remove"]["executor_id"] == new_node
    assert [r.executor_info.uuid for r in job["results"]] == [new_node]
    # the express lane, not the wave, owns the claim: the request must not touch it
    assert rest_miner_service.in_flight == {new_node: EXPRESS_LANE}


@pytest.mark.asyncio
async def test_the_wave_skips_an_executor_the_express_lane_holds_and_releases_its_own(
    rest_miner_service, monkeypatch, caplog
):
    from core.config import settings

    monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", True)
    on_express, known = str(uuid4()), str(uuid4())
    rest_miner_service.in_flight[on_express] = EXPRESS_LANE
    rest_miner_service.miner_returns(on_express, known)
    seen_during_wave = {}

    async def create_task(miner_info, executor_info, **_):
        seen_during_wave.update(rest_miner_service.in_flight)
        if executor_info.uuid == known:
            raise RuntimeError("ssh dropped")  # a failing task must still release the claim
        return _job_result(executor_info.uuid)

    rest_miner_service.task_service.create_task = AsyncMock(side_effect=create_task)

    with caplog.at_level(logging.INFO):
        job = await _request(rest_miner_service)

    verified = [c.kwargs["executor_info"].uuid for c in rest_miner_service.task_service.create_task.call_args_list]
    assert verified == [known]
    assert seen_during_wave == {on_express: EXPRESS_LANE, known: CYCLE_LANE}
    assert rest_miner_service.in_flight == {on_express: EXPRESS_LANE}
    assert job["results"] == []
    assert "Executor left to the express lane this cycle" in caplog.text


# --- ExpressLane: discovery, caps, publish, retries -----------------------------------------


class _Harness:
    def __init__(self, monkeypatch, snapshot: dict[str, list[PortalExecutor]], miners: list[_Neuron]):
        from core.config import settings

        monkeypatch.setattr(settings, "EXPRESS_LANE_ENABLED", True)
        self.settings = settings
        self.redis_service = _redis_service()
        self.miner_service = MinerService.__new__(MinerService)
        self.miner_service.in_flight = {}
        self.miner_service.publish_machine_specs = AsyncMock()
        self.release = asyncio.Event()
        self.release.set()

        async def request_job_to_miner(payload, executor_id, **_):
            await self.release.wait()
            return self.job_for(payload, executor_id)

        self.miner_service.request_job_to_miner = AsyncMock(side_effect=request_job_to_miner)
        self.job_for = lambda payload, executor_id: {
            "miner_hotkey": payload.miner_hotkey,
            "miner_coldkey": payload.miner_coldkey,
            "results": [_job_result(executor_id)],
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
async def test_nothing_runs_before_the_first_cycle_has_completed(monkeypatch, wallet):
    new_node = str(uuid4())
    harness = _Harness(monkeypatch, {"miner-a": [_portal_executor(new_node)]}, [_Neuron("miner-a")])
    harness.inputs = None  # Validator.express_lane_cycle_inputs returns None until then

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
async def test_validated_foreign_and_wave_held_executors_are_not_touched(monkeypatch, wallet):
    validated, foreign, on_wave = str(uuid4()), str(uuid4()), str(uuid4())
    harness = _Harness(
        monkeypatch,
        {
            "miner-a": [
                _portal_executor(validated),
                _portal_executor(foreign, validator="another-validator"),
                _portal_executor(on_wave),
            ]
        },
        [_Neuron("miner-a")],
    )
    await harness.redis_service.mark_executors_validated([validated])
    harness.miner_service.in_flight[on_wave] = CYCLE_LANE

    assert await harness.tick_and_settle() == 0
    harness.miner_service.request_job_to_miner.assert_not_awaited()
    harness.miner_service.publish_machine_specs.assert_not_awaited()


@pytest.mark.asyncio
async def test_in_flight_caps_bound_a_registration_flood(monkeypatch, wallet):
    nodes = {
        "miner-a": [_portal_executor(str(uuid4()), registered_seconds_ago=300 - i) for i in range(5)],
        "miner-b": [_portal_executor(str(uuid4()), registered_seconds_ago=200 - i) for i in range(3)],
        "miner-c": [_portal_executor(str(uuid4()), registered_seconds_ago=100 - i) for i in range(2)],
    }
    harness = _Harness(monkeypatch, nodes, [_Neuron("miner-a"), _Neuron("miner-b"), _Neuron("miner-c")])
    monkeypatch.setattr(harness.settings, "EXPRESS_LANE_MAX_IN_FLIGHT", 4)
    monkeypatch.setattr(harness.settings, "EXPRESS_LANE_MAX_IN_FLIGHT_PER_MINER", 2)
    harness.release.clear()  # verifications block until released

    assert await harness.lane.tick() == 4
    in_flight = harness.miner_service.in_flight
    assert all(lane == EXPRESS_LANE for lane in in_flight.values())
    by_miner = {
        hotkey: sum(e.id in in_flight for e in executors) for hotkey, executors in nodes.items()
    }
    assert by_miner == {"miner-a": 2, "miner-b": 2, "miner-c": 0}  # oldest registrations first
    # a second tick while all four are still running launches nothing more
    assert await harness.lane.tick() == 0
    assert len(in_flight) == 4

    harness.release.set()
    await asyncio.gather(*harness.lane._tasks)
    assert harness.miner_service.publish_machine_specs.await_count == 4
    # the slots are free again: the next tick takes the next four
    harness.release.clear()
    assert await harness.lane.tick() == 4
    harness.release.set()
    await asyncio.gather(*harness.lane._tasks)
    assert await harness.lane.tick() == 2
    await asyncio.gather(*harness.lane._tasks)
    assert len(await harness.redis_service.get_validated_executors()) == 10


@pytest.mark.asyncio
async def test_an_executor_the_miner_does_not_return_is_retried_then_left_to_the_cycle(
    monkeypatch, wallet, caplog
):
    new_node = str(uuid4())
    harness = _Harness(monkeypatch, {"miner-a": [_portal_executor(new_node)]}, [_Neuron("miner-a")])
    # the miner's own portal snapshot has not caught up: it accepts the key for nobody
    harness.job_for = lambda payload, executor_id: {"miner_hotkey": "miner-a", "miner_coldkey": "c", "results": []}

    with caplog.at_level(logging.INFO):
        assert await harness.tick_and_settle() == 1
        assert await harness.tick_and_settle() == 0  # inside the retry backoff
        for _ in range(MAX_ATTEMPTS - 1):
            harness.lane._pending[new_node].not_before = 0.0  # backoff elapsed
            assert await harness.tick_and_settle() == 1
        assert await harness.tick_and_settle() == 0  # given up: the normal cycle owns it now

    assert harness.miner_service.request_job_to_miner.await_count == MAX_ATTEMPTS
    harness.miner_service.publish_machine_specs.assert_not_awaited()
    assert await harness.redis_service.get_validated_executors() == set()
    assert harness.miner_service.in_flight == {}
    assert caplog.text.count("[express] Executor not verified yet, will retry") == MAX_ATTEMPTS - 1
    assert "[express] Executor left to the normal cycle" in caplog.text


@pytest.mark.asyncio
async def test_a_failed_verification_is_published_so_the_provider_sees_why(monkeypatch, wallet, caplog):
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
    assert await harness.redis_service.get_validated_executors() == {new_node}
    published = [r for r in caplog.records if r.getMessage() == EXPRESS_PUBLISHED_EVENT]
    assert published[0].msg.extra["outcome"] == "failed"


# --- Redis seed and portal auth -------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_validated_set_round_trips_through_redis():
    redis_service = _redis_service()
    ids = [str(uuid4()) for _ in range(3)]

    await redis_service.mark_executors_validated(ids)
    await redis_service.mark_executors_validated([])

    assert await redis_service.get_validated_executors() == set(ids)
    assert await redis_service.redis.scard(EXPRESS_LANE_VALIDATED_SET) == 3


def test_the_portal_auth_blob_is_what_signature_auth_miner_verifies():
    """lium-platform apps/portal/backend/src/auth/signature_auth.py builds
    AuthenticationPayload(timestamp=..., miner_hotkey=...) and verifies json.dumps(model_dump(),
    sort_keys=True); the two repos agree only on these bytes."""
    keypair = bittensor.Keypair.create_from_uri("//LiumTestValidator")
    blob = portal_miner_auth_blob(keypair.ss58_address, 1_757_174_400)

    assert blob == json.dumps({"miner_hotkey": keypair.ss58_address, "timestamp": 1_757_174_400}, sort_keys=True)
    assert keypair.verify(blob, f"0x{keypair.sign(blob).hex()}")


@pytest.mark.asyncio
async def test_get_all_executors_parses_the_bulk_snapshot_and_fails_soft(monkeypatch, wallet):
    executor_id = str(uuid4())
    responses = iter(
        [
            (200, {"miner-a": [{"id": executor_id, "validator_hotkey": VALIDATOR_HOTKEY,
                                 "created_at": "2026-09-06T16:40:00", "executor_ip_address": "198.51.100.7",
                                 "executor_ip_port": "8001", "price_per_gpu": 1.5}]}),
            (503, "portal down"),
        ]
    )
    sent_headers = {}

    class _Response:
        def __init__(self, status, body):
            self.status, self._body = status, body

        async def json(self):
            return self._body

        async def text(self):
            return str(self._body)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    class _Session:
        def __init__(self, *_, **__):
            pass

        def get(self, url, headers):
            sent_headers.update(headers)
            return _Response(*next(responses))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr("clients.validator_portal_api.aiohttp.ClientSession", _Session)

    snapshot = await ValidatorPortalAPI.get_all_executors()

    assert sent_headers["hotkey"] == VALIDATOR_HOTKEY and sent_headers["signature"] == "0x010203"
    assert list(snapshot) == ["miner-a"]
    (executor,) = snapshot["miner-a"]
    assert executor.id == executor_id and executor.validator_hotkey == VALIDATOR_HOTKEY
    assert executor.created_at == datetime(2026, 9, 6, 16, 40, tzinfo=UTC)
    assert await ValidatorPortalAPI.get_all_executors() is None
