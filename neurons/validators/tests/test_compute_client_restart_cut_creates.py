import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.compute_client import (
    CREATE_CUT_OFF_BY_RESTART_STEP,
    INFLIGHT_CREATE_MAX_AGE_SECONDS,
    ComputeClient,
    OutgoingMessages,
)
from payload_models.payloads import ContainerCreateRequest, ContainerDeleteRequest, FailedContainerRequest, WorkloadKind
from services.redis_service import InflightBackendCreate, RedisService


class _FakePipeline:
    """MULTI/EXEC for write_atomically: queues the writes, applies them at execute"""

    def __init__(self, redis) -> None:
        self.redis, self.ops = redis, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.ops.append((getattr(self.redis, name), args, kwargs))

    async def execute(self) -> None:
        for method, args, kwargs in self.ops:
            await method(*args, **kwargs)


class _FakeRedis:
    """the hash commands (and SET) RedisService uses, in memory"""

    def pipeline(self, transaction: bool = True) -> _FakePipeline:
        return _FakePipeline(self)

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.strings: dict[str, str] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.strings[key] = value

    async def hset(self, key: str, field: str, value: str) -> None:
        self.hashes.setdefault(key, {})[field] = value

    async def hget(self, key: str, field: str) -> str | None:
        return self.hashes.get(key, {}).get(field)

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        return {k.encode(): v.encode() for k, v in self.hashes.get(key, {}).items()}

    async def hdel(self, key: str, *fields: str) -> None:
        for field in fields:
            self.hashes.get(key, {}).pop(field, None)


def _redis_service() -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = _FakeRedis()
    service.lock = asyncio.Lock()
    return service


def _client(redis_service: RedisService) -> ComputeClient:
    client = ComputeClient.__new__(ComputeClient)
    client.message_queue = []
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.miner_service = MagicMock()
    client.miner_service.redis_service = redis_service
    client.resume_count_by_pod_id = {}
    client.deleted_pod_ids = set()
    client.miner_drivers = asyncio.Queue()
    client.resumed_creates_report = None
    return client


@pytest.mark.asyncio
async def test_a_create_cut_off_by_a_restart_is_reported_failed_and_its_pending_flag_cleared() -> None:
    redis_service = _redis_service()
    await redis_service.add_pending_pod("miner-hk", "exec-1", "pod-1")
    # already resumed once (DAH-5437): past MAX_CREATE_RESUMES it is reported failed
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.FILLER, resumes=1)
    client = _client(redis_service)

    await client.fail_creates_cut_off_by_restart()

    assert len(client.message_queue) == 1
    failed = client.message_queue[0]
    assert isinstance(failed, FailedContainerRequest)
    assert (failed.miner_hotkey, failed.executor_id, failed.pod_id) == ("miner-hk", "exec-1", "pod-1")
    assert failed.workload_kind == WorkloadKind.FILLER
    assert failed.failure_step == CREATE_CUT_OFF_BY_RESTART_STEP
    assert not await redis_service.renting_in_progress("miner-hk", "exec-1", "pod-1")
    # kept until the report is sent: a second restart before the WebSocket connects reports it again
    assert list(await redis_service.inflight_backend_creates()) == ["pod-1"]


@pytest.mark.asyncio
async def test_the_restart_report_untracks_the_create_once_it_is_sent():
    redis_service = _redis_service()
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.FILLER, resumes=1)
    client = _client(redis_service)
    client.message_queue = OutgoingMessages()
    client.lock = asyncio.Lock()
    client.ws = object()
    client.send_model = AsyncMock()
    await client.fail_creates_cut_off_by_restart()

    sender = asyncio.create_task(client.handle_send_messages())
    for _ in range(50):
        if not await redis_service.inflight_backend_creates():
            break
        await asyncio.sleep(0.01)
    sender.cancel()

    assert client.send_model.await_count == 1
    assert await redis_service.inflight_backend_creates() == {}


@pytest.mark.asyncio
async def test_a_finished_customer_rental_keeps_its_pending_flag_and_is_not_reported() -> None:
    # a customer create that succeeded leaves its pending-pod flag on purpose (the rental probe clears it later)
    redis_service = _redis_service()
    await redis_service.add_pending_pod("miner-hk", "exec-1", "pod-running")
    client = _client(redis_service)

    await client.fail_creates_cut_off_by_restart()

    assert client.message_queue == []
    assert await redis_service.renting_in_progress("miner-hk", "exec-1", "pod-running")


@pytest.mark.asyncio
async def test_a_create_is_tracked_while_it_runs_and_untracked_when_it_ends() -> None:
    redis_service = _redis_service()
    client = _client(redis_service)
    seen_while_running: list[dict[str, InflightBackendCreate]] = []

    async def fake_drive_miner_job(job_request: ContainerCreateRequest) -> None:
        seen_while_running.append(await redis_service.inflight_backend_creates())
        raise RuntimeError("the create failed")

    client._drive_miner_job = fake_drive_miner_job
    request = ContainerCreateRequest.model_construct(
        pod_id="pod-2", miner_hotkey="miner-hk", executor_id="exec-2", workload_kind=WorkloadKind.CUSTOMER_RENTAL
    )

    with pytest.raises(RuntimeError):
        await client.miner_driver(request)

    assert seen_while_running[0]["pod-2"].executor_id == "exec-2"
    assert await redis_service.inflight_backend_creates() == {}


class _BrokenRedis(_FakeRedis):
    async def hset(self, key: str, field: str, value: str) -> None:
        raise ConnectionError("redis is down")

    async def hdel(self, key: str, *fields: str) -> None:
        raise ConnectionError("redis is down")


@pytest.mark.asyncio
async def test_a_create_still_runs_and_returns_when_redis_cannot_track_it() -> None:
    # a lost tracking entry costs only the restart report; a lost create would leave the pod pending
    redis_service = _redis_service()
    redis_service.redis = _BrokenRedis()
    client = _client(redis_service)

    async def fake_drive_miner_job(job_request: ContainerCreateRequest) -> str:
        return "created"

    client._drive_miner_job = fake_drive_miner_job
    request = ContainerCreateRequest.model_construct(
        pod_id="pod-3", miner_hotkey="miner-hk", executor_id="exec-3", workload_kind=WorkloadKind.FILLER
    )

    assert await client.miner_driver(request) == "created"



def _create_request() -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="miner-hk",
        executor_id="exec-1",
        pod_id="pod-1",
        workload_kind=WorkloadKind.FILLER,
        docker_image="img:tag",
        gpu_uuids=["g0"],
        command_id="cmd-1",
    )


@pytest.mark.asyncio
async def test_a_create_cut_off_by_a_restart_is_listed_for_the_backend_to_send_again() -> None:
    redis_service = _redis_service()
    await redis_service.add_pending_pod("miner-hk", "exec-1", "pod-1")
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.FILLER)
    client = _client(redis_service)

    await client.fail_creates_cut_off_by_restart()

    # nothing runs here and nothing fails: the backend re-sends the create from its own records
    assert client.miner_drivers.empty()
    assert client.message_queue == []
    assert client.resumed_creates_report.pod_ids == ["pod-1"]
    assert not await redis_service.renting_in_progress("miner-hk", "exec-1", "pod-1")
    # the resume is counted in Redis, so a second restart before the re-send reports it failed
    assert (await redis_service.inflight_backend_creates())["pod-1"].resumes == 1


@pytest.mark.asyncio
async def test_the_backends_resend_is_tracked_with_its_resume_counted() -> None:
    redis_service = _redis_service()
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.FILLER)
    client = _client(redis_service)
    await client.fail_creates_cut_off_by_restart()
    seen: list[InflightBackendCreate] = []

    async def fake_drive_miner_job(job_request: ContainerCreateRequest) -> None:
        seen.append((await redis_service.inflight_backend_creates())["pod-1"])

    client._drive_miner_job = fake_drive_miner_job
    await client._run_miner_job(_create_request())

    assert seen[0].resumes == 1
    assert await redis_service.inflight_backend_creates() == {}


@pytest.mark.asyncio
async def test_a_create_already_resumed_once_is_reported_failed() -> None:
    redis_service = _redis_service()
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.FILLER, resumes=1)
    client = _client(redis_service)

    await client.fail_creates_cut_off_by_restart()

    assert [m.failure_step for m in client.message_queue] == [CREATE_CUT_OFF_BY_RESTART_STEP]
    # not listed: the backend sends it no more and fails it at once even if this failure report is lost
    assert client.resumed_creates_report.pod_ids == []


SECRETS = {
    "docker_username": "renter",
    "docker_password": "registry-pw",
    "custom_options": {"environment": {"HF_TOKEN": "env-secret"}},
    "external_volume_info": {"name": "b", "plugin": "p", "iam_user_access_key": "ak", "iam_user_secret_key": "sk-1"},
}


@pytest.mark.asyncio
async def test_the_redis_record_of_a_create_holds_ids_and_counters_only() -> None:
    # Rustam 3 Nov: no logins, passwords or tokens in Redis; the record never carries the request
    redis_service = _redis_service()
    client = _client(redis_service)
    request = ContainerCreateRequest(**{**_create_request().model_dump(), **SECRETS})
    stored_raw: list[str] = []

    async def fake_drive_miner_job(job_request: ContainerCreateRequest) -> None:
        stored_raw.append(redis_service.redis.hashes["inflight_backend_creates"]["pod-1"])

    client._drive_miner_job = fake_drive_miner_job
    await client._run_miner_job(request)

    record = json.loads(stored_raw[0])
    assert set(record) == {"miner_hotkey", "executor_id", "workload_kind", "started_at_unix_seconds", "resumes"}
    for secret in ("registry-pw", "env-secret", "sk-1", "HF_TOKEN", "environment", "docker_password"):
        assert secret not in stored_raw[0]


@pytest.mark.asyncio
async def test_an_old_record_with_the_whole_request_is_rewritten_without_it_on_start() -> None:
    redis_service = _redis_service()
    request = ContainerCreateRequest(**{**_create_request().model_dump(), **SECRETS})
    old = InflightBackendCreate(
        miner_hotkey="miner-hk", executor_id="exec-1", workload_kind=WorkloadKind.FILLER, started_at_unix_seconds=1
    ).model_dump(mode="json")
    await redis_service.redis.hset(
        "inflight_backend_creates",
        "pod-1",
        json.dumps({**old, "started_at_unix_seconds": int(time.time()), "request": request.model_dump_json()}),
    )
    client = _client(redis_service)

    await client.fail_creates_cut_off_by_restart()

    raw = redis_service.redis.hashes["inflight_backend_creates"]["pod-1"]
    assert "request" not in json.loads(raw) and "registry-pw" not in raw and "env-secret" not in raw
    assert client.resumed_creates_report.pod_ids == ["pod-1"]


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)

    async def recv(self) -> str:
        return json.dumps({"status": "success"})

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_the_resumed_creates_report_goes_first_on_the_first_socket_only() -> None:
    # Redis lost every in-flight record: the sweep still reports, with an empty list, so the backend fails the cut creates
    client = _client(_redis_service())
    client.keypair = None
    await client.fail_creates_cut_off_by_restart()
    first, second = _FakeSocket(), _FakeSocket()

    auth = MagicMock(model_dump_json=lambda: "auth")
    with patch("clients.compute_client.AuthenticateRequest.from_keypair", return_value=auth):
        await client.handle_connection(first)
        await client.handle_connection(second)

    assert first.sent[0] == "auth"
    report = json.loads(first.sent[1])
    assert (report["message_type"], report["pod_ids"], len(first.sent)) == ("ResumedCreatesReport", [], 2)
    # never again: on a later socket the backend would fail the creates it sent on the first one
    assert second.sent == ["auth"]


@pytest.mark.asyncio
async def test_a_pod_deleted_during_its_create_is_not_resumed_after_a_restart() -> None:
    redis_service = _redis_service()
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.FILLER)
    client = _client(redis_service)
    client._drive_miner_job = AsyncMock()

    await client._run_miner_job(
        ContainerDeleteRequest.model_construct(pod_id="pod-1", miner_hotkey="miner-hk", executor_id="exec-1")
    )
    await client.fail_creates_cut_off_by_restart()

    assert client.miner_drivers.empty()
    assert client.resumed_creates_report.pod_ids == []


@pytest.mark.asyncio
async def test_a_delete_before_the_creates_record_write_is_not_undone_by_it() -> None:
    from services.docker_service import inflight_creates

    redis_service = _redis_service()
    client = _client(redis_service)
    client._drive_miner_job = AsyncMock()

    with inflight_creates.track("pod-1"):  # the create is picked up, its record write still to come
        await client._run_miner_job(
            ContainerDeleteRequest.model_construct(pod_id="pod-1", miner_hotkey="miner-hk", executor_id="exec-1")
        )
        await client._track_backend_create_for_restart_report(_create_request())

    assert await redis_service.inflight_backend_creates() == {}


@pytest.mark.asyncio
async def test_a_create_whose_container_runs_is_not_resumed_after_a_restart() -> None:
    # it ended, but its record removal failed; a re-run would replace the running container by name
    redis_service = _redis_service()
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.FILLER)
    await redis_service.hset(
        "rented_machines_prefix", "1.2.3.4:22", json.dumps({"containers": [{"name": "pod_pod-1", "pod_id": "pod-1"}]})
    )
    client = _client(redis_service)

    await client.fail_creates_cut_off_by_restart()

    assert client.miner_drivers.empty()
    assert client.message_queue == []
    assert client.resumed_creates_report.pod_ids == []
    assert await redis_service.inflight_backend_creates() == {}


@pytest.mark.asyncio
async def test_a_record_older_than_the_longest_create_is_dropped_not_resumed() -> None:
    # its removal failed long ago; the pod may be running now, and a re-run would replace its container by name
    redis_service = _redis_service()
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.CUSTOMER_RENTAL)
    stale = (await redis_service.inflight_backend_creates())["pod-1"]
    stale.started_at_unix_seconds -= INFLIGHT_CREATE_MAX_AGE_SECONDS + 1
    await redis_service.redis.hset("inflight_backend_creates", "pod-1", stale.model_dump_json())
    client = _client(redis_service)

    await client.fail_creates_cut_off_by_restart()

    assert client.miner_drivers.empty()
    assert client.message_queue == []
    assert client.resumed_creates_report.pod_ids == []
    assert await redis_service.inflight_backend_creates() == {}


class _SocketFailingAfterAuth(_FakeSocket):
    async def send(self, raw: str) -> None:
        if self.sent:
            raise ConnectionError("socket closed")
        self.sent.append(raw)


@pytest.mark.asyncio
async def test_a_report_whose_send_fails_is_kept_for_the_next_socket() -> None:
    client = _client(_redis_service())
    client.keypair = None
    await client.fail_creates_cut_off_by_restart()

    auth = MagicMock(model_dump_json=lambda: "auth")
    with patch("clients.compute_client.AuthenticateRequest.from_keypair", return_value=auth):
        with pytest.raises(ConnectionError):
            await client.handle_connection(_SocketFailingAfterAuth())

    assert client.resumed_creates_report is not None
