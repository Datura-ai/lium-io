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
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    ContainerDeleteRequest,
    ExecutorRentFinishedRequest,
    FailedContainerRequest,
    WorkloadKind,
)
from protocol.vc_protocol.compute_requests import Response
from services.redis_service import InflightBackendCreate, RedisService


class _FakeRedis:
    """the hash commands RedisService uses, in memory"""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, field: str, value: str) -> None:
        self.hashes.setdefault(key, {})[field] = value

    async def hget(self, key: str, field: str) -> str | None:
        return self.hashes.get(key, {}).get(field)

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        return {k.encode(): v.encode() for k, v in self.hashes.get(key, {}).items()}

    async def hdel(self, key: str, *fields: str) -> None:
        for field in fields:
            self.hashes.get(key, {}).pop(field, None)

    async def eval(self, script: str, numkeys: int, key: str, field: str, expected: str) -> int:
        # the one script RedisService runs on a hash: HDEL the field only while it still holds `expected`
        assert "HDEL" in script and numkeys == 1
        if self.hashes.get(key, {}).get(field) != expected:
            return 0
        del self.hashes[key][field]
        return 1


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
    client.report_awaiting = {}
    client.create_accepted_while_report_pending = False
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


class _ReconnectedSocket:
    """a backend socket that accepts the auth and then closes"""

    def __init__(self) -> None:
        self.send = AsyncMock()

    async def recv(self) -> str:
        return Response(status="success").model_dump_json()

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


async def _create_and_send_container_created(redis_service: RedisService, command_id: str = "") -> ComputeClient:
    client = _client(redis_service)
    client.message_queue = OutgoingMessages()
    client.lock = asyncio.Lock()
    client.keypair = MagicMock()
    client.get_miner_axon_info = AsyncMock(return_value=MagicMock(ip="1.2.3.4", port=8000))

    async def handle_container(job_request: ContainerCreateRequest) -> ContainerCreated:
        await redis_service.add_pending_pod("miner-hk", "exec-1", "pod-1", command_id=command_id)  # what the create does
        return ContainerCreated(
            miner_hotkey="miner-hk", executor_id="exec-1", pod_id="pod-1",
            container_name="c-1", volume_name="v-1", port_maps=[(22, 40022)], command_id=command_id,
        )

    client.miner_service.handle_container = handle_container
    request = ContainerCreateRequest.model_construct(
        pod_id="pod-1", miner_hotkey="miner-hk", executor_id="exec-1", workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        command_id=command_id,
    )
    await client._drive_miner_job(request)
    # the send loop takes it off the queue and the socket accepts it; the backend then restarts (1012)
    client.message_queue.pop(0)
    return client


@pytest.mark.asyncio
async def test_a_container_created_lost_to_a_backend_reconnect_is_resent_once(monkeypatch) -> None:
    monkeypatch.setattr("clients.compute_client.AuthenticateRequest.from_keypair", lambda _: Response(status="success"))
    redis_service = _redis_service()
    client = await _create_and_send_container_created(redis_service)

    await client.handle_connection(_ReconnectedSocket())

    assert len(client.message_queue) == 1
    resent = client.message_queue[0]
    assert isinstance(resent, ContainerCreated)
    assert (resent.pod_id, resent.container_name, resent.port_maps) == ("pod-1", "c-1", [(22, 40022)])

    await client.handle_connection(_ReconnectedSocket())  # still queued: not doubled
    assert len(client.message_queue) == 1


@pytest.mark.asyncio
async def test_the_rent_finished_receipt_stops_the_resend() -> None:
    redis_service = _redis_service()
    client = await _create_and_send_container_created(redis_service)

    await client._drive_miner_job(
        ExecutorRentFinishedRequest.model_construct(miner_hotkey="miner-hk", executor_id="exec-1", pod_id="pod-1")
    )
    await client.resend_unacked_container_created()

    assert client.message_queue == []
    assert await redis_service.unacked_container_created() == {}


@pytest.mark.asyncio
async def test_an_unanswered_container_created_is_dropped_once_its_pending_flag_is_gone() -> None:
    redis_service = _redis_service()
    client = await _create_and_send_container_created(redis_service)
    await redis_service.hdel("pending_pods_prefix", "miner-hk:exec-1")  # aged out / wiped on a validator start

    await client.resend_unacked_container_created()

    assert client.message_queue == []
    assert await redis_service.unacked_container_created() == {}


@pytest.mark.parametrize(
    ("saved_success", "pending", "replayed"),
    [
        # created, report unanswered: the startup sweep leaves it to the replay and does not list it
        (True, True, True),
        # no saved success: the create was cut off, listed for the backend to send again (DAH-5437)
        (False, True, False),
        # a saved success whose pending flag is gone is never replayed, so the create is listed
        (True, False, False),
    ],
)
@pytest.mark.asyncio
async def test_the_startup_sweep_leaves_a_saved_success_to_the_replay(saved_success, pending, replayed) -> None:
    redis_service = _redis_service()
    if saved_success:
        client = await _create_and_send_container_created(redis_service)
    else:
        client = _client(redis_service)
        client.message_queue = OutgoingMessages()
        client.lock = asyncio.Lock()
        await redis_service.add_pending_pod("miner-hk", "exec-1", "pod-1")
    if not pending:
        await redis_service.hdel("pending_pods_prefix", "miner-hk:exec-1")
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.CUSTOMER_RENTAL)

    await client.fail_creates_cut_off_by_restart()
    await client.resend_unacked_container_created()

    assert client.resumed_creates_report.pod_ids == ([] if replayed else ["pod-1"])
    assert [type(m) for m in client.message_queue] == ([ContainerCreated] if replayed else [])
    assert await redis_service.renting_in_progress("miner-hk", "exec-1", "pod-1") == replayed
    assert (await redis_service.unacked_container_created() != {}) == replayed
    if replayed:
        assert await redis_service.inflight_backend_creates() == {}


class _BackendSocket(_FakeSocket):
    """accepts the auth, then yields `frames` and closes"""

    def __init__(self, frames: list[str]) -> None:
        super().__init__()
        self.frames = list(frames)

    async def __anext__(self) -> str:
        if not self.frames:
            raise StopAsyncIteration
        return self.frames.pop(0)


def _receipt(create_command_id: str) -> str:
    return ExecutorRentFinishedRequest(
        miner_hotkey="miner-hk", executor_id="exec-1", pod_id="pod-1", create_command_id=create_command_id
    ).model_dump_json()


async def _client_with_a_replayed_success(command_id: str = "cmd-a") -> ComputeClient:
    redis_service = _redis_service()
    client = await _create_and_send_container_created(redis_service, command_id)
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.CUSTOMER_RENTAL)
    await client.fail_creates_cut_off_by_restart()
    client.handle_message = AsyncMock()
    client.ws = None
    return client


OTHER_FRAME = json.dumps({"message_type": "SomethingElse"})


@pytest.mark.parametrize(
    ("frames", "applied"),
    [
        # the backend applied the success and answered it: the report follows
        ([_receipt("cmd-a")], True),
        # a backend that sends receipts without ids
        ([_receipt("")], True),
        # other traffic is handled while the report waits; the report goes right after the receipt
        ([OTHER_FRAME, _receipt("cmd-a")], True),
        # its handler raised (the backend logs it and reads on): no receipt, so no report to fail the pod
        ([OTHER_FRAME], False),
        # a receipt for another dispatch of the pod does not answer this success
        ([_receipt("cmd-b")], False),
    ],
)
@pytest.mark.asyncio
async def test_the_resumed_creates_report_waits_for_the_receipt_of_each_replayed_success(frames, applied) -> None:
    client = await _client_with_a_replayed_success()
    socket = _BackendSocket(frames)

    auth = MagicMock(model_dump_json=lambda: "auth")
    with patch("clients.compute_client.AuthenticateRequest.from_keypair", return_value=auth):
        await client.handle_connection(socket)

    # the success went first, on the socket, not queued a second time
    assert socket.sent[0] == "auth"
    assert json.loads(socket.sent[1])["command_id"] == "cmd-a"
    assert client.message_queue == []
    # no wait holds other traffic back
    assert [c.args[0] for c in client.handle_message.await_args_list] == frames
    if applied:
        assert json.loads(socket.sent[2])["message_type"] == "ResumedCreatesReport"
        assert len(socket.sent) == 3
        assert client.resumed_creates_report is None
    else:
        # never sent without the receipt, however long it takes; the next socket replays and waits again
        assert len(socket.sent) == 2
        assert client.resumed_creates_report is not None


@pytest.mark.parametrize("took_a_create", [False, True])
@pytest.mark.asyncio
async def test_a_report_still_waiting_when_its_socket_closes(took_a_create) -> None:
    client = await _client_with_a_replayed_success()

    auth = MagicMock(model_dump_json=lambda: "auth")
    with patch("clients.compute_client.AuthenticateRequest.from_keypair", return_value=auth):
        await client.handle_connection(_BackendSocket([]))
        client.create_accepted_while_report_pending = took_a_create
        socket = _BackendSocket([_receipt("cmd-a")])
        await client.handle_connection(socket)

    if took_a_create:
        # the backend would fail that create as lost (sent before this socket authenticated): the report is dropped
        # and the success is replayed like any other unanswered one
        assert socket.sent == ["auth"]
        assert [type(m) for m in client.message_queue] == [ContainerCreated]
    else:
        assert [json.loads(m).get("message_type") for m in socket.sent[1:]] == ["ContainerCreated", "ResumedCreatesReport"]
    assert client.resumed_creates_report is None


@pytest.mark.parametrize(
    ("pending", "saved", "receipt", "pending_left", "saved_left"),
    [
        # the receipt answers the dispatch that is pending and saved
        (["cmd-a"], "cmd-a", "cmd-a", [], None),
        # a backend that sends no id: as before, everything for the pod goes
        (["cmd-a"], "cmd-a", "", [], None),
        # flags and a saved success written before ids: cleared by any receipt
        ([""], "", "cmd-a", [], None),
        # a late receipt for the first create while the reboot (cmd-b) is still running: the reboot keeps its flag
        (["cmd-a", "cmd-b"], "cmd-a", "cmd-a", ["cmd-b"], None),
        # ... and, once the reboot saved its success, that success too
        (["cmd-b"], "cmd-b", "cmd-a", ["cmd-b"], "cmd-b"),
    ],
)
@pytest.mark.asyncio
async def test_a_receipt_clears_only_the_pending_state_of_the_dispatch_it_names(
    pending, saved, receipt, pending_left, saved_left
) -> None:
    redis_service = _redis_service()
    for command_id in pending:
        await redis_service.add_pending_pod("miner-hk", "exec-1", "pod-1", command_id=command_id)
    created = ContainerCreated(
        miner_hotkey="miner-hk", executor_id="exec-1", pod_id="pod-1",
        container_name="c-1", volume_name="v-1", port_maps=[(22, 40022)], command_id=saved,
    )
    await redis_service.add_unacked_container_created("pod-1", created.model_dump_json())
    client = _client(redis_service)
    client.get_miner_axon_info = AsyncMock(return_value=MagicMock(ip="1.2.3.4", port=8000))

    await client._drive_miner_job(
        ExecutorRentFinishedRequest(
            miner_hotkey="miner-hk", executor_id="exec-1", pod_id="pod-1", create_command_id=receipt
        )
    )

    left = await redis_service.get_pending_pods("miner-hk", "exec-1", max_age_seconds=None)
    assert [item.get("command_id", "") for item in left] == pending_left
    unacked = await redis_service.unacked_container_created()
    assert (json.loads(unacked["pod-1"])["command_id"] if unacked else None) == saved_left


@pytest.mark.parametrize("failing_read", ["unacked_container_created", "renting_in_progress"])
@pytest.mark.asyncio
async def test_a_redis_error_in_the_replay_keeps_the_report_for_the_next_socket(failing_read) -> None:
    # sent without the replay, the report would make the backend fail a create whose container exists
    redis_service = _redis_service()
    client = await _create_and_send_container_created(redis_service)
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.CUSTOMER_RENTAL)
    await client.fail_creates_cut_off_by_restart()
    setattr(redis_service, failing_read, AsyncMock(side_effect=ConnectionError("redis down")))
    socket = _FakeSocket()

    auth = MagicMock(model_dump_json=lambda: "auth")
    with patch("clients.compute_client.AuthenticateRequest.from_keypair", return_value=auth):
        with pytest.raises(ConnectionError):
            await client.handle_connection(socket)

    assert socket.sent == ["auth"]
    assert client.resumed_creates_report is not None


@pytest.mark.parametrize(("saved", "new_dispatch", "saved_left"), [("cmd-a", "cmd-b", None), ("cmd-b", "cmd-b", "cmd-b")])
@pytest.mark.asyncio
async def test_a_new_dispatch_of_the_pod_drops_the_success_saved_for_an_older_one(saved, new_dispatch, saved_left) -> None:
    # replayed while the reboot runs, the old success would complete the reboot with the old container's ports
    redis_service = _redis_service()
    created = ContainerCreated(
        miner_hotkey="miner-hk", executor_id="exec-1", pod_id="pod-1",
        container_name="c-1", volume_name="v-1", port_maps=[(22, 40022)], command_id=saved,
    )
    await redis_service.add_unacked_container_created("pod-1", created.model_dump_json())

    await redis_service.add_pending_pod("miner-hk", "exec-1", "pod-1", command_id=new_dispatch)

    unacked = await redis_service.unacked_container_created()
    assert (json.loads(unacked["pod-1"])["command_id"] if unacked else None) == saved_left


@pytest.mark.asyncio
async def test_a_late_receipt_never_deletes_a_success_saved_after_its_check() -> None:
    redis_service = _redis_service()
    old, new = (
        ContainerCreated(
            miner_hotkey="miner-hk", executor_id="exec-1", pod_id="pod-1",
            container_name="c-1", volume_name="v-1", port_maps=[(22, port)], command_id=command_id,
        ).model_dump_json()
        for port, command_id in [(40022, "cmd-a"), (40023, "cmd-b")]
    )
    await redis_service.add_unacked_container_created("pod-1", old)
    read = redis_service.hget

    async def read_then_reboot_saves(key: str, field: str):
        saved = await read(key, field)
        await redis_service.add_unacked_container_created("pod-1", new)  # the reboot's success lands here
        return saved

    redis_service.hget = read_then_reboot_saves
    await redis_service.remove_pending_pod("miner-hk", "exec-1", "pod-1", command_id="cmd-a")

    assert (await redis_service.unacked_container_created())["pod-1"] == new
