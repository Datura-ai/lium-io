import asyncio
import json
from fnmatch import fnmatch
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.compute_client import ComputeClient, OutgoingMessages
from core.config import settings
from payload_models.payloads import (
    AddSshPublicKeyRequest,
    ContainerCreateRequest,
    ContainerCreated,
    FailedContainerErrorCodes,
    FailedContainerErrorTypes,
    FailedContainerRequest,
    SshPubKeyAdded,
)
from services.redis_service import BACKEND_COMMAND_PREFIX, RedisService


class FakeRedis:
    def __init__(self):
        self.now = 0
        self.values = {}
        self.refreshed = asyncio.Event()
        self.updates = 0
        self.fail_refresh = False

    async def get(self, key):
        value, expires = self.values.get(key, (None, 0))
        return value if expires > self.now else None

    async def set(self, key, value, *, nx=False, ex):
        if nx and await self.get(key) is not None:
            return False
        self.values[key] = (value, self.now + ex)
        return True

    async def delete(self, key):
        self.values.pop(key, None)

    async def eval(self, script, numkeys, key, claim, value, ttl):
        self.updates += 1
        if value == claim and self.fail_refresh:
            self.fail_refresh = False
            raise ConnectionError("heartbeat unavailable")
        if await self.get(key) == claim:
            if value:
                await self.set(key, value, ex=ttl)
            else:
                await self.delete(key)
            if value == claim:
                self.refreshed.set()

    async def hset(self, key, field, value):
        self.values.setdefault(key, ({}, float("inf")))[0][field] = value

    async def hgetall(self, key):
        return dict(self.values.get(key, ({}, 0))[0])

    async def hdel(self, key, *fields):
        for field in fields:
            self.values.get(key, ({}, 0))[0].pop(field, None)

    async def scan_iter(self, *, match, count):
        for key in list(self.values):
            if fnmatch(key, match):
                yield key


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "BACKEND_COMMAND_CLAIM_TTL_SECONDS", 3)
    monkeypatch.setattr(settings, "BACKEND_COMMAND_HEARTBEAT_SECONDS", 0.001)
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis()
    service.lock = asyncio.Lock()
    client = ComputeClient.__new__(ComputeClient)
    client.message_queue = OutgoingMessages()
    client.lock = asyncio.Lock()
    client.logging_extra = {}
    client.miner_service = MagicMock(redis_service=service)
    client.deleted_pod_ids = set()
    client.unrecorded_create_acceptances = {}
    client._track_backend_create_for_restart_report = AsyncMock()
    client._untrack_backend_create_for_restart_report = AsyncMock()
    client.get_miner_axon_info = AsyncMock(return_value=MagicMock(ip="127.0.0.1", port=8000))
    client.miner_service.handle_container = AsyncMock(
        return_value=ContainerCreated(
            miner_hotkey="miner",
            executor_id="executor",
            pod_id="pod",
            container_name="container",
            volume_name="volume",
            port_maps=[],
        )
    )
    return client


def request(command_id="command"):
    return ContainerCreateRequest(
        command_id=command_id,
        miner_hotkey="miner",
        executor_id="executor",
        pod_id="pod",
        docker_image="image",
        gpu_uuids=[],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("heartbeat_error", [False, True])
async def test_create_duplicates_during_and_after_long_job(client, caplog, heartbeat_error):
    client.miner_service.redis_service.redis.fail_refresh = heartbeat_error
    started, finish = asyncio.Event(), asyncio.Event()
    response = client.miner_service.handle_container.return_value

    async def handle_container(job):
        started.set()
        await finish.wait()
        return response

    client.miner_service.handle_container.side_effect = handle_container
    redis = client.miner_service.redis_service.redis
    task = asyncio.create_task(client.miner_driver(request()))
    try:
        await asyncio.wait_for(started.wait(), 1)
        with caplog.at_level("INFO", logger="clients.compute_client"):
            await client.miner_driver(request())
        assert "Ignoring running backend command duplicate" in caplog.text
        assert client.message_queue == []
        for _ in range(5):
            redis.now += 1
            await asyncio.wait_for(redis.refreshed.wait(), 1)
            redis.refreshed.clear()
        await client.miner_driver(request())
        assert client.message_queue == []
    finally:
        finish.set()
        await task
    updates = redis.updates
    await asyncio.sleep(0.01)
    assert redis.updates == updates
    await client.miner_driver(request())
    assert client.miner_service.handle_container.await_count == 1
    assert client.message_queue == [response, response]
    assert redis.values[BACKEND_COMMAND_PREFIX + "command"][1] == redis.now + 900
    redis.now += 901
    await client.miner_driver(request())
    assert client.miner_service.handle_container.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [("one", "two"), ("", "")])
async def test_same_pod_with_independent_commands(client, ids):
    if not ids[0]:
        client.miner_service.redis_service = MagicMock()
    for command_id in ids:
        await client.miner_driver(request(command_id))
    assert client.miner_service.handle_container.await_count == 2
    if not ids[0]:
        # no command id, no dedup claim (the created-report keep is DAH-5380's, not dedup's)
        assert not [c for c in client.miner_service.redis_service.mock_calls if "backend_command" in c[0]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,answers",
    [(None, 0), (None, 2), (RuntimeError, 0), (RuntimeError, 2), (asyncio.CancelledError, 0)],
)
async def test_finished_replies_and_failure_without_answer_retry(client, error, answers):
    response = client.miner_service.handle_container.return_value

    async def drive(job):
        for _ in range(answers):
            client.message_queue.append(response)
        if error:
            raise error("job failed")

    client._drive_miner_job = AsyncMock(side_effect=drive)
    for _ in range(2):
        try:
            await client.miner_driver(request())
        except (RuntimeError, asyncio.CancelledError):
            assert error
    assert client._drive_miner_job.await_count == (2 if error and not answers else 1)
    assert client.message_queue == [response] * (answers * 2)


@pytest.mark.asyncio
async def test_restart_clears_only_running_claims_and_retry_answers(client):
    service = client.miner_service.redis_service
    await service.claim_backend_command("interrupted", json.dumps({"state": "running"}))
    await service.redis.set("unrelated", "keep", ex=900)
    await client.miner_driver(request("finished"))
    await service.clear_running_backend_commands()
    assert await service.redis.get(BACKEND_COMMAND_PREFIX + "interrupted") is None
    assert await service.redis.get("unrelated") == "keep"
    await client.miner_driver(request("finished"))
    await client.miner_driver(request("interrupted"))
    assert client.miner_service.handle_container.await_count == 2
    assert len(client.message_queue) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["claim_backend_command", "update_backend_command", "invalid_reply", "retried_set"]
)
async def test_redis_failure_fails_open(client, monkeypatch, operation):
    service = client.miner_service.redis_service
    if operation == "invalid_reply":
        await service.redis.set(
            BACKEND_COMMAND_PREFIX + "command",
            json.dumps({"state": "finished", "responses": ["{}"]}),
            ex=900,
        )
    elif operation == "retried_set":
        original_set = service.redis.set

        async def retried_set(*args, **kwargs):
            await original_set(*args, **kwargs)
            return False

        monkeypatch.setattr(service.redis, "set", retried_set)
    else:
        monkeypatch.setattr(service, operation, AsyncMock(side_effect=ConnectionError))
    await client.miner_driver(request())
    assert client.miner_service.handle_container.await_count == 1
    assert len(client.message_queue) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        SshPubKeyAdded(miner_hotkey="miner", executor_id="executor", pod_id="pod", user_public_keys=["k"]),
        FailedContainerRequest(
            miner_hotkey="miner",
            executor_id="executor",
            pod_id="pod",
            error_type=FailedContainerErrorTypes.AddSSkeyFailed,
            msg="failed",
        ),
    ],
)
async def test_ssh_key_add_answer_echoes_the_request_command_id(client, answer):
    client.miner_service.handle_container.return_value = answer
    add = AddSshPublicKeyRequest(
        command_id="restore:1",
        miner_hotkey="miner",
        executor_id="executor",
        pod_id="pod",
        container_name="container",
        user_public_keys=["k"],
    )

    await client.miner_driver(add)

    assert [json.loads(m.model_dump_json())["command_id"] for m in client.message_queue] == ["restore:1"]


@pytest.mark.asyncio
async def test_a_create_failure_echoes_the_create_command_id(client):
    """DAH-5592: the backend fences a creation failure on the create it answers (a late failure of a reboot the
    backend already gave up on must not fail or exempt the reboot that replaced it)."""
    client.miner_service.handle_container.return_value = FailedContainerRequest(
        miner_hotkey="miner",
        executor_id="executor",
        pod_id="pod",
        error_type=FailedContainerErrorTypes.ContainerCreationFailed,
        msg="failed",
    )

    await client.miner_driver(request("create-a"))

    assert [json.loads(m.model_dump_json())["command_id"] for m in client.message_queue] == ["create-a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "kept"),
    [(FailedContainerErrorCodes.ExecutorUnreachable, True), (FailedContainerErrorCodes.InvalidExecutorId, True),
     (FailedContainerErrorCodes.UnknownError, False)],
)
@pytest.mark.parametrize("reconnect_during_cleanup", [False, True])
async def test_a_host_fault_create_failure_is_sent_again_on_each_new_socket_for_two_hours(
    client, monkeypatch, code, kept, reconnect_during_cleanup
):
    """DAH-5592 review 5765544241: the backend may lose this failure (its database and broker down, then a restart),
    and only it stops the renter's billing; the validator sends it again on every new socket until it is too old."""
    import services.redis_service as redis_module

    client.miner_service.handle_container.return_value = FailedContainerRequest(
        miner_hotkey="miner",
        executor_id="executor",
        pod_id="pod",
        error_type=FailedContainerErrorTypes.ContainerCreationFailed,
        error_code=code,
        msg="Error: Executor unreachable",
    )
    clock = [1000.0]
    monkeypatch.setattr(redis_module.time, "time", lambda: clock[0])
    if reconnect_during_cleanup:  # review 5766355530: a socket that replaces this one before the command finishes

        async def reconnect_before_finish(_pod_id):
            client.message_queue.clear()  # the first socket took the frame
            await client._resend_host_fault_failures()
            assert len(client.message_queue) == int(kept)

        client._untrack_backend_create_for_restart_report.side_effect = reconnect_before_finish
    await client.miner_driver(request("reboot-a"))
    client.message_queue.clear()

    await client._resend_host_fault_failures()  # the next socket authenticated

    sent = [json.loads(m.model_dump_json()) for m in client.message_queue]
    assert [(m["command_id"], m["error_code"]) for m in sent] == ([("reboot-a", code.value)] if kept else [])
    client.message_queue.clear()
    clock[0] += settings.HOST_FAULT_FAILURE_RESEND_SECONDS + 1
    await client._resend_host_fault_failures()
    assert client.message_queue == []
    assert await client.miner_service.redis_service.hgetall(redis_module.HOST_FAULT_FAILURES_KEY) == {}


@pytest.mark.asyncio
async def test_a_failed_read_of_the_kept_failures_raises_so_the_next_socket_reads_them_again(client):
    """DAH-5592 review (29b7155): a Redis read error must not end the resend for this socket's lifetime; it raises,
    run_forever reconnects, and the failure goes out once Redis is back."""
    import services.redis_service as redis_module

    client.miner_service.handle_container.return_value = FailedContainerRequest(
        miner_hotkey="miner",
        executor_id="executor",
        pod_id="pod",
        error_type=FailedContainerErrorTypes.ContainerCreationFailed,
        error_code=FailedContainerErrorCodes.ExecutorUnreachable,
        msg="Error: Executor unreachable",
    )
    await client.miner_driver(request("reboot-a"))
    client.message_queue.clear()
    redis = client.miner_service.redis_service
    real_hgetall = redis.hgetall

    async def down(key):
        raise ConnectionError("redis down")

    redis.hgetall = down
    with pytest.raises(ConnectionError):
        await client._resend_host_fault_failures()
    assert client.message_queue == []

    redis.hgetall = real_hgetall  # Redis back; the reconnect authenticates a new socket
    await client._resend_host_fault_failures()
    assert [json.loads(m.model_dump_json())["command_id"] for m in client.message_queue] == ["reboot-a"]
    assert await redis.hgetall(redis_module.HOST_FAULT_FAILURES_KEY)  # still kept for later sockets


class _BackendSocket:
    """One backend process: it accepts the auth, takes every frame sent to it, then closes."""

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


async def _deliver(client, socket) -> list[dict]:
    """What handle_send_messages does while `socket` is open: send everything queued on it."""
    client.ws = socket
    while len(client.message_queue):
        await client.send_model(client.message_queue.pop(0))
    return [json.loads(raw) for raw in socket.sent if raw.startswith("{")]


@pytest.mark.asyncio
async def test_a_reboot_failure_the_backend_lost_in_a_restart_is_sent_again_on_the_new_socket(client):
    """DAH-5592 (lium-platform#2460 review): with its database and broker down the backend keeps the billing-stop for a
    failed reboot only in memory, and a restart loses it. The validator sent the failure on the old socket; the
    restarted backend's first socket gets it again, with the same command_id, so the backend records it then."""
    client.keypair = None
    client.resumed_creates_report = None
    client.miner_service.handle_container.return_value = FailedContainerRequest(
        miner_hotkey="miner",
        executor_id="executor",
        pod_id="pod",
        error_type=FailedContainerErrorTypes.ContainerCreationFailed,
        error_code=FailedContainerErrorCodes.ExecutorUnreachable,
        msg="Error: Executor unreachable",
    )
    before, after = _BackendSocket(), _BackendSocket()  # the backend process before and after its restart
    auth = MagicMock(model_dump_json=lambda: "auth")

    with patch("clients.compute_client.AuthenticateRequest.from_keypair", return_value=auth):
        await client.handle_connection(before)
        await client.miner_driver(request("reboot-a"))
        first = await _deliver(client, before)  # the backend took it but kept it only in memory, then restarted
        await client.handle_connection(after)
        again = await _deliver(client, after)

    def failures(frames):
        return [(m["command_id"], m["error_code"]) for m in frames if m.get("error_code")]

    code = FailedContainerErrorCodes.ExecutorUnreachable.value
    assert failures(first) == [("reboot-a", code)]
    assert failures(again) == [("reboot-a", code)]
