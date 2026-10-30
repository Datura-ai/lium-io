import asyncio
import json
from fnmatch import fnmatch
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.compute_client import ComputeClient, OutgoingMessages
from core.config import settings
from payload_models.payloads import (
    AddSshPublicKeyRequest,
    ContainerCreateRequest,
    ContainerCreated,
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
    client = ComputeClient.__new__(ComputeClient)
    client.message_queue = OutgoingMessages()
    client.lock = asyncio.Lock()
    client.logging_extra = {}
    client.miner_service = MagicMock(redis_service=service)
    client._write_backend_create_record = AsyncMock(return_value=True)
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
        assert client.miner_service.redis_service.mock_calls == []


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
