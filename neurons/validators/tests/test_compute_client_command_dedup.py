import asyncio
import json
from fnmatch import fnmatch
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import clients.compute_client as compute_client_module
from clients.compute_client import ComputeClient, OutgoingMessages
from core.config import settings
from payload_models.payloads import (
    AddSshPublicKeyRequest,
    ExecutorRentFinishedRequest,
    ContainerCreateRequest,
    ContainerCreated,
    ContainerDeleteRequest,
    FailedContainerErrorCodes,
    FailedContainerErrorTypes,
    FailedContainerRequest,
    RemoveSshPublicKeysRequest,
    SshPubKeyAdded,
    SshPubKeyRemoved,
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
        self.values[key] = (value, float("inf") if ex is None else self.now + ex)
        return True

    async def delete(self, key):
        self.values.pop(key, None)

    async def expire(self, key, ttl):
        if await self.get(key) is not None:
            self.values[key] = (self.values[key][0], self.now + ttl)

    async def eval(self, script, numkeys, key, claim, value, ttl):
        self.updates += 1
        if value == claim and self.fail_refresh:
            self.fail_refresh = False
            raise ConnectionError("heartbeat unavailable")
        if await self.get(key) == claim:
            if value and not ttl:   # KEEPTTL
                self.values[key] = (value, self.values[key][1])
            elif value:
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

    async def sadd(self, key, member):
        self.values.setdefault(key, (set(), float("inf")))[0].add(member)

    async def srem(self, key, member):
        self.values.get(key, (set(), 0))[0].discard(member)

    async def smembers(self, key):
        return set(self.values.get(key, (set(), 0))[0])

    async def scan_iter(self, *, match, count):
        for key in list(self.values):
            if fnmatch(key, match):
                yield key

    async def exists(self, key):
        return int(key in self.values and self.values[key][1] > self.now)

    async def ttl(self, key):
        if not await self.exists(key):
            return -2
        return -1 if self.values[key][1] == float("inf") else int(self.values[key][1] - self.now)

    def pipeline(self, transaction=True):
        return FakePipeline(self)


class FakePipeline:
    """WATCH/MULTI/EXEC for the dispatch-order cleanup; this fake has no concurrent writer, so a watch never fails
    (the WatchError path runs on fakeredis in test_compute_client_restart_cut_creates.py)."""

    def __init__(self, redis):
        self.redis, self.queued = redis, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def watch(self, *keys):
        pass

    async def get(self, key):
        return await self.redis.get(key)

    async def reset(self):
        self.queued = []

    def multi(self):
        pass

    def expire(self, key, ttl):
        self.queued.append((key, ttl))

    async def execute(self):
        for key, ttl in self.queued:
            await self.redis.expire(key, ttl)
        self.queued = []


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "BACKEND_COMMAND_CLAIM_TTL_SECONDS", 3)
    monkeypatch.setattr(settings, "BACKEND_COMMAND_HEARTBEAT_SECONDS", 0.001)
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis()
    service.lock = asyncio.Lock()
    # no pending mark of an earlier run of the command (compute_client._command_ran_before)
    service.get_pending_pods = AsyncMock(return_value=[])
    client = ComputeClient.__new__(ComputeClient)
    client.message_queue = OutgoingMessages()
    client.lock = asyncio.Lock()
    client.logging_extra = {}
    client.miner_service = MagicMock(redis_service=service)
    client._write_backend_create_record = AsyncMock(return_value=True)
    client.deleted_pod_ids = set()
    client.unrecorded_create_acceptances = {}
    client.resume_by_pod_id = {}
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
async def test_create_duplicates_during_and_after_long_job(client, caplog):
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
        # far past the 60-second claim of other commands, with no heartbeat: a create's claim needs none
        redis.now += 100 * settings.BACKEND_COMMAND_CLAIM_TTL_SECONDS
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
    # a create's answer outlives every same-ID resend: it only expires once its receipt or a delete says none can come
    assert redis.values[BACKEND_COMMAND_PREFIX + "command"][1] == float("inf")


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
    "operation", ["update_backend_command", "retried_set"]
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
    else:  # a failed finish write is retried until it lands
        monkeypatch.setattr(compute_client_module, "BACKEND_CREATE_TRACKING_RETRY_SECONDS", 0)
        monkeypatch.setattr(service, operation, AsyncMock(side_effect=[ConnectionError, None]))
    await client.miner_driver(request())
    assert client.miner_service.handle_container.await_count == 1
    assert len(client.message_queue) == 1
    if operation == "update_backend_command":
        assert service.update_backend_command.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["claim_backend_command", "invalid_reply"])
async def test_a_create_without_a_readable_dedupe_record_runs_nothing_and_answers_nothing(client, monkeypatch, operation):
    """A create whose first copy ran must never run again because Redis failed on the resend, and the validator must
    not claim it did not run: it runs nothing and answers nothing, so the backend's same-ID resend settles it later."""
    service = client.miner_service.redis_service
    await client.miner_driver(request())  # the first copy runs and records its answer
    client.message_queue.clear()
    if operation == "invalid_reply":
        await service.redis.set(
            BACKEND_COMMAND_PREFIX + "command", json.dumps({"state": "finished", "responses": ["{}"]}), ex=900
        )
    else:
        monkeypatch.setattr(service, operation, AsyncMock(side_effect=ConnectionError))

    await client.miner_driver(request())

    assert client.miner_service.handle_container.await_count == 1
    assert list(client.message_queue) == []


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
@pytest.mark.parametrize(
    "answer",
    [
        SshPubKeyRemoved(miner_hotkey="miner", executor_id="executor", pod_id="pod", user_public_keys=["k"]),
        FailedContainerRequest(
            miner_hotkey="miner",
            executor_id="executor",
            pod_id="pod",
            error_type=FailedContainerErrorTypes.AddSSkeyFailed,
            msg="failed",
        ),
    ],
)
async def test_ssh_key_remove_answer_echoes_the_request_command_id(client, answer):
    client.miner_service.handle_container.return_value = answer
    remove = RemoveSshPublicKeysRequest(
        command_id="remove:1",
        miner_hotkey="miner",
        executor_id="executor",
        pod_id="pod",
        container_name="container",
        user_public_keys=["k"],
    )

    await client.miner_driver(remove)

    assert [json.loads(m.model_dump_json())["command_id"] for m in client.message_queue] == ["remove:1"]


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

        async def reconnect_before_finish(_pod_id, _create_id):
            client.message_queue.clear()  # the first socket took the frame
            await client._resend_host_fault_failures()
            assert len(client.message_queue) == int(kept)

        # the create's cleanup step: it retires the in-flight create record after the failure was kept
        client._retire_backend_create_record = AsyncMock(side_effect=reconnect_before_finish)
    await client.miner_driver(request("reboot-a"))
    if reconnect_during_cleanup:
        client._retire_backend_create_record.assert_awaited_once()
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


@pytest.mark.asyncio
async def test_a_create_resent_after_an_outage_outlasting_the_answer_ttl_replays_and_never_runs_again(client, monkeypatch):
    """The backend resends a create it got no answer for under the same command id while the pod is pending (every
    30 min). A resend refused in a Redis outage, then sent again after the ordinary answer TTL, still finds the first
    answer: it is replayed and the create does not run a second time."""
    service = client.miner_service.redis_service
    await client.miner_driver(request())   # the first copy runs; its answer is lost on the way back
    [first] = list(client.message_queue)
    client.message_queue.clear()
    claim = service.claim_backend_command
    monkeypatch.setattr(service, "claim_backend_command", AsyncMock(side_effect=ConnectionError))
    await client.miner_driver(request())   # resent during the outage: nothing runs, nothing is answered
    assert list(client.message_queue) == []
    monkeypatch.setattr(service, "claim_backend_command", claim)
    service.redis.now += settings.BACKEND_COMMAND_ANSWER_TTL_SECONDS + 30 * 60   # the backend's next resend

    await client.miner_driver(request())

    assert client.miner_service.handle_container.await_count == 1
    assert list(client.message_queue) == [first]


@pytest.mark.asyncio
async def test_the_receipt_ends_a_create_answer_like_any_other(client, monkeypatch):
    """Once the backend's receipt for the create arrives it never resends that command id: the answer then expires
    with the ordinary TTL."""
    service = client.miner_service.redis_service
    monkeypatch.setattr(service, "remove_pending_pod", AsyncMock())   # the receipt's pending-mark side, tested elsewhere
    await client.miner_driver(request())
    key = BACKEND_COMMAND_PREFIX + "command"
    assert service.redis.values[key][1] == float("inf")

    await client.miner_driver(ExecutorRentFinishedRequest(
        miner_hotkey="miner", executor_id="executor", pod_id="pod", create_command_id="command", command_id="receipt"
    ))

    assert service.redis.values[key][1] == service.redis.now + settings.BACKEND_COMMAND_ANSWER_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_create_whose_finish_write_is_lost_is_never_run_again_by_a_resend(client, monkeypatch):
    redis = client.miner_service.redis_service.redis
    response = client.miner_service.handle_container.return_value
    monkeypatch.setattr(compute_client_module, "BACKEND_CREATE_TRACKING_RETRY_SECONDS", 0)
    real_eval, lost = redis.eval, asyncio.Event()
    resent = asyncio.Event()

    async def eval_(script, numkeys, key, claim, value, ttl):
        if '"finished"' in value and not resent.is_set():
            lost.set()
            await asyncio.sleep(0)
            raise ConnectionError("redis unavailable")
        return await real_eval(script, numkeys, key, claim, value, ttl)

    monkeypatch.setattr(redis, "eval", eval_)
    task = asyncio.create_task(client.miner_driver(request()))
    await asyncio.wait_for(lost.wait(), 1)
    redis.now += 100 * settings.BACKEND_COMMAND_CLAIM_TTL_SECONDS  # the outage outlasts any short claim
    await client.miner_driver(request())   # the backend's resend while the outcome is unrecorded
    assert client.miner_service.handle_container.await_count == 1
    resent.set()
    await asyncio.wait_for(task, 1)
    await client.miner_driver(request())   # once written, a resend gets the recorded answer
    assert client.miner_service.handle_container.await_count == 1
    assert client.message_queue == [response, response]



@pytest.mark.asyncio
async def test_a_create_resent_long_after_any_bound_still_replays_until_a_receipt_or_delete(client, monkeypatch):
    """A pod whose create got no answer can stay pending (and be resent) far past any fixed bound: the answer stays
    until a delete of the pod on that node, then expires with the ordinary TTL, and a later run is a new create."""
    service = client.miner_service.redis_service
    await client.miner_driver(request())
    [first] = list(client.message_queue)
    service.redis.now += 10 * 365 * 24 * 3600
    await client.miner_driver(request())
    assert client.miner_service.handle_container.await_count == 1
    assert list(client.message_queue) == [first, first]
    monkeypatch.setattr(client, "_drive_miner_job", AsyncMock())   # the delete itself is tested elsewhere
    monkeypatch.setattr(client, "_retire_backend_create_records_of_pod", AsyncMock())

    other_node = ContainerDeleteRequest(
        miner_hotkey="miner", executor_id="moved-from", pod_id="pod", command_id="d1", container_name="c", volume_name="v"
    )
    await client.miner_driver(other_node)   # a delete on another node is not this create's cancellation
    assert service.redis.values[BACKEND_COMMAND_PREFIX + "command"][1] == float("inf")
    await client.miner_driver(
        ContainerDeleteRequest(miner_hotkey="miner", executor_id="executor", pod_id="pod", command_id="d2", container_name="c", volume_name="v")
    )
    key = BACKEND_COMMAND_PREFIX + "command"
    assert service.redis.values[key][1] == service.redis.now + settings.BACKEND_COMMAND_ANSWER_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_failed_create_is_answered_like_any_command_and_leaves_no_kept_record(client):
    """The backend sends no receipt for a failed create: its answer expires with the ordinary TTL, and it leaves the
    pod's index, so failures never accumulate kept records."""
    service = client.miner_service.redis_service
    failure = FailedContainerRequest(
        miner_hotkey="miner", executor_id="executor", pod_id="pod", msg="failed",
        error_type=FailedContainerErrorTypes.ContainerCreationFailed,
    )
    client.miner_service.handle_container.return_value = failure
    await client.miner_driver(request())
    assert service.redis.values[BACKEND_COMMAND_PREFIX + "command"][1] == (
        service.redis.now + settings.BACKEND_COMMAND_ANSWER_TTL_SECONDS
    )
    assert await service.redis.smembers("backend_creates_of_pod:pod|executor") == set()


@pytest.mark.asyncio
async def test_a_command_other_than_a_create_tries_its_finish_write_once(client, monkeypatch):
    """Only a create's outcome is retried until written; any other command's claim expires and its resend runs, so a
    lost finish write never holds its driver."""
    service = client.miner_service.redis_service
    monkeypatch.setattr(service, "update_backend_command", AsyncMock(side_effect=ConnectionError))
    monkeypatch.setattr(client, "_run_miner_job", AsyncMock())
    add = AddSshPublicKeyRequest(
        command_id="add:1", miner_hotkey="miner", executor_id="executor", pod_id="pod",
        container_name="container", user_public_keys=["k"],
    )
    await asyncio.wait_for(client.miner_driver(add), 1)
    assert service.update_backend_command.await_count == 1
