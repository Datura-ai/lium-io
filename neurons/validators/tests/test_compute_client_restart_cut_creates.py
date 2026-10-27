import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.compute_client import CREATE_CUT_OFF_BY_RESTART_STEP, ComputeClient, OutgoingMessages
from payload_models.payloads import ContainerCreateRequest, FailedContainerRequest, WorkloadKind
from services.redis_service import PENDING_PODS_PREFIX, InflightBackendCreate, RedisService


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
    return client


@pytest.mark.asyncio
async def test_a_create_cut_off_by_a_restart_is_reported_failed_and_its_pending_flag_cleared() -> None:
    redis_service = _redis_service()
    await redis_service.add_pending_pod("miner-hk", "exec-1", "pod-1")
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.FILLER)
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
    await redis_service.add_inflight_backend_create("pod-1", "miner-hk", "exec-1", WorkloadKind.FILLER)
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
    seen_while_running: list[dict[str, list[InflightBackendCreate]]] = []

    async def fake_drive_miner_job(job_request: ContainerCreateRequest) -> None:
        seen_while_running.append(await redis_service.inflight_backend_creates())
        raise RuntimeError("the create failed")

    client._drive_miner_job = fake_drive_miner_job
    request = ContainerCreateRequest.model_construct(
        pod_id="pod-2", miner_hotkey="miner-hk", executor_id="exec-2", workload_kind=WorkloadKind.CUSTOMER_RENTAL
    )

    with pytest.raises(RuntimeError):
        await client.miner_driver(request)

    assert [create.executor_id for create in seen_while_running[0]["pod-2"]] == ["exec-2"]
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


async def _settle(condition) -> bool:
    for _ in range(50):
        if await condition():
            return True
        await asyncio.sleep(0)
    return await condition()


@pytest.mark.parametrize(
    "failing_write, newer_create",
    [
        pytest.param("insert", False, id="failed_insert_is_retried_while_the_create_runs"),
        pytest.param("remove", False, id="failed_removal_is_retried_after_the_create_ends"),
        pytest.param("remove", True, id="late_removal_never_erases_a_newer_creates_record"),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_create_record_write_is_retried_until_it_lands(monkeypatch, failing_write, newer_create) -> None:
    # the rented-pod recovery stays off a pod while this record exists: a lost insert would let it
    # race a create still running past its pending mark and lock; a lost removal would veto it forever
    monkeypatch.setattr("clients.compute_client.BACKEND_CREATE_TRACKING_RETRY_SECONDS", 0)
    redis_service = _redis_service()
    failures = [ConnectionError("redis blip")]
    if failing_write == "insert":
        real_hset = redis_service.redis.hset

        async def flaky_hset(key: str, field: str, value: str) -> None:
            if failures:
                raise failures.pop()
            await real_hset(key, field, value)

        redis_service.redis.hset = flaky_hset
    else:
        real_remove = redis_service.remove_inflight_backend_create

        async def remove_once_broken(pod_id: str, create_id: str = "") -> None:
            if failures:
                raise failures.pop()
            await real_remove(pod_id, create_id)

        redis_service.remove_inflight_backend_create = remove_once_broken
    client = _client(redis_service)
    seen_while_running: list[bool] = []

    async def fake_drive_miner_job(job_request: ContainerCreateRequest) -> str:
        seen_while_running.append(await _settle(lambda: redis_service.backend_create_inflight("pod-4")))
        return "created"

    client._drive_miner_job = fake_drive_miner_job
    request = ContainerCreateRequest.model_construct(
        pod_id="pod-4", miner_hotkey="miner-hk", executor_id="exec-4", workload_kind=WorkloadKind.CUSTOMER_RENTAL
    )

    assert await client.miner_driver(request) == "created"
    assert seen_while_running == [True]
    if newer_create:
        # a newer create of the pod writes its record before the old one's removal is retried
        await redis_service.add_inflight_backend_create(
            "pod-4", "miner-hk", "exec-4", WorkloadKind.CUSTOMER_RENTAL, "newer"
        )
        assert await _settle(_no_retirements)
        assert _create_ids(await redis_service.inflight_backend_creates(), "pod-4") == ["newer"]
    else:
        assert await _settle(lambda: _no_record(redis_service, "pod-4"))


async def _no_record(redis_service: RedisService, pod_id: str) -> bool:
    return not await redis_service.backend_create_inflight(pod_id)


async def _no_retirements() -> bool:
    from clients.compute_client import _retiring_backend_create_records

    return not _retiring_backend_create_records


@pytest.mark.parametrize(
    "removal_fails, newer_create",
    [
        pytest.param(True, False, id="failed_removal_after_the_report_is_retried"),
        pytest.param(False, True, id="report_never_erases_a_newer_creates_record"),
    ],
)
@pytest.mark.asyncio
async def test_a_restart_report_retires_only_its_own_create_record(monkeypatch, removal_fails, newer_create) -> None:
    monkeypatch.setattr("clients.compute_client.BACKEND_CREATE_TRACKING_RETRY_SECONDS", 0)
    redis_service = _redis_service()
    await redis_service.add_inflight_backend_create("pod-4", "miner-hk", "exec-4", WorkloadKind.CUSTOMER_RENTAL, "old")
    real_remove = redis_service.remove_inflight_backend_create
    failures = [ConnectionError("redis blip")] if removal_fails else []

    async def remove_once_broken(pod_id: str, create_id: str = "") -> None:
        if failures:
            raise failures.pop()
        await real_remove(pod_id, create_id)

    redis_service.remove_inflight_backend_create = remove_once_broken
    client = _client(redis_service)
    client.message_queue = OutgoingMessages()
    client.lock = asyncio.Lock()
    client.ws = object()

    async def send(_msg) -> None:
        if newer_create:  # the backend retried the pod and its create wrote a record meanwhile
            await redis_service.add_inflight_backend_create(
                "pod-4", "miner-hk", "exec-4", WorkloadKind.CUSTOMER_RENTAL, "newer"
            )

    client.send_model = AsyncMock(side_effect=send)
    await client.fail_creates_cut_off_by_restart()

    sender = asyncio.create_task(client.handle_send_messages())
    try:
        assert await _settle(lambda: _sent_and_settled(client))
    finally:
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)

    if newer_create:
        assert _create_ids(await redis_service.inflight_backend_creates(), "pod-4") == ["newer"]
    else:
        assert await _no_record(redis_service, "pod-4")


async def _sent_and_settled(client: ComputeClient) -> bool:
    return client.send_model.await_count > 0 and await _no_retirements()


def _create_ids(creates: dict[str, list[InflightBackendCreate]], pod_id: str) -> list[str]:
    return sorted(create.create_id for create in creates.get(pod_id, []))


@pytest.mark.parametrize(
    "a_ends_first, b_rejected",
    [
        pytest.param(True, False, id="first_create_ends_first"),
        pytest.param(False, False, id="second_create_ends_first"),
        pytest.param(False, True, id="overlapping_create_rejected"),
    ],
)
@pytest.mark.asyncio
async def test_overlapping_creates_of_a_pod_keep_it_excluded_until_every_one_ends(a_ends_first, b_rejected) -> None:
    redis_service = _redis_service()
    client = _client(redis_service)
    gates = {"a": asyncio.Event(), "b": asyncio.Event()}

    async def fake_drive_miner_job(job_request: ContainerCreateRequest) -> str:
        if job_request.executor_id == "b" and b_rejected:
            raise RuntimeError("renting in progress")
        await gates[job_request.executor_id].wait()
        return "created"

    client._drive_miner_job = fake_drive_miner_job

    def run(name: str) -> asyncio.Task:
        request = ContainerCreateRequest.model_construct(
            pod_id="pod-5", miner_hotkey="miner-hk", executor_id=name, workload_kind=WorkloadKind.CUSTOMER_RENTAL
        )
        return asyncio.create_task(client.miner_driver(request))

    a = run("a")
    await asyncio.sleep(0)
    b = run("b")
    if b_rejected:
        with pytest.raises(RuntimeError):
            await b
        assert await redis_service.backend_create_inflight("pod-5")  # A still runs
        gates["a"].set()
        await a
    else:
        first, second = ("a", "b") if a_ends_first else ("b", "a")
        gates[first].set()
        await asyncio.sleep(0.01)
        assert await redis_service.backend_create_inflight("pod-5")  # the other one still runs
        gates[second].set()
        await asyncio.gather(a, b)
    assert not await redis_service.backend_create_inflight("pod-5")


class _RedisFailingOnce(_FakeRedis):
    def __init__(self, failing: str) -> None:
        super().__init__()
        self.failing = failing
        self.pending_pod_removals = 0

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        if self.failing == "enumeration":
            self.failing = ""
            raise ConnectionError("redis blip")
        return await super().hgetall(key)

    def _pending_pod_write(self, key: str) -> None:
        # each remove_pending_pod rewrites the executor's pending list: fail the second, after one landed
        if self.failing == "second_cleanup" and key == PENDING_PODS_PREFIX:
            self.pending_pod_removals += 1
            if self.pending_pod_removals == 2:
                self.failing = ""
                raise ConnectionError("redis blip")

    async def hset(self, key: str, field: str, value: str) -> None:
        self._pending_pod_write(key)
        await super().hset(key, field, value)

    async def hdel(self, key: str, *fields: str) -> None:
        self._pending_pod_write(key)
        await super().hdel(key, *fields)


@pytest.mark.parametrize("failing", ["enumeration", "second_cleanup"])
@pytest.mark.asyncio
async def test_a_failed_startup_sweep_is_retried_and_reports_each_create_once(monkeypatch, failing) -> None:
    # a dead create's record left by a failed sweep would keep the rented-pod recovery off its pod for good
    monkeypatch.setattr("clients.compute_client.BACKEND_CREATE_TRACKING_RETRY_SECONDS", 0)
    redis_service = _redis_service()
    for pod_id in ("pod-6", "pod-7"):
        await redis_service.add_pending_pod("miner-hk", "exec-6", pod_id)
        await redis_service.add_inflight_backend_create(pod_id, "miner-hk", "exec-6", WorkloadKind.FILLER, pod_id)
    redis_service.redis.__class__ = _RedisFailingOnce
    redis_service.redis.failing, redis_service.redis.pending_pod_removals = failing, 0
    client = _client(redis_service)

    await client.fail_creates_cut_off_by_restart_until_done()

    assert redis_service.redis.failing == ""  # the failure did happen
    assert sorted(report.pod_id for report in client.message_queue) == ["pod-6", "pod-7"]
    assert client._restart_reported_create_ids == {"pod-6": ["pod-6"], "pod-7": ["pod-7"]}
    for pod_id in ("pod-6", "pod-7"):
        assert not await redis_service.renting_in_progress("miner-hk", "exec-6", pod_id)
