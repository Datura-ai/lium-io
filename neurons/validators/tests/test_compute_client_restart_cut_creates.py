import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.compute_client import CREATE_CUT_OFF_BY_RESTART_STEP, ComputeClient, OutgoingMessages
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
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


async def _create_and_send_container_created(redis_service: RedisService) -> ComputeClient:
    client = _client(redis_service)
    client.message_queue = OutgoingMessages()
    client.lock = asyncio.Lock()
    client.keypair = MagicMock()
    client.get_miner_axon_info = AsyncMock(return_value=MagicMock(ip="1.2.3.4", port=8000))

    async def handle_container(job_request: ContainerCreateRequest) -> ContainerCreated:
        await redis_service.add_pending_pod("miner-hk", "exec-1", "pod-1")  # what the create does
        return ContainerCreated(
            miner_hotkey="miner-hk", executor_id="exec-1", pod_id="pod-1",
            container_name="c-1", volume_name="v-1", port_maps=[(22, 40022)],
        )

    client.miner_service.handle_container = handle_container
    request = ContainerCreateRequest.model_construct(
        pod_id="pod-1", miner_hotkey="miner-hk", executor_id="exec-1", workload_kind=WorkloadKind.CUSTOMER_RENTAL
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
