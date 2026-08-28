"""DAH-2792: the validator -> connector -> backend path carries delivery stamps so the backend
can measure how long a message waited at each hop, and the connector opens the websocket with
limits that survive a whole scoring cycle pushed in one burst.
"""
import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.compute_client import WS_MAX_QUEUE, WS_PING_INTERVAL, WS_PING_TIMEOUT, ComputeClient
from protocol.vc_protocol.validator_requests import ExecutorSpecRequest, RentedMachineRequest
from services.miner_service import MinerService
from services.redis_service import MACHINE_SPEC_CHANNEL

pytest_plugins = ["fixtures.incentive_fixtures"]


def _client(message_queue: list | None = None) -> ComputeClient:
    client = ComputeClient.__new__(ComputeClient)
    client.keypair = MagicMock(ss58_address="validator-hotkey")
    client.lock = asyncio.Lock()
    client.message_queue = message_queue or []
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.compute_app_uri = "wss://backend.example/validator"
    client.miner_service = MagicMock()
    return client


async def _bridge_machine_spec(payload: dict[str, Any]) -> ExecutorSpecRequest:
    async def listen():
        yield {"channel": MACHINE_SPEC_CHANNEL.encode(), "data": json.dumps(payload).encode()}
        raise asyncio.CancelledError

    pubsub = MagicMock(listen=listen, aclose=AsyncMock())
    client = _client()
    client.miner_service.redis_service.subscribe = AsyncMock(return_value=pubsub)
    with pytest.raises(asyncio.CancelledError):
        await client.subscribe_mesages_from_redis()
    return client.message_queue[0]


@pytest.mark.asyncio
async def test_publisher_stamps_sent_at_and_batch_total(create_job_result, mock_settings) -> None:
    # Arrange
    jobs = [create_job_result(), create_job_result()]
    redis_service = MagicMock()
    redis_service.publish = AsyncMock()
    service = MinerService(
        ssh_service=MagicMock(),
        task_service=MagicMock(),
        redis_service=redis_service,
        attestation_service=MagicMock(),
    )

    # Act
    await service.publish_machine_specs(jobs, miner_hotkey="hk", miner_coldkey="ck")

    # Assert
    payloads = [call.args[1] for call in redis_service.publish.await_args_list]
    assert [payload["batch_total"] for payload in payloads] == [2, 2]
    assert all(isinstance(payload["sent_at"], float) for payload in payloads)


@pytest.mark.asyncio
async def test_bridge_carries_sent_at_and_batch_total(create_job_result, mock_settings) -> None:
    # Arrange
    redis_service = MagicMock()
    redis_service.publish = AsyncMock()
    service = MinerService(
        ssh_service=MagicMock(),
        task_service=MagicMock(),
        redis_service=redis_service,
        attestation_service=MagicMock(),
    )
    await service.publish_machine_specs([create_job_result()], miner_hotkey="hk", miner_coldkey="ck")
    _, payload = redis_service.publish.await_args.args

    # Act
    spec = await _bridge_machine_spec(payload)

    # Assert
    assert spec.sent_at == payload["sent_at"]
    assert spec.batch_total == 1


@pytest.mark.asyncio
async def test_bridge_tolerates_payload_without_stamps(create_job_result, mock_settings) -> None:
    # Arrange
    redis_service = MagicMock()
    redis_service.publish = AsyncMock()
    service = MinerService(
        ssh_service=MagicMock(),
        task_service=MagicMock(),
        redis_service=redis_service,
        attestation_service=MagicMock(),
    )
    await service.publish_machine_specs([create_job_result()], miner_hotkey="hk", miner_coldkey="ck")
    _, payload = redis_service.publish.await_args.args
    del payload["sent_at"]
    del payload["batch_total"]

    # Act
    spec = await _bridge_machine_spec(payload)

    # Assert
    assert spec.sent_at is None
    assert spec.batch_total is None


@pytest.mark.asyncio
async def test_send_loop_stamps_forwarded_at_and_messages_still_waiting_behind() -> None:
    # Arrange
    client = _client([RentedMachineRequest(), RentedMachineRequest()])
    client.ws = MagicMock()
    client.ws.send = AsyncMock()
    task = asyncio.create_task(client.handle_send_messages())

    # Act
    while client.ws.send.await_count < 2:
        await asyncio.sleep(0)
    task.cancel()

    # Assert
    sent = [json.loads(call.args[0]) for call in client.ws.send.await_args_list]
    assert [message["queue_depth"] for message in sent] == [1, 0]
    assert all(isinstance(message["forwarded_at"], float) for message in sent)


def test_connect_opens_the_socket_with_burst_sized_limits() -> None:
    # Arrange
    client = _client()

    # Act
    with patch("clients.compute_client.websockets.connect") as connect:
        client.connect()

    # Assert
    connect.assert_called_once_with(
        "wss://backend.example/validator",
        max_queue=WS_MAX_QUEUE,
        ping_interval=WS_PING_INTERVAL,
        ping_timeout=WS_PING_TIMEOUT,
    )
    assert WS_MAX_QUEUE >= 1024
    assert WS_PING_TIMEOUT > WS_PING_INTERVAL
