"""DAH-2792: the validator -> connector -> backend path carries delivery stamps so the backend
can measure how long a message waited at each hop, and the connector opens the websocket with a
pong timeout long enough for the keepalive to survive a scoring-cycle burst.
"""
import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets
from clients.compute_client import WS_PING_INTERVAL, WS_PING_TIMEOUT, ComputeClient
from payload_models.payloads import ContainerCreated, DeliveryStamps
from protocol.vc_protocol.validator_requests import (
    ExecutorSpecRequest,
    RentedMachineRequest,
    ValidationEvent,
)
from services.miner_service import MinerService
from services.redis_service import MACHINE_SPEC_CHANNEL

pytest_plugins = ["fixtures.incentive_fixtures"]


def _client(message_queue: list[DeliveryStamps] | None = None) -> ComputeClient:
    client = ComputeClient.__new__(ComputeClient)
    client.keypair = MagicMock(ss58_address="validator-hotkey")
    client.lock = asyncio.Lock()
    client.message_queue = message_queue or []
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.compute_app_uri = "wss://backend.example/validator"
    client.miner_service = MagicMock()
    return client


async def _bridge_machine_spec(payload: dict[str, Any]) -> ExecutorSpecRequest:
    async def listen() -> AsyncIterator[dict[str, bytes]]:
        yield {"channel": MACHINE_SPEC_CHANNEL.encode(), "data": json.dumps(payload).encode()}
        raise asyncio.CancelledError

    pubsub = MagicMock(listen=listen, aclose=AsyncMock())
    client = _client()
    client.miner_service.redis_service.subscribe = AsyncMock(return_value=pubsub)
    with pytest.raises(asyncio.CancelledError):
        await client.subscribe_mesages_from_redis()
    return client.message_queue[0]


async def _published_payloads(jobs: list[Any]) -> list[dict[str, Any]]:
    redis_service = MagicMock()
    redis_service.publish = AsyncMock()
    service = MinerService(
        ssh_service=MagicMock(),
        task_service=MagicMock(),
        redis_service=redis_service,
        attestation_service=MagicMock(),
    )
    await service.publish_machine_specs(jobs, miner_hotkey="hk", miner_coldkey="ck")
    return [call.args[1] for call in redis_service.publish.await_args_list]


@pytest.mark.asyncio
async def test_publisher_stamps_sent_at_and_batch_total(create_job_result, mock_settings) -> None:
    # Arrange
    jobs = [create_job_result(), create_job_result()]

    # Act
    payloads = await _published_payloads(jobs)

    # Assert
    assert [payload["batch_total"] for payload in payloads] == [2, 2]
    assert all(isinstance(payload["sent_at"], float) for payload in payloads)


@pytest.mark.asyncio
async def test_bridge_carries_sent_at_and_batch_total(create_job_result, mock_settings) -> None:
    # Arrange
    [payload] = await _published_payloads([create_job_result()])

    # Act
    spec = await _bridge_machine_spec(payload)

    # Assert
    assert spec.sent_at == payload["sent_at"]
    assert spec.batch_total == 1


@pytest.mark.asyncio
async def test_bridge_tolerates_payload_without_stamps(create_job_result, mock_settings) -> None:
    # Arrange: a validator from before DAH-2792 publishes no stamps
    [payload] = await _published_payloads([create_job_result()])
    del payload["sent_at"]
    del payload["batch_total"]

    # Act
    spec = await _bridge_machine_spec(payload)

    # Assert
    assert spec.sent_at is None
    assert spec.batch_total is None


@pytest.mark.asyncio
async def test_bridge_carries_structured_validation_event(create_job_result, mock_settings) -> None:
    job = create_job_result(log_text="GPU mismatch >>> legacy JSON")
    job.validation_event = ValidationEvent(
        event="GPU mismatch",
        reason_code="GPU_MISMATCH",
        severity="critical",
        impact="Node cannot be listed",
        remediation="Check the installed GPU",
        what_we_saw={"expected": "A100", "actual": "T4"},
        check_id="executor.validate.gpu",
        when=datetime(2026, 9, 1, tzinfo=UTC),
    )

    [payload] = await _published_payloads([job])
    spec = await _bridge_machine_spec(payload)

    assert payload["validation_event"]["reason_code"] == "GPU_MISMATCH"
    assert isinstance(spec.validation_event, ValidationEvent)
    assert spec.validation_event.what_we_saw == {"expected": "A100", "actual": "T4"}
    assert spec.validation_event.when == datetime(2026, 9, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_bridge_tolerates_payload_without_validation_event(create_job_result, mock_settings) -> None:
    [payload] = await _published_payloads([create_job_result()])
    del payload["validation_event"]

    spec = await _bridge_machine_spec(payload)

    assert spec.validation_event is None


async def _drain_send_loop(client: ComputeClient, expected_sends: int) -> list[dict[str, Any]]:
    # the loop never returns on its own; the mocked socket stops it after the expected sends
    sent_messages: list[dict[str, Any]] = []

    async def send(raw_message: str) -> None:
        sent_messages.append(json.loads(raw_message))
        if len(sent_messages) == expected_sends:
            raise asyncio.CancelledError

    client.ws = MagicMock(send=send)
    with pytest.raises(asyncio.CancelledError):
        await client.handle_send_messages()
    return sent_messages


@pytest.mark.asyncio
async def test_send_loop_stamps_forwarded_at_and_messages_still_waiting_behind() -> None:
    # Arrange
    client = _client([RentedMachineRequest(), RentedMachineRequest()])

    # Act
    sent = await _drain_send_loop(client, expected_sends=2)

    # Assert
    assert [message["queue_depth"] for message in sent] == [1, 0]
    assert all(isinstance(message["forwarded_at"], float) for message in sent)


@pytest.mark.asyncio
async def test_send_loop_stamps_container_responses_too() -> None:
    # Arrange: the queue mixes container responses (payload_models) with validator requests
    container_created = ContainerCreated(
        miner_hotkey="hk", executor_id="ex", pod_id="pod", container_name="c", volume_name="v", port_maps=[]
    )
    client = _client([container_created, RentedMachineRequest()])

    # Act
    sent = await _drain_send_loop(client, expected_sends=2)

    # Assert
    assert [message["message_type"] for message in sent] == ["ContainerCreated", "RentedMachineRequest"]
    assert [message["queue_depth"] for message in sent] == [1, 0]


@pytest.mark.asyncio
async def test_resent_message_is_stamped_again_by_the_retry() -> None:
    # Arrange: the first attempt dies on a closed socket, tenacity sends the same object again
    message = RentedMachineRequest()
    client = _client()
    stamps: list[float] = []

    async def send(raw_message: str) -> None:
        stamps.append(json.loads(raw_message)["forwarded_at"])
        if len(stamps) == 1:
            raise websockets.ConnectionClosedError(None, None)

    client.ws = MagicMock(send=send)

    # Act
    with patch("clients.compute_client.time.time", side_effect=[100.0, 110.0]):
        await client.send_model(message)

    # Assert: the retry reports its own moment, not the first attempt's
    assert stamps == [100.0, 110.0]
    assert message.forwarded_at == 110.0


def test_connect_sets_keepalive_ping_interval_and_timeout() -> None:
    # Arrange
    client = _client()

    # Act
    with patch("clients.compute_client.websockets.connect") as connect:
        client.connect()

    # Assert
    connect.assert_called_once_with(
        "wss://backend.example/validator",
        ping_interval=WS_PING_INTERVAL,
        ping_timeout=WS_PING_TIMEOUT,
    )
