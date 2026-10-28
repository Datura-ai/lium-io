import asyncio
from unittest.mock import AsyncMock

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from clients.compute_client import MAX_FRAME_BYTES, ComputeClient, OutgoingMessages
from payload_models.payloads import FailedAddDebugSshKey


def _client(ws) -> ComputeClient:
    client = ComputeClient.__new__(ComputeClient)
    client.message_queue = OutgoingMessages()
    client.lock = asyncio.Lock()
    client.logging_extra = {}
    client.ws = ws
    return client


def _message(text: str) -> FailedAddDebugSshKey:
    return FailedAddDebugSshKey(miner_hotkey="hk", executor_id="ex", msg=text)


async def _drain(client: ComputeClient) -> None:
    task = asyncio.create_task(client.handle_send_messages())
    for _ in range(50):
        await asyncio.sleep(0)
    task.cancel()


@pytest.mark.asyncio
async def test_an_oversized_message_is_dropped_before_sending_and_the_next_one_still_goes():
    ws = AsyncMock()
    client = _client(ws)
    big = _message("x" * MAX_FRAME_BYTES)
    small = _message("pod-1")
    client.message_queue.append(big)
    client.message_queue.append(small)

    await _drain(client)

    assert len(client.message_queue) == 0
    assert ws.send.await_count == 1
    assert '"pod-1"' in ws.send.await_args.args[0]


@pytest.mark.asyncio
async def test_a_message_the_backend_closes_with_1009_is_not_retried_or_requeued():
    ws = AsyncMock()
    ws.send.side_effect = ConnectionClosedError(Close(1009, "message too big"), Close(1009, "message too big"), True)
    client = _client(ws)
    client.message_queue.append(_message("pod-1"))

    await _drain(client)

    assert ws.send.await_count == 1
    assert len(client.message_queue) == 0
