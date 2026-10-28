import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import tenacity
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
async def test_messages_on_a_socket_the_backend_closed_with_1009_stay_queued():
    # after a 1009 close every later send on that socket raises 1009 too, so none of them is the oversized one
    ws = AsyncMock()
    client = _client(ws)
    first, second = _message("pod-1"), _message("pod-2")
    client.message_queue.append(first)
    client.message_queue.append(second)

    def closed(_payload):
        client.ws = None  # as run_forever does once the socket is gone
        raise ConnectionClosedError(
            Close(1009, "message too big"), Close(1009, "message too big"), True
        )

    ws.send.side_effect = closed

    with patch.object(ComputeClient.send_model.retry, "wait", tenacity.wait_none()):
        await _drain(client)

    assert list(client.message_queue) == [first, second]
