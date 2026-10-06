"""A cancelled chain read keeps the websocket to itself until its thread ends."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from clients.subtensor_client import SubtensorClient


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_first", [False, True])
async def test_chain_reads_stay_serial_until_a_cancelled_read_thread_ends(cancel_first):
    # Arrange: hold the first synchronous read open while a second read queues.
    client = SubtensorClient.__new__(SubtensorClient)
    client._chain_reads_in_thread = True
    client._chain_read_lock = asyncio.Lock()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    second_started = threading.Event()

    def first_read():
        started.set()
        try:
            assert release.wait(2), "test did not release the first read"
        finally:
            finished.set()

    def second_read():
        second_started.set()

    first = asyncio.create_task(client._run_chain_read(first_read))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 1)
        # Act: cancellation must not release ownership of a still-running read.
        if cancel_first:
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        second = asyncio.create_task(client._run_chain_read(second_read))
        overlapped = await asyncio.to_thread(second_started.wait, 0.15)
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 1)

    # Assert: the websocket cannot serve two synchronous reads at the same time.
    assert not overlapped, "second chain read started while the cancelled worker was still running"


def _hold_chain_read_until(release: threading.Event, started: threading.Event):
    def chain_read():
        started.set()
        assert release.wait(2), "test did not release the chain read"

    return chain_read


@pytest.mark.asyncio
async def test_warm_up_redial_waits_for_the_chain_read_in_its_thread():
    # Arrange: a rent's chain read holds the websocket in its thread
    client = SubtensorClient.__new__(SubtensorClient)
    client.default_extra = {}
    client._chain_reads_in_thread = True
    client._chain_read_lock = asyncio.Lock()
    client._return_to_first_endpoint = MagicMock()
    client.set_subtensor = MagicMock(side_effect=RuntimeError("stop the warm-up here"))
    client._switch_endpoint_after_read_failure = MagicMock()
    started = threading.Event()
    release = threading.Event()
    read = asyncio.create_task(client._run_chain_read(_hold_chain_read_until(release, started)))
    assert await asyncio.to_thread(started.wait, 1)

    # Act
    warm_up = asyncio.create_task(client._warm_up_subtensor())
    await asyncio.sleep(0.15)
    redialled_during_the_read = client._return_to_first_endpoint.called
    release.set()
    await read
    await asyncio.sleep(0.05)
    warm_up.cancel()

    # Assert: the redial may close the websocket the thread is reading from
    assert not redialled_during_the_read
    assert client._return_to_first_endpoint.called


@pytest.mark.asyncio
async def test_warm_up_endpoint_switch_after_a_failed_load_waits_for_the_chain_read_in_its_thread():
    # Arrange: the warm-up's first load fails while a rent's chain read holds the websocket
    client = SubtensorClient.__new__(SubtensorClient)
    client.default_extra = {}
    client._chain_reads_in_thread = True
    client._chain_read_lock = asyncio.Lock()
    client._return_to_first_endpoint = MagicMock()
    client.set_subtensor = MagicMock()
    client._switch_endpoint_after_read_failure = MagicMock()
    started = threading.Event()
    release = threading.Event()
    reads: list[asyncio.Task] = []

    async def failing_load_beside_a_rent_read():
        reads.append(asyncio.create_task(client._run_chain_read(_hold_chain_read_until(release, started))))
        assert await asyncio.to_thread(started.wait, 1)
        raise RuntimeError("chain read failed")

    client.get_miners = failing_load_beside_a_rent_read

    with patch.object(SubtensorClient, "_subtensor", MagicMock()):
        # Act
        warm_up = asyncio.create_task(client._warm_up_subtensor())
        await asyncio.sleep(0.15)
        switched_during_the_read = client._switch_endpoint_after_read_failure.called
        release.set()
        await reads[0]
        await asyncio.sleep(0.05)
        warm_up.cancel()

    # Assert: the switch drops the subtensor and closes the websocket the thread is reading from
    assert not switched_during_the_read
    assert client._switch_endpoint_after_read_failure.called


@pytest.mark.asyncio
async def test_shutdown_returns_after_the_cancelled_warm_up_read_thread_ends():
    # Arrange: the warm-up is cancelled while its chain read runs in a thread
    client = SubtensorClient.__new__(SubtensorClient)
    client._chain_reads_in_thread = True
    client._chain_read_lock = asyncio.Lock()
    started = threading.Event()
    release = threading.Event()
    hold_read = _hold_chain_read_until(release, started)
    with (
        patch.object(SubtensorClient, "_instance", client),
        patch.object(SubtensorClient, "_initialized", True),
        patch.object(
            SubtensorClient, "_warm_up_task", asyncio.create_task(client._run_chain_read(hold_read))
        ),
    ):
        assert await asyncio.to_thread(started.wait, 1)

        # Act
        shutdown = asyncio.create_task(SubtensorClient.shutdown())
        await asyncio.sleep(0.15)
        returned_during_the_read = shutdown.done()
        release.set()
        await shutdown

    # Assert: a new instance after shutdown must not share the websocket with the old thread
    assert not returned_during_the_read
