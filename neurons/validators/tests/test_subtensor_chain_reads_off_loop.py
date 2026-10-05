import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.compute_client import ComputeClient
from clients.subtensor_client import SubtensorClient
from clients.validator_portal_api import ValidatorPortalAPI

CHAIN_READ_SECONDS = 0.5


def _slow_metagraph() -> MagicMock:
    time.sleep(CHAIN_READ_SECONDS)
    return MagicMock(neurons=[])


def _slow_evm_query_map(**_kwargs: object) -> list[tuple[int, tuple[str, int]]]:
    time.sleep(CHAIN_READ_SECONDS)
    return [(7, ("0xabc", 100))]


def _make_connector_subtensor_client() -> SubtensorClient:
    client = SubtensorClient.__new__(SubtensorClient)
    client.debug_miner = None
    client.default_extra = {}
    client.netuid = 1
    client.miners = []
    client.uid_to_evm_address = {}
    client.hotkey_to_evm_address = {}
    client.redis_service = AsyncMock()
    client.redis_service.get.return_value = None
    client._has_alerted_for_stale_portal_snapshot = False
    client._chain_reads_in_thread = True
    client._chain_read_lock = asyncio.Lock()
    client._miners_fetch_lock = asyncio.Lock()
    client.get_metagraph = _slow_metagraph
    client.get_node = MagicMock(return_value=MagicMock(query_map=_slow_evm_query_map))
    return client


async def _record_loop_tick_gaps(stalls: list[float]) -> None:
    last_tick = time.perf_counter()
    while True:
        await asyncio.sleep(0.01)
        now = time.perf_counter()
        stalls.append(now - last_tick)
        last_tick = now


@pytest.mark.asyncio
async def test_connector_chain_sync_keeps_the_event_loop_serving() -> None:
    client = _make_connector_subtensor_client()
    stalls: list[float] = []

    with patch.object(ValidatorPortalAPI, "get_opted_in_miners", AsyncMock(return_value=[])):
        watcher = asyncio.create_task(_record_loop_tick_gaps(stalls))
        await client.fetch_miners()
        assert len(stalls) > 10  # the loop kept ticking while the metagraph was read
        await client.sync_evm_address_maps()
        watcher.cancel()

    assert max(stalls) < CHAIN_READ_SECONDS / 2
    assert client.miners == []
    assert client.uid_to_evm_address == {7: "0xabc"}


@pytest.mark.asyncio
async def test_rent_during_the_warm_up_first_load_joins_it_with_one_metagraph_read() -> None:
    client = _make_connector_subtensor_client()
    client._return_to_first_endpoint = MagicMock()
    client.set_subtensor = MagicMock()
    serving_miner = MagicMock(hotkey="miner-hotkey", uid=7)
    metagraph_reads: list[float] = []

    def _counted_slow_metagraph() -> MagicMock:
        metagraph_reads.append(time.perf_counter())
        time.sleep(CHAIN_READ_SECONDS)
        return MagicMock(neurons=[serving_miner])

    client.get_metagraph = _counted_slow_metagraph

    with (
        patch.object(SubtensorClient, "_subtensor", MagicMock()),
        patch.object(ValidatorPortalAPI, "get_opted_in_miners", AsyncMock(return_value=[])),
    ):
        warm_up = asyncio.create_task(client._warm_up_subtensor())
        await asyncio.sleep(CHAIN_READ_SECONDS / 2)  # the warm-up's read is in the thread now
        miner = await client.get_miner("miner-hotkey")
        warm_up.cancel()

    assert miner is serving_miner
    assert len(metagraph_reads) == 1


@pytest.mark.asyncio
async def test_connector_connects_to_backend_while_the_first_miners_load_is_pending() -> None:
    # Arrange
    subtensor_client = _make_connector_subtensor_client()
    subtensor_client._return_to_first_endpoint = MagicMock()
    subtensor_client.set_subtensor = MagicMock()
    portal_released = asyncio.Event()

    async def _held_portal_request() -> list:
        await portal_released.wait()
        return []

    async def _backend_never_answering():
        await asyncio.Event().wait()
        yield

    compute_client = ComputeClient.__new__(ComputeClient)
    compute_client.logging_extra = {}
    compute_client.connect = MagicMock(side_effect=_backend_never_answering)
    held_portal = AsyncMock(side_effect=_held_portal_request)

    with (
        patch.object(SubtensorClient, "_subtensor", MagicMock()),
        patch.object(SubtensorClient, "_warm_up_task", None),
        patch.object(SubtensorClient, "get_instance", return_value=subtensor_client),
        patch.object(ValidatorPortalAPI, "get_opted_in_miners", held_portal),
        patch.multiple(
            ComputeClient,
            handle_send_messages=AsyncMock(),
            subscribe_mesages_from_redis=AsyncMock(),
            poll_rented_machines=AsyncMock(),
            poll_executors_uptime=AsyncMock(),
            poll_revenue_per_gpu_type=AsyncMock(),
        ),
    ):
        # Act
        run_forever = asyncio.create_task(compute_client.run_forever())
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(1):
                while not compute_client.connect.called:
                    await asyncio.sleep(0.01)
        connect_attempted = compute_client.connect.called
        run_forever.cancel()
        SubtensorClient._warm_up_task.cancel()

    # Assert
    assert held_portal.await_count == 1
    assert not portal_released.is_set()
    assert connect_attempted


@pytest.mark.asyncio
async def test_only_the_connector_reads_the_chain_in_a_thread() -> None:
    # Arrange
    bare_client = SubtensorClient.__new__(SubtensorClient)
    running_warm_up = MagicMock(done=MagicMock(return_value=False))
    connector_initialize = AsyncMock()

    async def _backend_never_answering():
        await asyncio.Event().wait()
        yield

    compute_client = ComputeClient.__new__(ComputeClient)
    compute_client.logging_extra = {}
    compute_client.connect = MagicMock(side_effect=_backend_never_answering)

    # Act
    with (
        patch.object(SubtensorClient, "get_instance", return_value=bare_client),
        patch.object(SubtensorClient, "_warm_up_task", running_warm_up),
    ):
        main_validator_client = await SubtensorClient.initialize()
    with (
        patch.object(SubtensorClient, "initialize", connector_initialize),
        patch.multiple(
            ComputeClient,
            handle_send_messages=AsyncMock(),
            subscribe_mesages_from_redis=AsyncMock(),
            poll_rented_machines=AsyncMock(),
            poll_executors_uptime=AsyncMock(),
            poll_revenue_per_gpu_type=AsyncMock(),
        ),
    ):
        run_forever = asyncio.create_task(compute_client.run_forever())
        await asyncio.sleep(0.05)
        run_forever.cancel()

    # Assert
    assert SubtensorClient._chain_reads_in_thread is False
    assert main_validator_client._chain_reads_in_thread is False
    connector_initialize.assert_awaited_once_with(chain_reads_in_thread=True)


def _make_main_validator_subtensor_client() -> SubtensorClient:
    client = _make_connector_subtensor_client()
    client._chain_reads_in_thread = False
    return client


@pytest.mark.asyncio
async def test_main_validator_concurrent_empty_get_miners_each_fetch_as_on_main() -> None:
    client = _make_main_validator_subtensor_client()
    fetches: list[int] = []

    async def _fetch_miners_yielding_to_the_loop() -> None:
        fetches.append(1)
        await asyncio.sleep(0.01)
        client.miners = [MagicMock()]

    client.fetch_miners = _fetch_miners_yielding_to_the_loop

    await asyncio.gather(client.get_miners(), client.get_miners())

    assert len(fetches) == 2


@pytest.mark.asyncio
async def test_main_validator_warm_up_retry_after_failed_evm_sync_fetches_miners_again() -> None:
    client = _make_main_validator_subtensor_client()
    client._return_to_first_endpoint = MagicMock()
    client.set_subtensor = MagicMock()
    client._switch_endpoint_after_read_failure = MagicMock()
    fetches: list[int] = []

    async def _fetch_miners() -> None:
        fetches.append(1)
        client.miners = [MagicMock()]

    client.fetch_miners = _fetch_miners
    client.sync_evm_address_maps = AsyncMock(side_effect=[RuntimeError("evm sync failed"), None])

    with (
        patch.object(SubtensorClient, "_subtensor", MagicMock()),
        patch("clients.subtensor_client.SUBTENSOR_BACKOFF_INITIAL", 0),
    ):
        warm_up = asyncio.create_task(client._warm_up_subtensor())
        while client.sync_evm_address_maps.await_count < 2:
            await asyncio.sleep(0.01)
        warm_up.cancel()

    assert len(fetches) == 2
