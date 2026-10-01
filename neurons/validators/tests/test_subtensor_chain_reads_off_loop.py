import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
