"""SubtensorClient.get_time_from_block gives the cycle id: the block's chain time in UTC, or None with no answer."""

import time
from unittest.mock import MagicMock

import pytest

from clients.subtensor_client import SubtensorClient


def _client(node) -> MagicMock:
    fake_self = MagicMock()
    fake_self.default_extra = {}
    fake_self.get_node.return_value = node
    return fake_self


@pytest.mark.asyncio
async def test_a_blocks_time_is_its_chain_timestamp_in_utc_whatever_the_hosts_zone(monkeypatch):
    node = MagicMock()
    node.query.return_value.value = 1795114358000  # 2026-11-19 18:52:38 UTC, in ms
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        assert await SubtensorClient.get_time_from_block(_client(node), 9542550) == "2026-11-19 18:52:38"
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


@pytest.mark.asyncio
async def test_a_chain_that_does_not_answer_gives_no_time():
    node = MagicMock()
    node.get_block_hash.side_effect = ConnectionError("down")
    assert await SubtensorClient.get_time_from_block(_client(node), 9542550) is None


@pytest.mark.asyncio
async def test_a_peer_with_no_hash_for_the_block_gives_no_time_never_its_best_blocks():
    node = MagicMock()
    node.get_block_hash.return_value = None
    node.query.return_value.value = 1795164758000
    assert await SubtensorClient.get_time_from_block(_client(node), 9542550) is None
    node.query.assert_not_called()
