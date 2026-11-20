"""SubtensorClient.get_time_from_block gives the cycle id: the block's chain time in UTC, or None with no answer."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from clients.subtensor_client import SubtensorClient

BLOCK = 9542550


class _Chain:
    """A node's RPC: the finalized height, the best chain's hash per height and each hash's timestamp (ms)."""

    def __init__(self, finalized: int, hashes: dict[int, str | None], times: dict[str, int]):
        self.finalized, self.hashes, self.times = finalized, hashes, times
        self.down = False

    def rpc_request(self, method, params):
        if self.down:
            raise ConnectionError("down")
        if method == "chain_getFinalizedHead":
            return {"result": f"0xfinal{self.finalized}"}
        if method == "chain_getHeader":
            return {"result": {"number": hex(self.finalized)}}
        if method == "chain_getBlockHash":
            return {"result": self.hashes.get(params[0])}
        raise AssertionError(method)

    def query(self, module, storage, block_hash):
        assert (module, storage) == ("Timestamp", "Now")
        return SimpleNamespace(value=self.times[block_hash])

    def get_block_hash(self, block):
        raise AssertionError("the client's cached get_block_hash keeps a null answer")


def _client(node) -> MagicMock:
    fake_self = MagicMock()
    fake_self.default_extra = {}
    fake_self.get_node.return_value = node
    return fake_self


def _time_of(node):
    return SubtensorClient.get_time_from_block(_client(node), BLOCK)


@pytest.mark.asyncio
async def test_a_blocks_time_is_its_chain_timestamp_in_utc_whatever_the_hosts_zone(monkeypatch):
    node = _Chain(BLOCK, {BLOCK: "0xabc"}, {"0xabc": 1795114358000})  # 2026-11-19 18:52:38 UTC, in ms
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        assert await _time_of(node) == "2026-11-19 18:52:38"
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


@pytest.mark.asyncio
async def test_a_chain_that_does_not_answer_gives_no_time():
    node = _Chain(BLOCK, {BLOCK: "0xabc"}, {"0xabc": 1795114358000})
    node.down = True
    assert await _time_of(node) is None


@pytest.mark.asyncio
async def test_a_peer_with_no_hash_for_the_block_gives_no_time_never_its_best_blocks():
    node = _Chain(BLOCK, {BLOCK: None}, {})
    node.query = MagicMock()
    assert await _time_of(node) is None
    node.query.assert_not_called()


@pytest.mark.asyncio
async def test_a_peer_that_answered_no_hash_is_asked_again_and_its_later_answer_is_used():
    """The client's get_block_hash keeps a height's first answer, a null one included, so the block stayed unreadable
    after the peer caught up. Every attempt asks the chain again."""
    node = _Chain(BLOCK, {BLOCK: None}, {"0xabc": 1795114358000})

    assert await _time_of(node) is None
    node.hashes[BLOCK] = "0xabc"
    assert await _time_of(node) == "2026-11-19 18:52:38"


@pytest.mark.asyncio
async def test_a_block_reorganized_before_finality_reports_under_one_cycle_id_only():
    """Review on f37a14f: two best-chain hashes at one height (a reorganization) gave two timestamps, so one job block
    reported under two cycle ids and settlement could count it twice. Only a finalized block gives a time: while
    finality is behind, no attempt runs, and every attempt after it reads the one finalized block."""
    node = _Chain(BLOCK - 1, {BLOCK: "0xfirst"}, {"0xfirst": 1795114358000, "0xsecond": 1795114370000})
    answers = [await _time_of(node)]
    node.hashes[BLOCK] = "0xsecond"  # the best chain at this height changes
    answers.append(await _time_of(node))
    node.finalized = BLOCK  # finality reaches the block on the second branch
    answers += [await _time_of(node), await _time_of(node)]

    assert answers == [None, None, "2026-11-19 18:52:50", "2026-11-19 18:52:50"]
    assert len({answer for answer in answers if answer}) == 1
