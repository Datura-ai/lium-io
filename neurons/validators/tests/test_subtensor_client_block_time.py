"""SubtensorClient.get_time_from_block gives the cycle id: the block's chain time in UTC, or None with no answer."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from clients.subtensor_client import SubtensorClient

BLOCK = 9542550


class _Chain:
    """A node's RPC over hash-linked headers: hash -> (height, parent hash, timestamp in ms), and the finalized head."""

    def __init__(self, blocks: dict[str, tuple[int, str, int]], finalized: str):
        self.blocks, self.finalized = blocks, finalized
        self.down = False

    def rpc_request(self, method, params):
        self.calls += 1
        if self.down:
            raise ConnectionError("down")
        if method == "chain_getFinalizedHead":
            return {"result": self.finalized}
        if method == "chain_getHeader":
            block = self.blocks.get(params[0])
            return {"result": block and {"number": hex(block[0]), "parentHash": block[1]}}
        raise AssertionError(f"{method}: a lookup by height can reach a peer on another branch")

    calls = 0

    def query(self, module, storage, params=None, block_hash=None):
        self.calls += 1
        if (module, storage) == ("System", "BlockHash"):
            # the state at `block_hash`: the hash of the block at the height on that block's own chain, zero beyond it
            at, (height,) = block_hash, params
            while at in self.blocks and self.blocks[at][0] > height:
                at = self.blocks[at][1]
            found = at in self.blocks and self.blocks[at][0] == height
            return SimpleNamespace(value=at if found else "0x" + "0" * 64)
        assert (module, storage) == ("Timestamp", "Now")
        return SimpleNamespace(value=self.blocks[block_hash][2])

    def get_block_hash(self, block):
        raise AssertionError("the client's cached get_block_hash keeps a null answer")


def _branch(name: str, start: int, end: int, parent: str, first_ms: int) -> dict[str, tuple[int, str, int]]:
    """Blocks start..end on one branch, 12 s apart, the first one's parent `parent`."""
    blocks = {}
    for height in range(start, end + 1):
        blocks[f"0x{name}{height}"] = (height, parent, first_ms + (height - start) * 12000)
        parent = f"0x{name}{height}"
    return blocks


def _client(node) -> MagicMock:
    fake_self = MagicMock()
    fake_self.default_extra = {}
    fake_self.get_node.return_value = node
    return fake_self


def _time_of(node):
    return SubtensorClient.get_time_from_block(_client(node), BLOCK)


T = 1795114358000  # 2026-11-19 18:52:38 UTC, in ms
CHAIN = _branch("a", BLOCK - 1, BLOCK + 2, "0xroot", T - 12000)


@pytest.mark.asyncio
async def test_a_blocks_time_is_its_chain_timestamp_in_utc_whatever_the_hosts_zone(monkeypatch):
    node = _Chain(CHAIN, f"0xa{BLOCK + 2}")
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        assert await _time_of(node) == "2026-11-19 18:52:38"
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


@pytest.mark.asyncio
async def test_a_chain_that_does_not_answer_gives_no_time():
    node = _Chain(CHAIN, f"0xa{BLOCK + 2}")
    node.down = True
    assert await _time_of(node) is None


@pytest.mark.asyncio
async def test_a_block_finality_has_not_reached_gives_no_time_and_is_read_once_it_has():
    node = _Chain(CHAIN, f"0xa{BLOCK - 1}")
    assert await _time_of(node) is None
    node.finalized = f"0xa{BLOCK}"
    assert await _time_of(node) == "2026-11-19 18:52:38"


@pytest.mark.asyncio
async def test_a_block_older_than_the_head_states_hash_map_gives_no_time():
    node = _Chain(CHAIN, f"0xa{BLOCK + 2}")
    node.query = MagicMock(return_value=SimpleNamespace(value="0x" + "0" * 64))
    assert await _time_of(node) is None
    assert all(call.args[:2] == ("System", "BlockHash") for call in node.query.call_args_list)


@pytest.mark.asyncio
async def test_a_head_far_past_the_block_is_read_in_a_fixed_number_of_calls():
    """The read costs the same few calls however far the finalized head is past the job block."""
    node = _Chain(_branch("a", BLOCK - 1, BLOCK + 500, "0xroot", T - 12000), f"0xa{BLOCK + 500}")
    assert await _time_of(node) == "2026-11-19 18:52:38"
    assert node.calls == 4  # finalized head, its header, the block's hash in its state, the block's timestamp


@pytest.mark.asyncio
async def test_a_block_reorganized_before_finality_reports_under_one_cycle_id_only():
    """Reviews on f37a14f and e95f474: two branches at one height (a reorganization, or a lookup by height that a
    reconnect sent to a peer on the other branch) gave two timestamps, so one job block reported under two cycle ids.
    The hash is read from the finalized head's own state, so every attempt reads the finalized branch."""
    a = _branch("a", BLOCK, BLOCK + 2, "0xroot", T)
    b = _branch("b", BLOCK, BLOCK + 2, "0xroot", T + 12000)  # the competing branch: 18:52:50 at the same height
    node = _Chain({**a, **b}, "0xroot")
    node.blocks["0xroot"] = (BLOCK - 1, "0xgenesis", T - 12000)
    answers = [await _time_of(node)]  # finality is behind the height
    node.finalized = f"0xb{BLOCK + 2}"  # finality reaches it on branch b; a peer may still serve branch a's blocks
    answers += [await _time_of(node), await _time_of(node)]

    assert answers == [None, "2026-11-19 18:52:50", "2026-11-19 18:52:50"]
