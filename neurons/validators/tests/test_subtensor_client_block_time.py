"""SubtensorClient.get_time_from_block gives the cycle id: the block's chain time in UTC, or None with no answer."""

import time
from unittest.mock import MagicMock

import pytest

from clients.subtensor_client import SubtensorClient

BLOCK = 9542550


# the chain's real storage keys: Timestamp::Now, and the System::BlockHash map's prefix (its key then holds the
# height's twox64 hash and the height as little-endian u32)
TIMESTAMP_NOW = "0xf0c365c3cf59d671eb72da0e7a4113c49f1f0515f462cdcf84e0f1d6045dfcbb"
BLOCK_HASH_MAP = "0x26aa394eea5630e07c48ae0c9558cef7a44704b568d21667356a5a050c118746"
ERROR = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "Client error"}}


def _hash(label: str) -> str:
    return "0x" + label.encode().hex().rjust(64, "0")


class _Chain:
    """A node's RPC over hash-linked headers: hash -> (height, parent hash, timestamp in ms), and the finalized head."""

    def __init__(self, blocks: dict[str, tuple[int, str, int]], finalized: str):
        self.blocks, self.finalized = blocks, finalized
        self.down = False

    calls = 0

    def rpc_request(self, method, params):
        self.calls += 1
        if self.down:
            raise ConnectionError("down")
        if method == "chain_getFinalizedHead":
            return {"result": self.finalized}
        if method == "chain_getHeader":
            block = self.blocks.get(params[0])
            return {"result": block and {"number": hex(block[0]), "parentHash": block[1]}}
        if method == "state_getStorage":
            return self.storage(*params)
        raise AssertionError(f"{method}: a lookup by height can reach a peer on another branch")

    def storage(self, key, at):
        if key == TIMESTAMP_NOW:
            return {"result": "0x" + self.blocks[at][2].to_bytes(8, "little").hex()}
        assert key.startswith(BLOCK_HASH_MAP) and len(key) == len(BLOCK_HASH_MAP) + 24, key
        # the state at `at`: the hash of the block at the height on that block's own chain, unset beyond it
        height = int.from_bytes(bytes.fromhex(key[-8:]), "little")
        while at in self.blocks and self.blocks[at][0] > height:
            at = self.blocks[at][1]
        return {"result": at if at in self.blocks and self.blocks[at][0] == height else None}

    def query(self, *args, **kwargs):
        raise AssertionError("the SDK's storage query decodes an error reply as zero, hiding it from the fallback")

    def get_block_hash(self, block):
        raise AssertionError("the client's cached get_block_hash keeps a null answer")


def _branch(name: str, start: int, end: int, parent: str, first_ms: int) -> dict[str, tuple[int, str, int]]:
    """Blocks start..end on one branch, 12 s apart, the first one's parent `parent`."""
    blocks = {}
    for height in range(start, end + 1):
        blocks[_hash(f"{name}{height}")] = (height, parent, first_ms + (height - start) * 12000)
        parent = _hash(f"{name}{height}")
    return blocks


def _client(node) -> MagicMock:
    fake_self = MagicMock()
    fake_self.default_extra = {}
    fake_self.get_node.return_value = node
    return fake_self


def _time_of(node):
    return SubtensorClient.get_time_from_block(_client(node), BLOCK)


ROOT = _hash("root")
T = 1795114358000  # 2026-11-19 18:52:38 UTC, in ms
CHAIN = _branch("a", BLOCK - 1, BLOCK + 2, ROOT, T - 12000)


@pytest.mark.asyncio
async def test_a_blocks_time_is_its_chain_timestamp_in_utc_whatever_the_hosts_zone(monkeypatch):
    node = _Chain(CHAIN, _hash(f"a{BLOCK + 2}"))
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        assert await _time_of(node) == "2026-11-19 18:52:38"
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


@pytest.mark.asyncio
async def test_a_chain_that_does_not_answer_gives_no_time():
    node = _Chain(CHAIN, _hash(f"a{BLOCK + 2}"))
    node.down = True
    assert await _time_of(node) is None


@pytest.mark.asyncio
async def test_a_block_finality_has_not_reached_gives_no_time_and_is_read_once_it_has():
    node = _Chain(CHAIN, _hash(f"a{BLOCK - 1}"))
    assert await _time_of(node) is None
    node.finalized = _hash(f"a{BLOCK}")
    assert await _time_of(node) == "2026-11-19 18:52:38"


@pytest.mark.asyncio
@pytest.mark.parametrize("unset", [None, "0x" + "0" * 64])
async def test_a_block_older_than_the_head_states_hash_map_gives_no_time(unset):
    node = _Chain(CHAIN, _hash(f"a{BLOCK + 2}"))
    node.storage = lambda key, at: {"result": unset}
    assert await _time_of(node) is None


@pytest.mark.asyncio
async def test_a_head_far_past_the_block_is_read_in_a_fixed_number_of_calls():
    """The read costs the same few calls however far the finalized head is past the job block."""
    node = _Chain(_branch("a", BLOCK - 1, BLOCK + 500, ROOT, T - 12000), _hash(f"a{BLOCK + 500}"))
    assert await _time_of(node) == "2026-11-19 18:52:38"
    assert node.calls == 4  # finalized head, its header, the block's hash in its state, the block's timestamp


@pytest.mark.asyncio
async def test_a_block_reorganized_before_finality_reports_under_one_cycle_id_only():
    """Reviews on f37a14f and e95f474: two branches at one height (a reorganization, or a lookup by height that a
    reconnect sent to a peer on the other branch) gave two timestamps, so one job block reported under two cycle ids.
    The hash is read from the finalized head's own state, so every attempt reads the finalized branch."""
    a = _branch("a", BLOCK, BLOCK + 2, ROOT, T)
    b = _branch("b", BLOCK, BLOCK + 2, ROOT, T + 12000)  # the competing branch: 18:52:50 at the same height
    node = _Chain({**a, **b}, ROOT)
    node.blocks[ROOT] = (BLOCK - 1, _hash("genesis"), T - 12000)
    answers = [await _time_of(node)]  # finality is behind the height
    node.finalized = _hash(f"b{BLOCK + 2}")  # finality reaches it on branch b; a peer may still serve branch a's blocks
    answers += [await _time_of(node), await _time_of(node)]

    assert answers == [None, "2026-11-19 18:52:50", "2026-11-19 18:52:50"]


class _RejectsFinalizedHead(_Chain):
    """An endpoint that serves storage and the metagraph but answers chain_getFinalizedHead with an error."""

    def rpc_request(self, method, params):
        if method == "chain_getFinalizedHead":
            self.calls += 1
            return {**ERROR, "error": {"code": -32601, "message": "Method not found"}}
        return super().rpc_request(method, params)


@pytest.mark.asyncio
async def test_an_endpoint_that_rejects_the_finalized_head_is_left_for_the_fallback():
    primary = _RejectsFinalizedHead(CHAIN, _hash(f"a{BLOCK + 2}"))
    fallback = _Chain(CHAIN, _hash(f"a{BLOCK + 2}"))
    client = _client(primary)
    client._switch_endpoint_after_read_failure.side_effect = lambda error: setattr(
        client.get_node, "return_value", fallback
    )

    assert await SubtensorClient.get_time_from_block(client, BLOCK) == "2026-11-19 18:52:38"
    client._switch_endpoint_after_read_failure.assert_called_once()
    client.set_subtensor.assert_called_once()  # the retry dials the next endpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("unset", [None, "0x" + "00" * 8])
async def test_an_unset_or_zero_timestamp_gives_no_cycle_id(unset):
    """Read as 1970-01-01 it would settle the block under a second cycle id next to its real one."""
    node = _Chain(CHAIN, _hash(f"a{BLOCK + 2}"))
    real, replies = node.storage, iter([unset])

    def storage(key, at):
        value = next(replies, "real") if key == TIMESTAMP_NOW else "real"
        return real(key, at) if value == "real" else {"result": value}

    node.storage = storage
    assert await _time_of(node) == "2026-11-19 18:52:38"  # the next attempt reads the real value
    replies = iter([unset] * 3)
    assert await _time_of(node) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("item", [TIMESTAMP_NOW, BLOCK_HASH_MAP])
async def test_an_endpoint_that_fails_a_storage_read_is_left_for_the_fallback(item):
    """The endpoint still serves the head and headers; its error reply to a storage read moves the read on, where a
    query decoding it as zero kept it on that endpoint and gave no cycle id on every tick."""
    primary = _Chain(CHAIN, _hash(f"a{BLOCK + 2}"))
    real = primary.storage
    primary.storage = lambda key, at: ERROR if key.startswith(item) else real(key, at)
    fallback = _Chain(CHAIN, _hash(f"a{BLOCK + 2}"))
    client = _client(primary)
    client._switch_endpoint_after_read_failure.side_effect = lambda error: setattr(
        client.get_node, "return_value", fallback
    )

    assert await SubtensorClient.get_time_from_block(client, BLOCK) == "2026-11-19 18:52:38"
    client._switch_endpoint_after_read_failure.assert_called_once()
