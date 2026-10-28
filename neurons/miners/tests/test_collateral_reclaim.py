"""The miner's reclaim path for executors that still hold collateral on the contract."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from core.collateral import CollateralClient

CONTRACT = "0x8A4023FdD1eaA7b242F3723a7d096B6CC693c7C6"
OLD_CONTRACT = "0x999F9A49A85e9D6E981cad42f197349f50172bEB"
EXECUTOR = "3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b"
# A random well-formed key; it signs nothing in these tests.
MINER_KEY = "0x" + "11" * 32
FINALIZED_HASH = "0x" + "f1" * 32
LATEST_HASH = "0x" + "1a" * 32
HEAD_HASH = "0x" + "2b" * 32


@pytest.fixture
def chain(monkeypatch):
    """Per-contract collateral and reclaim requests, read by address instead of over RPC.

    `reclaims` is the latest state and `reclaims_at[block hash]` the state at a block; a reclaim read names the
    block it was read at in `reclaim_blocks` (None: latest), and `after_reclaim_read` runs after each one."""
    state = SimpleNamespace(
        collateral={}, reclaims={}, reads=[], reclaims_at={}, reclaim_blocks=[],
        finalized_hash=FINALIZED_HASH, after_reclaim_read=lambda: None,
        collateral_at={}, collateral_blocks=[], latest_hash=LATEST_HASH, after_collateral_read=lambda: None,
    )

    async def get_executor_collateral(self, executor_uuid, block_hash=None):
        state.reads.append(self.contract_address)
        state.collateral_blocks.append(block_hash)
        collateral = state.collateral if block_hash is None else state.collateral_at.get(block_hash, state.collateral)
        found = Decimal(collateral.get(self.contract_address, 0))
        state.after_collateral_read()
        return found

    async def head_parent_hash(self):
        return state.latest_hash

    async def head_block_hashes(self):
        return HEAD_HASH, state.latest_hash

    async def get_reclaim_request(self, reclaim_request_id, block_hash=None):
        state.reads.append(self.contract_address)
        state.reclaim_blocks.append(block_hash)
        default = (bytes(16), "0x" + "00" * 20, 0, 0)
        reclaims = state.reclaims if block_hash is None else state.reclaims_at.get(block_hash, state.reclaims)
        found = reclaims.get((self.contract_address, reclaim_request_id), default)
        state.after_reclaim_read()
        return found

    async def finalized_block_hash(self):
        return state.finalized_hash

    monkeypatch.setattr(CollateralClient, "get_executor_collateral", get_executor_collateral)
    monkeypatch.setattr(CollateralClient, "get_reclaim_request", get_reclaim_request)
    monkeypatch.setattr(CollateralClient, "finalized_block_hash", finalized_block_hash)
    monkeypatch.setattr(CollateralClient, "head_parent_hash", head_parent_hash)
    monkeypatch.setattr(CollateralClient, "head_block_hashes", head_block_hashes)
    # the earlier-send check still runs, lock and temp record included, without asking finney for its chain ID
    monkeypatch.setattr(CollateralClient, "_pinned_chain_id", AsyncMock(return_value=964))
    return state


@pytest.fixture
def cli_services(monkeypatch):
    """The contract version each CliService is built for; its reclaim calls succeed."""
    import cli as cli_module

    built = []

    class FakeCliService:
        def __init__(self, private_key=None, with_executor_db=False, version="1.0.2"):
            built.append(version)

        async def reclaim_collateral(self, executor_uuid):
            return True

        async def finalize_reclaim_request(self, reclaim_request_id):
            return True

    monkeypatch.setattr(cli_module, "CliService", FakeCliService)
    return built


@pytest.mark.parametrize("holder,version", [(OLD_CONTRACT, "1.0.0"), (CONTRACT, "1.0.2")])
def test_reclaim_uses_the_contract_that_holds_the_collateral(chain, cli_services, holder, version):
    from cli import cli

    chain.collateral[holder] = "0.5"
    result = CliRunner().invoke(
        cli, ["reclaim-collateral", "--executor_uuid", EXECUTOR], input=f"{MINER_KEY}\n"
    )
    assert result.exit_code == 0, result.output
    assert cli_services == [version]


MALFORMED_KEYS = ["0x" + "ab" * 8, "not-a-hex-private-key", "zq" * 32]


@pytest.mark.parametrize("bad_key", MALFORMED_KEYS)
@pytest.mark.parametrize(
    "args",
    [
        ["reclaim-collateral", "--executor_uuid", EXECUTOR],
        ["reclaim-collateral", "--executor_uuid", EXECUTOR, "--contract", "1.0.2"],
        ["finalize-reclaim-request", "--reclaim-request-id", "7"],
        ["finalize-reclaim-request", "--reclaim-request-id", "7", "--contract", "1.0.2"],
        ["associate-eth"],
        ["get-eth-ss58-address"],
        ["transfer-tao-to-eth-address", "--amount", "1"],
        ["get-balance-of-eth-address"],
    ],
    ids=[
        "reclaim",
        "reclaim-contract",
        "finalize",
        "finalize-contract",
        "associate-eth",
        "get-eth-ss58-address",
        "transfer-tao",
        "get-balance",
    ],
)
def test_malformed_key_exits_1_and_logs_an_error_without_the_key(chain, cli_services, caplog, args, bad_key):
    from cli import cli

    if args[0] in ("reclaim-collateral", "finalize-reclaim-request"):
        result = CliRunner().invoke(cli, args, input=f"{bad_key}\n")
    else:
        result = CliRunner().invoke(cli, [*args, "--private-key", bad_key])

    assert result.exit_code == 1, result.output
    assert cli_services == []
    assert chain.reads == []
    assert "private key is malformed" in caplog.text
    assert bad_key not in caplog.text + result.output


async def test_remove_executor_refuses_when_only_the_head_holds_a_deposit(chain):
    """Review 5399092277 at c21535c: the parent H holds nothing and its child C, the head, holds a deposit, while
    `latest` reaches a backend still at H. Only the read pinned to C sees the deposit."""
    from core.utils import versions_holding_collateral

    chain.collateral = {}
    chain.collateral_at[LATEST_HASH] = {}
    chain.collateral_at[HEAD_HASH] = {CONTRACT: "0.01"}

    assert await versions_holding_collateral(EXECUTOR) == ["1.0.2"]
    assert HEAD_HASH in chain.collateral_blocks
