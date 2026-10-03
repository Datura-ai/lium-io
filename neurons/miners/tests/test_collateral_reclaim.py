"""The miner's reclaim path for executors that still hold collateral on the contract."""

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from click.testing import CliRunner
from eth_account import Account

from core.collateral import CollateralClient, executor_uuid_bytes, h160_to_ss58

CONTRACT = "0x8A4023FdD1eaA7b242F3723a7d096B6CC693c7C6"
OLD_CONTRACT = "0x999F9A49A85e9D6E981cad42f197349f50172bEB"
EXECUTOR = "3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b"
# A random well-formed key; it signs nothing in these tests.
MINER_KEY = "0x" + "11" * 32
FINALIZED_HASH = "0x" + "f1" * 32
LATEST_HASH = "0x" + "1a" * 32


def test_h160_to_ss58_maps_an_evm_address_to_its_mirror_account():
    assert h160_to_ss58(CONTRACT) == "5CZfCUByDkz17Pdk2yBvuhe7kFhZBkU8LWqfKyodX2MaEFBR"
    assert h160_to_ss58(CONTRACT.removeprefix("0x")) == h160_to_ss58(CONTRACT)


def test_executor_uuid_bytes_is_the_16_byte_contract_key():
    assert executor_uuid_bytes(EXECUTOR) == UUID(EXECUTOR).bytes
    with pytest.raises(ValueError):
        executor_uuid_bytes("0xabcd")


def test_client_encodes_the_reclaim_and_finalize_calls():
    client = CollateralClient(
        network="finney", contract_address=CONTRACT.lower(), miner_key=MINER_KEY
    )
    assert client.contract_address == CONTRACT
    assert client.miner_address is not None

    def selector(signature: str) -> str:
        return "0x" + client.w3.keccak(text=signature)[:4].hex().removeprefix("0x")

    reclaim = client.contract.encode_abi(
        fn_name="reclaimCollateral",
        args=[executor_uuid_bytes(EXECUTOR), "Manual reclaim", bytes(16)],
    )
    finalize = client.contract.encode_abi(fn_name="finalizeReclaim", args=[7])
    assert reclaim.startswith(selector("reclaimCollateral(bytes16,string,bytes16)"))
    assert UUID(EXECUTOR).bytes.hex() in reclaim
    assert finalize == selector("finalizeReclaim(uint256)") + f"{7:064x}"


def test_cli_keeps_reclaim_commands_and_registers_executors_without_a_deposit():
    from cli import cli

    commands = set(cli.commands)
    assert {
        "reclaim-collateral", "finalize-reclaim-request", "get-reclaim-requests",
        "get-executor-collateral", "get-miner-collateral", "transfer-tao-to-eth-address",
    } <= commands
    assert "deposit-collateral" not in commands

    help_text = CliRunner().invoke(cli, ["add-executor", "--help"]).output
    assert "--price" in help_text
    assert "--deposit-amount" not in help_text


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


def test_contract_versions_cover_both_deployed_contracts():
    from core.config import settings
    from core.utils import get_collateral_contract

    assert {v["address"] for v in settings.CONTRACT_VERSIONS.values()} == {CONTRACT, OLD_CONTRACT}
    assert get_collateral_contract(version="1.0.0").contract_address == OLD_CONTRACT
    assert get_collateral_contract().contract_address == CONTRACT


@pytest.mark.parametrize("holder,version", [(OLD_CONTRACT, "1.0.0"), (CONTRACT, "1.0.2")])
def test_reclaim_uses_the_contract_that_holds_the_collateral(chain, cli_services, holder, version):
    from cli import cli

    chain.collateral[holder] = "0.5"
    result = CliRunner().invoke(
        cli, ["reclaim-collateral", "--executor_uuid", EXECUTOR], input=f"{MINER_KEY}\n"
    )
    assert result.exit_code == 0, result.output
    assert cli_services == [version]


def test_reclaim_contract_option_skips_detection(chain, cli_services):
    from cli import cli

    args = ["reclaim-collateral", "--executor_uuid", EXECUTOR]
    result = CliRunner().invoke(cli, [*args, "--contract", "1.0.0"], input=f"{MINER_KEY}\n")
    assert result.exit_code == 0, result.output
    assert cli_services == ["1.0.0"]
    assert chain.reads == []


@pytest.mark.parametrize(
    "args",
    [
        ["reclaim-collateral", "--executor_uuid", EXECUTOR],
        ["finalize-reclaim-request", "--reclaim-request-id", "7"],
    ],
    ids=["no collateral", "no open request"],
)
def test_nothing_to_do_on_any_contract_exits_1_and_sends_nothing(chain, cli_services, args):
    from cli import cli

    result = CliRunner().invoke(cli, args, input=f"{MINER_KEY}\n")
    # the same exit code as the contract rejecting the call when --contract names one
    assert result.exit_code == 1, result.output
    assert cli_services == []
    assert set(chain.reads) == {CONTRACT, OLD_CONTRACT}


def test_finalize_uses_the_contract_with_this_miners_open_request(chain, cli_services):
    from cli import cli

    miner = Account.from_key(MINER_KEY).address
    chain.reclaims[(OLD_CONTRACT, 7)] = (UUID(EXECUTOR).bytes, miner, 10**17, 0)
    # the same id on the current contract belongs to another miner
    chain.reclaims[(CONTRACT, 7)] = (UUID(EXECUTOR).bytes, "0x" + "22" * 20, 10**17, 0)
    result = CliRunner().invoke(
        cli, ["finalize-reclaim-request", "--reclaim-request-id", "7"], input=f"{MINER_KEY}\n"
    )
    assert result.exit_code == 0, result.output
    assert cli_services == ["1.0.0"]


@pytest.mark.parametrize(
    "state_change,detected,prompted",
    [
        ("none", ["1.0.0"], False),
        # open on both at the finalized block; a read at latest sees the 1.0.2 one only after its own read
        ("both_open_between_reads", ["1.0.2", "1.0.0"], True),
        # opens on 1.0.2 after the finalized block: the answer is that one block's state
        ("1.0.2_opens_after_the_block", ["1.0.0"], False),
    ],
)
def test_finalize_reads_every_contract_at_one_finalized_block(chain, cli_services, state_change, detected, prompted):
    """Review of 7cd21ec: request 5 is open on 1.0.0, and on 1.0.2 by the time the second read runs. Reads at
    latest one after another reported 1.0.0 alone, so the CLI picked it without asking. Read at one block, the
    answer is never a mix of two chain states."""
    from cli import cli
    from core.utils import versions_with_open_reclaim

    miner = Account.from_key(MINER_KEY).address
    request = (UUID(EXECUTOR).bytes, miner, 10**17, 0)
    chain.reclaims[(OLD_CONTRACT, 5)] = request
    chain.reclaims_at[FINALIZED_HASH] = {(OLD_CONTRACT, 5): request}
    if state_change == "both_open_between_reads":
        chain.reclaims_at[FINALIZED_HASH][(CONTRACT, 5)] = request
    if state_change != "none":
        chain.after_reclaim_read = lambda: chain.reclaims.__setitem__((CONTRACT, 5), request)

    assert asyncio.run(versions_with_open_reclaim(5, miner)) == detected
    assert chain.reclaim_blocks == [FINALIZED_HASH, FINALIZED_HASH]

    result = CliRunner().invoke(
        cli, ["finalize-reclaim-request", "--reclaim-request-id", "5"], input=f"{MINER_KEY}\n2\n"
    )
    assert result.exit_code == 0, result.output
    assert ("Select contract version" in result.output) is prompted
    assert cli_services == ["1.0.0"]


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


@pytest.mark.parametrize("holder,removed", [(OLD_CONTRACT, False), (None, True)])
async def test_remove_executor_checks_every_contract_version(chain, holder, removed):
    import logging

    from services.cli_service import CliService

    if holder:
        chain.collateral[holder] = "0.01"
    deleted = []
    service = CliService.__new__(CliService)
    service.logger = logging.getLogger("test")
    service.executor_dao = SimpleNamespace(
        find_one=lambda address, port: SimpleNamespace(uuid=UUID(EXECUTOR)),
        delete_by_address_port=lambda address, port: deleted.append((address, port)),
    )
    assert await service.remove_executor("192.0.2.10", 8001) is removed
    assert bool(deleted) is removed


async def test_remove_executor_reads_every_contract_at_one_block(chain):
    """Review of e6f4b88: the old contract holds 0.01 TAO and the current one none. After the first read a deposit
    lands on the current contract and the old reclaim finalizes, so reads at latest one after another see zero on
    both and the executor's record is deleted while the current contract holds its collateral."""
    import logging

    from core.utils import versions_holding_collateral
    from services.cli_service import CliService

    chain.collateral = {OLD_CONTRACT: "0.01"}
    chain.collateral_at[LATEST_HASH] = {OLD_CONTRACT: "0.01"}
    chain.after_collateral_read = lambda: chain.collateral.update({CONTRACT: "0.01", OLD_CONTRACT: "0"})

    assert await versions_holding_collateral(EXECUTOR) == ["1.0.2", "1.0.0"]
    assert chain.collateral_blocks == [LATEST_HASH, None, LATEST_HASH, None]

    deleted = []
    service = CliService.__new__(CliService)
    service.logger = logging.getLogger("test")
    service.executor_dao = SimpleNamespace(
        find_one=lambda address, port: SimpleNamespace(uuid=UUID(EXECUTOR)),
        delete_by_address_port=lambda address, port: deleted.append((address, port)),
    )
    assert await service.remove_executor("192.0.2.10", 8001) is False
    assert deleted == []


async def test_remove_executor_refuses_when_only_the_latest_read_sees_a_deposit(chain):
    """Review 5394742345 at 4ac762f: a backend that does not know the pinned hash can answer from another state
    with no deposit on either contract, while `latest` shows the current contract's deposit."""
    import logging

    from core.utils import versions_holding_collateral
    from services.cli_service import CliService

    chain.collateral = {CONTRACT: "0.01"}
    chain.collateral_at[LATEST_HASH] = {}

    assert await versions_holding_collateral(EXECUTOR) == ["1.0.2"]

    deleted = []
    service = CliService.__new__(CliService)
    service.logger = logging.getLogger("test")
    service.executor_dao = SimpleNamespace(
        find_one=lambda address, port: SimpleNamespace(uuid=UUID(EXECUTOR)),
        delete_by_address_port=lambda address, port: deleted.append((address, port)),
    )
    assert await service.remove_executor("192.0.2.10", 8001) is False
    assert deleted == []
