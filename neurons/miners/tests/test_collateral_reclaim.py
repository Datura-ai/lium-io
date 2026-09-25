"""The miner's reclaim path for executors that still hold collateral on the contract."""

from decimal import Decimal
from types import SimpleNamespace
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


def test_h160_to_ss58_maps_an_evm_address_to_its_mirror_account():
    assert h160_to_ss58(CONTRACT) == "5CZfCUByDkz17Pdk2yBvuhe7kFhZBkU8LWqfKyodX2MaEFBR"
    assert h160_to_ss58(CONTRACT.removeprefix("0x")) == h160_to_ss58(CONTRACT)


def test_executor_uuid_bytes_is_the_16_byte_contract_key():
    assert executor_uuid_bytes(EXECUTOR) == UUID(EXECUTOR).bytes
    assert executor_uuid_bytes(UUID(EXECUTOR)) == UUID(EXECUTOR).bytes
    assert executor_uuid_bytes("0xabcd") == bytes.fromhex("abcd") + b"\0" * 14


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
        "reclaim-collateral",
        "finalize-reclaim-request",
        "get-reclaim-requests",
        "get-executor-collateral",
        "get-miner-collateral",
        "transfer-tao-to-eth-address",
    } <= commands
    assert "deposit-collateral" not in commands

    help_text = CliRunner().invoke(cli, ["add-executor", "--help"]).output
    assert "--price" in help_text
    assert "--deposit-amount" not in help_text


@pytest.fixture
def chain(monkeypatch):
    """Per-contract collateral and reclaim requests, read by address instead of over RPC."""
    state = SimpleNamespace(collateral={}, reclaims={}, reads=[])

    async def get_executor_collateral(self, executor_uuid):
        state.reads.append(self.contract_address)
        return Decimal(state.collateral.get(self.contract_address, 0))

    async def get_reclaim_request(self, reclaim_request_id):
        state.reads.append(self.contract_address)
        default = (bytes(16), "0x" + "00" * 20, 0, 0)
        return state.reclaims.get((self.contract_address, reclaim_request_id), default)

    monkeypatch.setattr(CollateralClient, "get_executor_collateral", get_executor_collateral)
    monkeypatch.setattr(CollateralClient, "get_reclaim_request", get_reclaim_request)
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
        cli, ["reclaim-collateral", "--executor_uuid", EXECUTOR, "--private-key", MINER_KEY]
    )
    assert result.exit_code == 0, result.output
    assert cli_services == [version]


def test_reclaim_contract_option_skips_detection(chain, cli_services):
    from cli import cli

    result = CliRunner().invoke(
        cli,
        [
            "reclaim-collateral",
            "--executor_uuid",
            EXECUTOR,
            "--private-key",
            MINER_KEY,
            "--contract",
            "1.0.0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert cli_services == ["1.0.0"]
    assert chain.reads == []


def test_reclaim_without_collateral_anywhere_sends_nothing(chain, cli_services):
    from cli import cli

    result = CliRunner().invoke(
        cli, ["reclaim-collateral", "--executor_uuid", EXECUTOR, "--private-key", MINER_KEY]
    )
    assert result.exit_code == 0, result.output
    assert cli_services == []
    assert sorted(chain.reads) == sorted([CONTRACT, OLD_CONTRACT])


def test_finalize_uses_the_contract_with_this_miners_open_request(chain, cli_services):
    from cli import cli

    miner = Account.from_key(MINER_KEY).address
    chain.reclaims[(OLD_CONTRACT, 7)] = (UUID(EXECUTOR).bytes, miner, 10**17, 0)
    # the same id on the current contract belongs to another miner
    chain.reclaims[(CONTRACT, 7)] = (UUID(EXECUTOR).bytes, "0x" + "22" * 20, 10**17, 0)
    result = CliRunner().invoke(
        cli, ["finalize-reclaim-request", "--reclaim-request-id", "7", "--private-key", MINER_KEY]
    )
    assert result.exit_code == 0, result.output
    assert cli_services == ["1.0.0"]


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
