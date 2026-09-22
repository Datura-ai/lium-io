"""The miner's reclaim path for executors that still hold collateral on the contract."""

from uuid import UUID

from click.testing import CliRunner

from core.collateral import CollateralClient, executor_uuid_bytes, h160_to_ss58

CONTRACT = "0x8A4023FdD1eaA7b242F3723a7d096B6CC693c7C6"
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

    reclaim = client.contract.encodeABI(
        fn_name="reclaimCollateral",
        args=[executor_uuid_bytes(EXECUTOR), "Manual reclaim", bytes(16)],
    )
    finalize = client.contract.encodeABI(fn_name="finalizeReclaim", args=[7])
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
