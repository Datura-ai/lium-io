"""CollateralClient against a fake JSON-RPC provider: what it signs and sends, and when it needs an RPC URL."""

import logging
import traceback
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

import pytest
import rlp
from click.testing import CliRunner
from eth_abi import encode
from eth_account import Account
from web3 import AsyncWeb3
from web3.exceptions import TimeExhausted
from web3.providers.async_base import AsyncBaseProvider

from core import collateral as collateral_module
from core.collateral import CollateralClient, CollateralConfigError, CollateralTransactionError

CONTRACT = "0x8A4023FdD1eaA7b242F3723a7d096B6CC693c7C6"
EXECUTOR = "3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b"
# A random well-formed key; it only signs transactions sent to the fake provider.
MINER_KEY = "0x" + "11" * 32
MINER = Account.from_key(MINER_KEY).address
TX_HASH = "0x" + "ab" * 32
BLOCK_HASH = "0x" + "cd" * 32
CHAIN_ID = 964
NONCE = 7
GAS_PRICE = 10_000_000_000
RPC_URL = "https://evm.example.invalid"


def selector(signature: str) -> str:
    return AsyncWeb3.keccak(text=signature)[:4].hex().removeprefix("0x")


def hex_encode(types, values) -> str:
    return "0x" + encode(types, values).hex()


class FakeProvider(AsyncBaseProvider):
    """Answers the JSON-RPC methods the client uses; eth_call is routed by function selector."""

    def __init__(
        self,
        calls=None,
        receipt_status=1,
        logs=None,
        revert_data=None,
        endpoint_uri=None,
        gas_price=GAS_PRICE,
    ):
        super().__init__()
        self.gas_price = gas_price
        self.endpoint_uri = endpoint_uri
        self.calls = calls or {}
        self.receipt_status = receipt_status
        self.logs = logs or []
        self.revert_data = revert_data
        self.requests = []
        self.sent = []

    async def is_connected(self, show_traceback: bool = False) -> bool:
        return True

    async def make_request(self, method, params):
        self.requests.append((method, params))
        if method == "eth_call":
            data = params[0]["data"].removeprefix("0x")
            if data[:8] in self.calls:
                return {"jsonrpc": "2.0", "id": 1, "result": self.calls[data[:8]]}
            error = {"code": 3, "message": "execution reverted", "data": self.revert_data or "0x"}
            return {"jsonrpc": "2.0", "id": 1, "error": error}
        if method == "eth_sendRawTransaction":
            self.sent.append(params[0])
            return {"jsonrpc": "2.0", "id": 1, "result": TX_HASH}
        results = {
            "eth_chainId": hex(CHAIN_ID),
            "eth_gasPrice": hex(self.gas_price),
            "eth_getTransactionCount": hex(NONCE),
            "eth_getTransactionReceipt": {
                "transactionHash": TX_HASH,
                "transactionIndex": "0x0",
                "blockHash": BLOCK_HASH,
                "blockNumber": "0x10",
                "from": MINER,
                "to": CONTRACT,
                "contractAddress": None,
                "cumulativeGasUsed": "0x5208",
                "gasUsed": "0x5208",
                "effectiveGasPrice": hex(GAS_PRICE),
                "status": hex(self.receipt_status),
                "logs": self.logs,
                "logsBloom": "0x" + "00" * 256,
                "type": "0x0",
            },
        }
        if method not in results:
            raise AssertionError(f"unexpected RPC call {method}")
        return {"jsonrpc": "2.0", "id": 1, "result": results[method]}


def open_reclaim(amount=10**17):
    return hex_encode(
        ["bytes16", "address", "uint256", "uint64"], [UUID(EXECUTOR).bytes, MINER, amount, 0]
    )


def reclaimed_log(reclaim_request_id=5, amount=10**17):
    return {
        "address": CONTRACT,
        "topics": [
            "0x"
            + AsyncWeb3.keccak(text="Reclaimed(uint256,bytes16,address,uint256)")
            .hex()
            .removeprefix("0x"),
            "0x" + f"{reclaim_request_id:064x}",
            "0x" + UUID(EXECUTOR).bytes.hex().ljust(64, "0"),
            "0x" + MINER.lower().removeprefix("0x").rjust(64, "0"),
        ],
        "data": hex_encode(["uint256"], [amount]),
        "blockNumber": "0x10",
        "blockHash": BLOCK_HASH,
        "transactionHash": TX_HASH,
        "transactionIndex": "0x0",
        "logIndex": "0x0",
        "removed": False,
    }


def client_with(provider, miner_key=MINER_KEY):
    client = CollateralClient(network="finney", contract_address=CONTRACT, miner_key=miner_key)
    client._w3 = AsyncWeb3(provider)
    return client


def decode_legacy(raw_transaction: str) -> dict:
    nonce, gas_price, gas, to, value, data, v, _, _ = rlp.decode(
        bytes.fromhex(raw_transaction.removeprefix("0x"))
    )
    return {
        "nonce": int.from_bytes(nonce, "big"),
        "gasPrice": int.from_bytes(gas_price, "big"),
        "gas": int.from_bytes(gas, "big"),
        "to": AsyncWeb3.to_checksum_address(to),
        "value": int.from_bytes(value, "big"),
        "data": data.hex(),
        "chainId": (int.from_bytes(v, "big") - 35) // 2,
    }


async def test_finalize_signs_one_transaction_with_the_node_nonce_gas_price_and_chain_id():
    provider = FakeProvider(
        calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()]
    )
    event = await client_with(provider).finalize_reclaim(5)

    assert event["args"]["reclaimRequestId"] == 5
    assert event["args"]["miner"] == MINER
    assert len(provider.sent) == 1
    raw = provider.sent[0]
    assert Account.recover_transaction(raw) == MINER
    assert decode_legacy(raw) == {
        "nonce": NONCE,
        "gasPrice": GAS_PRICE,
        "gas": collateral_module.GAS_LIMIT,
        "to": CONTRACT,
        "value": 0,
        "data": selector("finalizeReclaim(uint256)") + f"{5:064x}",
        "chainId": CHAIN_ID,
    }
    methods = [method for method, _ in provider.requests]
    assert ("eth_getTransactionCount", [MINER, "latest"]) in provider.requests
    assert methods.count("eth_sendRawTransaction") == 1


async def test_reverted_finalize_names_the_contract_error_and_not_the_key():
    provider = FakeProvider(
        calls={selector("reclaims(uint256)"): open_reclaim()},
        receipt_status=0,
        revert_data="0x" + selector("BeforeDenyTimeout()"),
    )
    with pytest.raises(CollateralTransactionError) as raised:
        await client_with(provider).finalize_reclaim(5)

    message = str(raised.value)
    assert message == f"Transaction {TX_HASH} reverted: BeforeDenyTimeout"
    assert MINER_KEY.removeprefix("0x")[:16] not in message
    replay = [params for method, params in provider.requests if method == "eth_call"][-1]
    assert replay[1] == "0x10"
    assert replay[0]["from"] == MINER
    assert replay[0]["data"].removeprefix("0x").startswith(selector("finalizeReclaim(uint256)"))
    assert set(replay[0]) <= {"from", "to", "data", "value"}


async def test_reverted_transaction_without_a_known_error_still_reports_the_hash():
    provider = FakeProvider(
        calls={selector("reclaims(uint256)"): open_reclaim()}, receipt_status=0, revert_data="0x"
    )
    with pytest.raises(CollateralTransactionError, match=f"Transaction {TX_HASH} reverted"):
        await client_with(provider).finalize_reclaim(5)


class ReplayFailsProvider(FakeProvider):
    """The replay eth_call of the sent transaction fails in transport, with the RPC URL in the error."""

    async def make_request(self, method, params):
        if method == "eth_call" and params[0]["data"].removeprefix("0x").startswith(
            selector("finalizeReclaim(uint256)")
        ):
            raise ConnectionError(f"Could not reach {RPC_URL}/?apikey=secret-rpc-key")
        return await super().make_request(method, params)


async def test_a_failed_replay_logs_its_error_class_and_not_the_url_or_the_key(caplog):
    provider = ReplayFailsProvider(
        calls={selector("reclaims(uint256)"): open_reclaim()}, receipt_status=0
    )
    with caplog.at_level(logging.WARNING, logger="core.collateral"):
        with pytest.raises(CollateralTransactionError) as raised:
            await client_with(provider).finalize_reclaim(5)

    assert str(raised.value) == f"Transaction {TX_HASH} reverted"
    logged = [r.getMessage() for r in caplog.records if r.name == "core.collateral"]
    assert logged == [
        "Could not replay the reverted transaction at block 16 to read its revert reason: "
        "ConnectionError"
    ]
    assert "secret-rpc-key" not in caplog.text
    assert RPC_URL not in caplog.text
    assert MINER_KEY.removeprefix("0x")[:16] not in caplog.text


async def test_send_without_a_key_sends_nothing():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    with pytest.raises(CollateralTransactionError, match="private key is required"):
        await client_with(provider, miner_key=None).finalize_reclaim(5)
    assert provider.sent == []


async def test_a_receipt_timeout_names_the_sent_transaction(monkeypatch, caplog):
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    client = client_with(provider)

    async def no_receipt(*_args, **_kwargs):
        raise TimeExhausted("no receipt")

    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", no_receipt)
    with caplog.at_level(logging.INFO, logger="core.collateral"):
        with pytest.raises(CollateralTransactionError, match=f"Transaction {TX_HASH} was sent"):
            await client.finalize_reclaim(5)

    assert len(provider.sent) == 1
    assert f"Sent transaction {TX_HASH}" in caplog.text


async def test_gas_price_quote_above_the_ceiling_signs_and_sends_nothing():
    ceiling_wei = collateral_module.DEFAULT_MAX_GAS_PRICE_GWEI * 10**9
    provider = FakeProvider(
        calls={selector("reclaims(uint256)"): open_reclaim()}, gas_price=ceiling_wei + 1
    )
    with pytest.raises(CollateralTransactionError, match="COLLATERAL_MAX_GAS_PRICE_GWEI"):
        await client_with(provider).finalize_reclaim(5)
    assert provider.sent == []
    assert "eth_getTransactionCount" not in [method for method, _ in provider.requests]


async def test_gas_price_quote_at_the_ceiling_is_signed():
    ceiling_wei = collateral_module.DEFAULT_MAX_GAS_PRICE_GWEI * 10**9
    provider = FakeProvider(
        calls={selector("reclaims(uint256)"): open_reclaim()},
        logs=[reclaimed_log()],
        gas_price=ceiling_wei,
    )
    await client_with(provider).finalize_reclaim(5)
    assert len(provider.sent) == 1
    assert decode_legacy(provider.sent[0])["gasPrice"] == ceiling_wei


async def test_configured_gas_price_ceiling_reaches_the_client(monkeypatch):
    from core import utils
    from core.config import settings

    monkeypatch.setattr(settings, "COLLATERAL_MAX_GAS_PRICE_GWEI", 5)
    client = utils.get_collateral_contract(miner_key=MINER_KEY)
    client._w3 = AsyncWeb3(FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}))
    with pytest.raises(CollateralTransactionError, match="5 gwei ceiling"):
        await client.finalize_reclaim(5)
    assert client._w3.provider.sent == []


def test_unknown_network_without_rpc_url_builds_a_client_without_a_connection():
    client = CollateralClient(network="archive", contract_address=CONTRACT, miner_key=MINER_KEY)
    assert client.contract_address == CONTRACT
    assert client.miner_address == MINER


async def test_contract_call_on_unknown_network_without_rpc_url_names_the_setting():
    client = CollateralClient(network="archive", contract_address=CONTRACT)
    with pytest.raises(CollateralConfigError, match="SUBTENSOR_EVM_RPC_URL") as raised:
        await client.get_executor_collateral(EXECUTOR)
    assert "archive" in str(raised.value)


async def test_contract_call_on_unknown_network_uses_the_rpc_url(monkeypatch):
    providers = []

    def fake_http_provider(endpoint_uri):
        providers.append(
            FakeProvider(
                calls={selector("collaterals(bytes16)"): hex_encode(["uint256"], [5 * 10**17])},
                endpoint_uri=endpoint_uri,
            )
        )
        return providers[-1]

    monkeypatch.setattr(collateral_module, "AsyncHTTPProvider", fake_http_provider)
    client = CollateralClient(network="archive", contract_address=CONTRACT, rpc_url=RPC_URL)
    assert await client.get_executor_collateral(EXECUTOR) == Decimal("0.5")
    assert [provider.endpoint_uri for provider in providers] == [RPC_URL]


@pytest.fixture
def archive_network(monkeypatch):
    from core.config import settings

    wallet = SimpleNamespace(get_hotkey=lambda: SimpleNamespace(ss58_address="5" + "C" * 47))
    monkeypatch.setattr(settings, "BITTENSOR_NETWORK", "archive")
    monkeypatch.setattr(settings, "SUBTENSOR_EVM_RPC_URL", None)
    monkeypatch.setattr(type(settings), "get_bittensor_wallet", lambda self: wallet)
    return settings


def test_non_collateral_command_runs_on_an_unknown_network(archive_network):
    from cli import cli

    result = CliRunner().invoke(cli, ["current-contract-version"])
    assert result.exit_code == 0, result.output
    assert CONTRACT in result.output


@pytest.mark.parametrize(
    "url,origin",
    [
        (
            "https://user:pw@evm.example.invalid:8443/v2/key?apikey=key",
            "https://evm.example.invalid:8443",
        ),
        ("https://evm.example.invalid/key", "https://evm.example.invalid"),
        ("evm.example.invalid/key", "<unparsed>"),
        (None, None),
    ],
)
def test_rpc_origin_keeps_only_the_scheme_and_host(url, origin):
    assert collateral_module.rpc_origin(url) == origin


RPC_SECRETS = (
    "fake-rpc-user",
    "fake-rpc-password",
    "fake-rpc-key-in-path",
    "fake-rpc-key-in-query",
)


@pytest.fixture
def rejecting_rpc():
    """A local JSON-RPC endpoint that answers every request 401, so web3 raises aiohttp's
    ClientResponseError, whose text holds the full request URL."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            hits.append(self.path)
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    url = (
        f"http://{RPC_SECRETS[0]}:{RPC_SECRETS[1]}@127.0.0.1:{port}"
        f"/v2/{RPC_SECRETS[2]}?apikey={RPC_SECRETS[3]}"
    )
    yield SimpleNamespace(url=url, origin=f"http://127.0.0.1:{port}", hits=hits)
    server.shutdown()
    server.server_close()


def assert_no_rpc_secret(text: str):
    for secret in RPC_SECRETS:
        assert secret not in text
    assert MINER_KEY.removeprefix("0x")[:16] not in text


@pytest.mark.parametrize(
    "args,stdin",
    [
        (
            ["reclaim-collateral", "--executor_uuid", EXECUTOR, "--private-key", MINER_KEY],
            None,
        ),
        (
            [
                "reclaim-collateral",
                "--executor_uuid",
                EXECUTOR,
                "--private-key",
                MINER_KEY,
                "--contract",
                "1.0.2",
            ],
            None,
        ),
        (
            ["finalize-reclaim-request", "--reclaim-request-id", "5", "--private-key", MINER_KEY],
            None,
        ),
        (["get-balance-of-eth-address", "--private-key", MINER_KEY], None),
        (
            [
                "finalize-reclaim-request",
                "--reclaim-request-id",
                "5",
                "--private-key",
                MINER_KEY,
                "--contract",
                "1.0.2",
            ],
            None,
        ),
        (["get-miner-collateral", "--contract", "1.0.2"], None),
        (
            [
                "get-executor-collateral",
                "--address",
                "192.0.2.10",
                "--port",
                "8001",
                "--contract",
                "1.0.2",
            ],
            None,
        ),
        (["get-reclaim-requests", "--contract", "1.0.2"], None),
        (["remove-executor", "--address", "192.0.2.10", "--port", "8001"], "y\n"),
    ],
    ids=lambda value: (
        value[0] + ("-contract" if "--contract" in value else "") if isinstance(value, list) else ""
    ),
)
def test_collateral_command_logs_leave_out_a_keyed_rpc_url(
    archive_network, rejecting_rpc, monkeypatch, caplog, args, stdin
):
    import services.cli_service as cli_service_module
    from cli import cli

    executor = SimpleNamespace(uuid=UUID(EXECUTOR))
    monkeypatch.setattr(archive_network, "SUBTENSOR_EVM_RPC_URL", rejecting_rpc.url)
    monkeypatch.setattr(cli_service_module, "get_db", lambda: iter([None]))
    monkeypatch.setattr(
        cli_service_module,
        "ExecutorDao",
        lambda session: SimpleNamespace(
            get_all_executors=lambda: [executor], find_one=lambda address, port: executor
        ),
    )
    monkeypatch.setattr(cli_service_module, "MinerSSHService", lambda: None)
    monkeypatch.setattr(cli_service_module, "ExecutorService", lambda **kwargs: None)

    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(cli, args, input=stdin)

    if result.exception is not None:
        assert isinstance(result.exception, SystemExit), "".join(
            traceback.format_exception(*result.exc_info)
        )
    assert rejecting_rpc.hits, "the command never reached the RPC endpoint"
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    logged = caplog.text + result.output + "".join(r.getMessage() for r in caplog.records)
    if result.exc_info is not None:
        logged += "".join(traceback.format_exception(*result.exc_info))
    assert_no_rpc_secret(logged)
    if args[0] != "remove-executor":
        assert f'"rpc_url": "{rejecting_rpc.origin}"' in logged
        assert '"error": "ClientResponseError"' in logged
    if "--contract" not in args and args[0] != "remove-executor":
        assert result.exit_code == 1


RPC_CLI_BOOTSTRAP = """
import sys
from types import SimpleNamespace
from unittest.mock import patch

patch("lium_core.shared_config.client.SharedConfigClient._fetch", return_value=None).start()
sys.path.insert(0, sys.argv.pop(1))
from core.config import Settings

wallet = SimpleNamespace(get_hotkey=lambda: SimpleNamespace(ss58_address="5" + "C" * 47))
Settings.get_bittensor_wallet = lambda self: wallet
from cli import cli

cli()
"""


@pytest.mark.parametrize(
    "args",
    [
        ["reclaim-collateral", "--executor_uuid", EXECUTOR, "--private-key", MINER_KEY],
        ["finalize-reclaim-request", "--reclaim-request-id", "5", "--private-key", MINER_KEY],
        # --contract skips detection: the send itself fails, and that must exit 1 too
        ["reclaim-collateral", "--executor_uuid", EXECUTOR, "--private-key", MINER_KEY, "--contract", "1.0.2"],
        ["finalize-reclaim-request", "--reclaim-request-id", "5", "--private-key", MINER_KEY, "--contract", "1.0.2"],
        ["get-balance-of-eth-address", "--private-key", MINER_KEY],
    ],
    ids=lambda args: "-".join(a for a in args if a in {"reclaim-collateral", "finalize-reclaim-request", "get-balance-of-eth-address", "--contract"}),
)
def test_collateral_command_process_output_leaves_out_a_keyed_rpc_url(rejecting_rpc, args):
    """The real CLI in its own process: nothing it writes to stdout or stderr, a traceback
    included, holds the RPC URL's userinfo, path or query."""
    import os
    import pathlib
    import subprocess
    import sys

    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    env = {
        **os.environ,
        "BITTENSOR_NETWORK": "archive",
        "SUBTENSOR_EVM_RPC_URL": rejecting_rpc.url,
        "PYTHONWARNINGS": "default",
    }
    result = subprocess.run(
        [sys.executable, "-c", RPC_CLI_BOOTSTRAP, str(src), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=src,
    )

    output = result.stdout + result.stderr
    assert rejecting_rpc.hits, output
    assert_no_rpc_secret(output)
    assert "Traceback" not in output
    assert result.returncode == 1
    assert f'"rpc_url": "{rejecting_rpc.origin}"' in output
    assert '"error": "ClientResponseError"' in output


def test_collateral_command_on_an_unknown_network_names_the_setting(archive_network, caplog):
    from cli import cli

    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(
            cli, ["reclaim-collateral", "--executor_uuid", EXECUTOR, "--private-key", MINER_KEY]
        )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "SUBTENSOR_EVM_RPC_URL" in caplog.text
    assert MINER_KEY.removeprefix("0x")[:16] not in caplog.text + result.output


async def test_finalize_names_a_closed_or_unknown_reclaim_request():
    """reclaims(id) answers zeros both for a finalized request and for one never opened."""
    reclaims = selector("reclaims(uint256)")
    zero = hex_encode(
        ["bytes16", "address", "uint256", "uint64"], [bytes(16), "0x" + "00" * 20, 0, 0]
    )
    provider = FakeProvider(calls={reclaims: zero})
    client = client_with(provider)

    with pytest.raises(CollateralTransactionError) as raised:
        await client.finalize_reclaim(9)
    assert "No open reclaim request 9 on this contract" in str(raised.value)
    assert "never opened" in str(raised.value)
    assert provider.sent == []
