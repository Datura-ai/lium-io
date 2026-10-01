"""CollateralClient against a fake JSON-RPC provider: what it signs and sends, and when it needs an RPC URL."""

import logging
import traceback
from json import JSONDecodeError
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

import pytest
import rlp
from click.testing import CliRunner
from eth_abi import encode
from eth_account import Account
from web3 import AsyncWeb3
from web3.providers.async_base import AsyncBaseProvider

from core import collateral as collateral_module
from core.collateral import (
    CollateralClient,
    CollateralConfigError,
    CollateralOutcomeUnknownError,
    CollateralTransactionError,
)

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
KEYED_RPC_URL = "https://fake-rpc-user:fake-rpc-password@evm.example.invalid/v2/fake-rpc-key-in-path?apikey=fake-rpc-key-in-query"


def selector(signature: str) -> str:
    return AsyncWeb3.keccak(text=signature)[:4].hex().removeprefix("0x")


def hex_encode(types, values) -> str:
    return "0x" + encode(types, values).hex()


SENDS = {selector("finalizeReclaim(uint256)"), selector("reclaimCollateral(bytes16,string,bytes16)")}


def bloom_of(logs) -> str:
    """A block's logs bloom built as the yellow paper's M3:2048, as one 2048-bit big-endian integer."""
    bits = 0
    for log in logs:
        for value in [log["address"], *log["topics"]]:
            digest = AsyncWeb3.keccak(hexstr=value)
            for i in (0, 2, 4):
                bits |= 1 << (int.from_bytes(digest[i : i + 2], "big") & 2047)
    return "0x" + bits.to_bytes(256, "big").hex()


class FakeProvider(AsyncBaseProvider):
    """Answers the JSON-RPC methods the client uses; eth_call is routed by function selector.

    A broadcast transaction is mined when `mine_sent` is on. Only mined hashes have a receipt, except TX_HASH, the
    hash every broadcast answers with, whose receipt is the one a send waits for.
    """

    SEND_ERRORS = {
        "refused": "insufficient funds",
        "known": "already known",
        "unknown account": "unknown account",
        "keyed refusal": f"insufficient funds for gas * price + value at {KEYED_RPC_URL}",
    }

    def __init__(
        self,
        calls=None,
        receipt_status=1,
        logs=None,
        revert_data=None,
        endpoint_uri=None,
        gas_price=GAS_PRICE,
        pending_nonce=NONCE,
        simulate_revert=None,
    ):
        super().__init__()
        self.pending_nonce = pending_nonce
        self.simulate_revert = simulate_revert
        self.gas_price = gas_price
        self.endpoint_uri = endpoint_uri
        self.calls = calls or {}
        self.receipt_status = receipt_status
        self.logs = logs or []
        self.revert_data = revert_data
        self.requests = []
        self.sent = []
        self.chain_id = CHAIN_ID
        self.send_error = None
        self.nonce = NONCE
        self.mine_sent = True
        self.mined = set()
        self.block_number = 16
        self.logs_by_fork = []
        # the "finalized" tag's number when it trails the head; the chain's block hash at a number, when it is
        # not the one receipts name
        self.finalized_number = None
        self.canonical_hashes = {}
        # a lagging backend on another fork: the hash it answers a read by number with
        self.fork_hashes = {}
        # hashes whose receipt request is answered with TX_HASH's receipt
        self.receipts_of_another = set()
        # block hashes a lagging backend answers a by-hash log read for with "unknown block"
        self.unknown_block_hashes = set()

    async def is_connected(self, show_traceback: bool = False) -> bool:
        return True

    def chain_hash(self, number: int) -> str:
        if number in self.canonical_hashes:
            return self.canonical_hashes[number]
        return BLOCK_HASH if number == 16 else "0x" + f"{number:064x}"

    def logs_in(self, block_hash: str) -> list:
        return [log for log in self.logs if log["blockHash"] == block_hash]

    def block(self, number: int) -> dict:
        block_hash = self.chain_hash(number)
        return {
            "number": hex(number),
            "hash": block_hash,
            "parentHash": self.chain_hash(number - 1),
            "logsBloom": bloom_of(self.logs_in(block_hash)),
        }

    def fork_block(self, number: int) -> dict:
        parent = self.fork_hashes.get(number - 1, self.chain_hash(number - 1))
        return {"number": hex(number), "hash": self.fork_hashes[number], "parentHash": parent, "logsBloom": bloom_of([])}

    async def make_request(self, method, params):
        self.requests.append((method, params))
        if method == "eth_call":
            data = params[0]["data"].removeprefix("0x")
            if params[1] == "latest" and data[:8] in SENDS:
                if self.simulate_revert is None:
                    return {"jsonrpc": "2.0", "id": 1, "result": "0x"}
                error = {"code": 3, "message": "execution reverted", "data": self.simulate_revert}
                return {"jsonrpc": "2.0", "id": 1, "error": error}
            if data[:8] in self.calls:
                return {"jsonrpc": "2.0", "id": 1, "result": self.calls[data[:8]]}
            error = {"code": 3, "message": "execution reverted", "data": self.revert_data or "0x"}
            return {"jsonrpc": "2.0", "id": 1, "error": error}
        if method == "eth_getTransactionCount":
            nonce = self.pending_nonce if params[1] == "pending" else self.nonce
            return {"jsonrpc": "2.0", "id": 1, "result": hex(nonce)}
        if method == "eth_sendRawTransaction":
            if self.send_error in self.SEND_ERRORS:
                return {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": self.SEND_ERRORS[self.send_error]}}
            self.sent.append(params[0])
            if self.mine_sent:
                self.mined.add(AsyncWeb3.keccak(hexstr=params[0]).hex())
            if self.send_error == "lost":
                raise ConnectionError(f"Could not reach {RPC_URL}/?apikey=secret-rpc-key")
            # a gateway that broadcast the bytes and answers with another transaction's hash
            answered = TX_HASH if self.send_error == "wrong_hash" else AsyncWeb3.keccak(hexstr=params[0]).hex()
            return {"jsonrpc": "2.0", "id": 1, "result": answered}
        if method == "eth_blockNumber":
            return {"jsonrpc": "2.0", "id": 1, "result": hex(self.block_number)}
        if method == "eth_getBlockByNumber":
            tag = params[0]
            if tag == "finalized" and self.finalized_number is not None:
                return {"jsonrpc": "2.0", "id": 1, "result": self.block(self.finalized_number)}
            if isinstance(tag, str) and tag.startswith("0x"):
                number = int(tag, 16)
                if number in self.fork_hashes:
                    return {"jsonrpc": "2.0", "id": 1, "result": self.fork_block(number)}
                return {"jsonrpc": "2.0", "id": 1, "result": self.block(number)}
            return {"jsonrpc": "2.0", "id": 1, "result": self.block(self.block_number)}
        if method == "eth_getBlockByHash":
            candidates = [n for n, h in self.canonical_hashes.items() if h == params[0]] + [16, int(params[0], 16)]
            found = next(
                (self.block(n) for n in candidates if n <= self.block_number and self.chain_hash(n) == params[0]), None
            )
            return {"jsonrpc": "2.0", "id": 1, "result": found}
        if method == "eth_getLogs" and "blockHash" in params[0]:
            if params[0]["blockHash"] in self.unknown_block_hashes:
                return {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "unknown block"}}
            return {"jsonrpc": "2.0", "id": 1, "result": self.logs_in(params[0]["blockHash"])}
        if method == "eth_getLogs":
            logs = self.logs_by_fork.pop(0) if self.logs_by_fork else self.logs
            return {"jsonrpc": "2.0", "id": 1, "result": logs}
        results = {
            "eth_chainId": hex(self.chain_id),
            "eth_gasPrice": hex(self.gas_price),
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
        if method == "eth_getTransactionReceipt" and params[0] in self.receipts_of_another:
            return {"jsonrpc": "2.0", "id": 1, "result": results[method]}
        if method == "eth_getTransactionReceipt" and params[0] != TX_HASH:
            if params[0] not in self.mined:
                return {"jsonrpc": "2.0", "id": 1, "result": None}
            results[method] = {**results[method], "transactionHash": params[0]}
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


def started_log(reclaim_request_id=5, amount=10**17, url="Manual reclaim"):
    return {
        "address": CONTRACT,
        "topics": [
            "0x"
            + AsyncWeb3.keccak(text="ReclaimProcessStarted(uint256,bytes16,address,uint256,uint64,string,bytes16)")
            .hex()
            .removeprefix("0x"),
            "0x" + f"{reclaim_request_id:064x}",
            "0x" + UUID(EXECUTOR).bytes.hex().ljust(64, "0"),
            "0x" + MINER.lower().removeprefix("0x").rjust(64, "0"),
        ],
        "data": hex_encode(["uint256", "uint64", "string", "bytes16"], [amount, 0, url, bytes(16)]),
        "blockNumber": "0x10",
        "blockHash": BLOCK_HASH,
        "transactionHash": TX_HASH,
        "transactionIndex": "0x0",
        "logIndex": "0x0",
        "removed": False,
    }


def sent_hash(provider, index=0) -> str:
    return AsyncWeb3.keccak(hexstr=provider.sent[index]).hex()


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


@pytest.mark.parametrize(
    "revert_data,message",
    [
        ("0x" + selector("BeforeDenyTimeout()"), "Transaction {} reverted: BeforeDenyTimeout"),
        ("0x", "Transaction {} reverted: execution reverted"),
    ],
    ids=["known-error", "no-known-error"],
)
async def test_reverted_finalize_names_the_contract_error_and_not_the_key(revert_data, message):
    provider = FakeProvider(
        calls={selector("reclaims(uint256)"): open_reclaim()},
        receipt_status=0,
        revert_data=revert_data,
    )
    with pytest.raises(CollateralTransactionError) as raised:
        await client_with(provider).finalize_reclaim(5)

    message = message.format(sent_hash(provider))
    assert str(raised.value) == message
    assert MINER_KEY.removeprefix("0x")[:16] not in message
    replay = [params for method, params in provider.requests if method == "eth_call"][-1]
    assert replay[1] == "0x10"
    assert replay[0]["from"] == MINER
    assert replay[0]["data"].removeprefix("0x").startswith(selector("finalizeReclaim(uint256)"))
    assert set(replay[0]) <= {"from", "to", "data", "value"}


class ReplayFailsProvider(FakeProvider):
    """The replay eth_call of the sent transaction fails in transport, with the RPC URL in the error."""

    async def make_request(self, method, params):
        if method == "eth_call" and params[1] != "latest" and params[0]["data"].removeprefix("0x").startswith(
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

    assert str(raised.value) == f"Transaction {sent_hash(provider)} reverted"
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


async def lost_receipt(*_args, **_kwargs):
    raise ConnectionError(f"Could not reach {RPC_URL}/?apikey=secret-rpc-key")


async def test_a_lost_receipt_is_an_unknown_outcome_and_the_retry_reads_it_and_sends_nothing_new(
    monkeypatch, caplog
):
    provider = FakeProvider()
    provider.mine_sent = False
    client = client_with(provider)

    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", lost_receipt)
    with caplog.at_level(logging.INFO, logger="core.collateral"):
        with pytest.raises(CollateralOutcomeUnknownError, match="was sent but its receipt") as raised:
            await client.reclaim_collateral(EXECUTOR)
    assert f"Transaction {sent_hash(provider)} was sent" in str(raised.value)
    assert "outcome is unknown" in str(raised.value)
    assert "secret-rpc-key" not in str(raised.value)
    assert f"Sent transaction {sent_hash(provider)}" in caplog.text

    # not mined, nonce unused: the same bytes go out again, and nothing new is signed
    with pytest.raises(CollateralOutcomeUnknownError, match="broadcast again and no new transaction was sent"):
        await client.reclaim_collateral(EXECUTOR)
    assert provider.sent == [provider.sent[0]] * 2

    # mined: the retry reports it and sends nothing
    provider.mined.add(AsyncWeb3.keccak(hexstr=provider.sent[0]).hex())
    provider.nonce = NONCE + 1
    with pytest.raises(CollateralTransactionError, match="succeeded in block 16; no new transaction was sent"):
        await client.reclaim_collateral(EXECUTOR)
    assert len(provider.sent) == 2


async def test_a_send_the_mempool_dropped_goes_out_again_as_the_same_bytes_and_later_work_is_not_blocked(
    monkeypatch,
):
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    provider.mine_sent = False
    client = client_with(provider)

    wait_for_receipt = client.w3.eth.wait_for_transaction_receipt
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", lost_receipt)
    with pytest.raises(CollateralOutcomeUnknownError):
        await client.finalize_reclaim(5)
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", wait_for_receipt)

    # evicted unmined: the nonce never advanced and nothing is pending; the retry broadcasts the same bytes,
    # which are mined this time, and reports that
    provider.mine_sent = True
    with pytest.raises(CollateralTransactionError, match="sent earlier, succeeded in block 16"):
        await client.finalize_reclaim(5)
    assert provider.sent == [provider.sent[0]] * 2

    # request 5 is closed now; another request still goes out, on the next nonce
    provider.nonce = provider.pending_nonce = NONCE + 1
    await client.finalize_reclaim(6)
    assert [decode_legacy(raw)["nonce"] for raw in provider.sent] == [NONCE, NONCE, NONCE + 1]


async def test_an_already_known_answer_is_an_unknown_outcome_and_the_retry_sends_the_same_bytes():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    provider.send_error = "known"
    client = client_with(provider)

    with pytest.raises(CollateralOutcomeUnknownError, match=r"may have been sent.*\(already known\)") as raised:
        await client.finalize_reclaim(5)
    assert "no transaction was sent" not in str(raised.value)
    assert provider.sent == []

    provider.send_error = None
    with pytest.raises(CollateralTransactionError, match="sent earlier, succeeded"):
        await client.finalize_reclaim(5)
    assert len(provider.sent) == 1


@pytest.mark.parametrize(
    "failure,first_status",
    [
        ("lost_receipt", 0),
        ("bad_json", 0),
        ("lost_receipt", 1),
        ("lost_answer", 1),
        ("upstream_timeout", 1),
        ("unknown_account", 1),
        ("nonce_too_low", 1),
        ("replacement_underpriced", 1),
        ("below_base_fee", 1),
        ("insufficient_funds", 1),
        ("wrong_hash_reply", 1),
    ],
)
async def test_an_unknown_outcome_never_uses_the_next_nonce(failure, first_status, monkeypatch):
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, receipt_status=first_status)
    client = client_with(provider)
    eth = client.w3.eth
    original_send = eth.send_raw_transaction

    async def unavailable(*_args, **_kwargs):
        raise TimeoutError()

    async def receipt_or_timeout(tx_hash, **_kwargs):
        # the answered hash (TX_HASH) has a mined receipt; the signed transaction has none
        receipt = await eth.get_transaction_receipt(tx_hash)
        if receipt is None:
            raise TimeoutError()
        return receipt

    if failure == "wrong_hash_reply":
        provider.send_error = "wrong_hash"
        provider.mine_sent = False

    def accepted_then(error):
        async def send(raw):
            await original_send(raw)
            raise error

        return send

    answers = {
        "bad_json": JSONDecodeError("bad response", "x", 0),
        "lost_answer": ConnectionError(f"Could not reach {KEYED_RPC_URL}"),
        # a gateway that forwarded the transaction and then timed out upstream
        "upstream_timeout": ValueError({"code": -32000, "message": f"upstream timeout at {KEYED_RPC_URL}"}),
        "unknown_account": ValueError({"code": -32000, "message": "unknown account"}),
        # a resend (web3's retry middleware, or a gateway's) of a transaction that already landed
        "nonce_too_low": ValueError({"code": -32000, "message": "nonce too low"}),
        # a gateway that forwarded the bytes to one upstream and answers with another upstream's refusal
        "replacement_underpriced": ValueError({"code": -32000, "message": "replacement transaction underpriced"}),
        "below_base_fee": ValueError({"code": -32000, "message": "gas price less than block base fee"}),
        "insufficient_funds": ValueError({"code": -32000, "message": "insufficient funds for gas * price + value"}),
    }
    if failure in answers:
        monkeypatch.setattr(eth, "send_raw_transaction", accepted_then(answers[failure]))
    wait = receipt_or_timeout if failure == "wrong_hash_reply" else unavailable
    monkeypatch.setattr(eth, "wait_for_transaction_receipt", wait)
    with pytest.raises(CollateralOutcomeUnknownError) as raised:
        await client.finalize_reclaim(5)
    # the error names the signed hash, whatever hash the RPC answered with
    assert sent_hash(provider).removeprefix("0x") in str(raised.value)
    assert client._read_sent_record(CHAIN_ID)["hash"] == sent_hash(provider)
    provider.send_error = None
    assert "no transaction was sent" not in str(raised.value)
    assert_no_rpc_secret(str(raised.value))

    monkeypatch.setattr(eth, "send_raw_transaction", original_send)
    monkeypatch.setattr(eth, "get_transaction_receipt", unavailable)
    provider.nonce = provider.pending_nonce = NONCE + 1
    with pytest.raises(CollateralTransactionError):
        await client.finalize_reclaim(5)
    assert {decode_legacy(raw)["nonce"] for raw in provider.sent} == {NONCE}


async def test_a_used_nonce_with_no_receipt_keeps_the_record_until_a_receipt_settles_it(monkeypatch):
    """A lagging RPC, or one that prunes old receipts, shows the nonce used with no receipt. The nonce does not say
    which transaction took it or how that ended, so nothing new is signed until an RPC serves the receipt."""
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    provider.mine_sent = False
    client = client_with(provider)
    wait_for_receipt = client.w3.eth.wait_for_transaction_receipt
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", lost_receipt)
    with pytest.raises(CollateralOutcomeUnknownError):
        await client.finalize_reclaim(5)
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", wait_for_receipt)
    signed_hash = client._read_sent_record(CHAIN_ID)["hash"]

    provider.nonce = provider.pending_nonce = NONCE + 1
    for _ in range(2):
        with pytest.raises(CollateralOutcomeUnknownError, match=f"nonce {NONCE} is used, so its outcome is unknown"):
            await client.finalize_reclaim(5)
    assert len(provider.sent) == 1
    assert client._read_sent_record(CHAIN_ID)["hash"] == signed_hash

    # an RPC that serves the receipt settles it, and only then is the next nonce signed
    provider.mined.add(signed_hash)
    with pytest.raises(CollateralTransactionError, match="sent earlier, succeeded in block 16"):
        await client.finalize_reclaim(5)
    assert client._read_sent_record(CHAIN_ID) is None
    provider.mine_sent = True
    await client.finalize_reclaim(5)
    assert [decode_legacy(raw)["nonce"] for raw in provider.sent] == [NONCE, NONCE + 1]


async def test_a_used_nonce_with_no_receipt_names_the_two_ways_to_settle_it(monkeypatch):
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.mine_sent = False
    client = client_with(provider)
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", lost_receipt)
    with pytest.raises(CollateralOutcomeUnknownError):
        await client.finalize_reclaim(5)

    provider.nonce = provider.pending_nonce = NONCE + 1
    with pytest.raises(CollateralOutcomeUnknownError) as raised:
        await client.finalize_reclaim(5)
    assert "SUBTENSOR_EVM_RPC_URL set to an RPC that serves its receipt" in str(raised.value)
    assert f"delete {client.sent_record_path}" in str(raised.value)
    assert "no transaction was sent" in str(raised.value)


async def test_a_refusal_that_echoes_the_rpc_url_is_reported_in_local_words_only():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    provider.send_error = "keyed refusal"
    client = client_with(provider)

    with pytest.raises(CollateralOutcomeUnknownError) as raised:
        await client.finalize_reclaim(5)
    assert "(insufficient funds for gas)" in str(raised.value)
    assert_no_rpc_secret(str(raised.value))
    assert client._read_sent_record(CHAIN_ID) is not None


async def test_an_unmined_send_whose_rebroadcast_is_not_accepted_stops_at_once_with_the_way_out(monkeypatch):
    """No 300 s receipt wait on every rerun: an answer that is not "already known" ends the run."""
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.mine_sent = False
    client = client_with(provider)
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", lost_receipt)
    with pytest.raises(CollateralOutcomeUnknownError):
        await client.finalize_reclaim(5)

    provider.send_error = "unknown account"
    with pytest.raises(CollateralOutcomeUnknownError, match="broadcasting it again failed") as raised:
        await client.finalize_reclaim(5)
    assert "delete" not in str(raised.value)
    assert f"replace-collateral-transaction`: it signs a replacement at the same nonce {NONCE}" in str(raised.value)
    assert len(provider.sent) == 1


async def test_a_receipt_that_names_another_transaction_is_an_unknown_outcome(monkeypatch):
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.mine_sent = False
    client = client_with(provider)
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", lost_receipt)
    with pytest.raises(CollateralOutcomeUnknownError):
        await client.finalize_reclaim(5)
    signed_hash = client._read_sent_record(CHAIN_ID)["hash"]
    provider.receipts_of_another.add(signed_hash)
    with pytest.raises(CollateralOutcomeUnknownError, match="another transaction's receipt"):
        await client.finalize_reclaim(5)
    assert client._read_sent_record(CHAIN_ID)["hash"] == signed_hash
    assert len(provider.sent) == 1


@pytest.mark.parametrize(
    "signed_nonce,latest_after,pending_after",
    [(NONCE, NONCE, NONCE + 1), (NONCE + 1, NONCE, NONCE)],
    ids=["still-pending-at-a-low-price", "lagging-rpc"],
)
async def test_an_unmined_send_is_replaced_at_its_own_nonce_and_its_record_is_never_dropped(
    monkeypatch, signed_nonce, latest_after, pending_after
):
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    provider.nonce = provider.pending_nonce = signed_nonce
    provider.mine_sent = False
    client = client_with(provider)
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", lost_receipt)
    with pytest.raises(CollateralOutcomeUnknownError):
        await client.finalize_reclaim(5)

    provider.nonce, provider.pending_nonce = latest_after, pending_after
    provider.gas_price = GAS_PRICE + 1
    with pytest.raises(CollateralOutcomeUnknownError, match="broadcast again and no new transaction was sent") as raised:
        await client.finalize_reclaim(5)
    assert "delete" not in str(raised.value)
    assert "replace-collateral-transaction" in str(raised.value)
    assert {decode_legacy(raw)["nonce"] for raw in provider.sent} == {signed_nonce}

    first_hash = sent_hash(provider)
    provider.mine_sent = True
    outcome = await client.replace_earlier_send()
    assert f"Transaction {sent_hash(provider, -1)}, the replacement, succeeded in block 16" in outcome
    replacement = decode_legacy(provider.sent[-1])
    assert [decode_legacy(raw)["nonce"] for raw in provider.sent] == [signed_nonce] * 3
    assert replacement["gasPrice"] == -(-GAS_PRICE * 9 // 8)
    assert replacement["data"] == decode_legacy(provider.sent[0])["data"]
    assert client._read_sent_record(CHAIN_ID) is None
    assert first_hash != sent_hash(provider, -1)


async def test_a_replacement_is_recorded_with_the_send_it_replaces_before_it_is_broadcast(monkeypatch):
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.mine_sent = False
    client = client_with(provider)
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", lost_receipt)
    monkeypatch.setattr(collateral_module, "RECEIPT_TIMEOUT_SEC", 0)
    with pytest.raises(CollateralOutcomeUnknownError):
        await client.finalize_reclaim(5)
    first_hash = sent_hash(provider)
    recorded_at_broadcast = []
    original_send = client.w3.eth.send_raw_transaction

    async def send(raw):
        recorded_at_broadcast.append(client._read_sent_record(CHAIN_ID))
        return await original_send(raw)

    monkeypatch.setattr(client.w3.eth, "send_raw_transaction", send)
    with pytest.raises(CollateralOutcomeUnknownError, match="not mined yet"):
        await client.replace_earlier_send()
    replacement_hash = sent_hash(provider, -1)
    assert recorded_at_broadcast[0]["hashes"] == [first_hash, replacement_hash]

    # the first transaction is the one mined: its receipt settles the record
    provider.mined.add(first_hash)
    with pytest.raises(CollateralTransactionError, match=f"Transaction {first_hash}, sent earlier, succeeded"):
        await client.finalize_reclaim(5)
    assert client._read_sent_record(CHAIN_ID) is None


ATTACKER_KEY = "0x" + "22" * 32
ATTACKER = Account.from_key(ATTACKER_KEY).address


def signed_record(key=MINER_KEY, to=CONTRACT, value=0, data=None, nonce=NONCE, chain_id=CHAIN_ID, **over) -> dict:
    data = data or "0x" + selector("finalizeReclaim(uint256)") + f"{5:064x}"
    signed = Account.sign_transaction(
        {"nonce": nonce, "gasPrice": GAS_PRICE, "gas": 200_000, "to": to, "value": value, "data": data,
         "chainId": chain_id},
        key,
    )
    raw = AsyncWeb3.to_hex(getattr(signed, "raw_transaction", None) or signed.rawTransaction)
    tx_hash = signed.hash.hex()
    return {"nonce": nonce, "hash": tx_hash, "raw": raw, "hashes": [tx_hash], **over}


@pytest.mark.parametrize(
    "record",
    [
        signed_record(key=ATTACKER_KEY, to=ATTACKER, value=10**18),
        signed_record(to=ATTACKER, value=10**18),
        signed_record(value=10**18),
        signed_record(data="0x" + selector("transferOwnership(address)") + ATTACKER[2:].lower().rjust(64, "0")),
        signed_record(chain_id=1),
        {**signed_record(nonce=NONCE + 3), "nonce": NONCE},
        signed_record(hash="0x" + "ee" * 32),
        {**signed_record(), "raw": "0xdeadbeef"},
    ],
    ids=[
        "another-key-sends-value-elsewhere", "value-to-another-address", "value-to-the-contract",
        "another-function", "another-chain", "another-nonce", "another-hash", "not-a-transaction",
    ],
)
async def test_a_forged_send_record_gets_nothing_signed_or_broadcast(record):
    """The record sits in the wallet directory the miner service writes; the key is typed in only later, so a
    record's bytes must prove this key signed a collateral call before a replacement copies anything from them."""
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.mine_sent = False
    client = client_with(provider)
    client._write_sent_record(CHAIN_ID, record)
    signed = []
    original_sign = client.miner_account.sign_transaction
    client.miner_account.sign_transaction = lambda transaction: signed.append(transaction) or original_sign(transaction)

    with pytest.raises(CollateralTransactionError, match="nothing was signed or broadcast"):
        await client.replace_earlier_send()
    assert signed == [] and provider.sent == []
    assert "eth_sendRawTransaction" not in [method for method, _ in provider.requests]
    assert client._read_sent_record(CHAIN_ID) == record


async def test_a_record_this_key_signed_for_the_contract_is_replaced_from_its_verified_bytes():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    client = client_with(provider)
    record = signed_record()
    # loose fields next to the bytes are never what gets signed
    client._write_sent_record(CHAIN_ID, {**record, "to": ATTACKER, "value": 10**18, "data": "0x"})

    outcome = await client.replace_earlier_send()
    assert "the replacement, succeeded" in outcome
    replacement = decode_legacy(provider.sent[-1])
    assert replacement["to"] == CONTRACT and replacement["value"] == 0 and replacement["nonce"] == NONCE
    assert replacement["data"] == decode_legacy(record["raw"])["data"]


@pytest.mark.parametrize("case", ["orphaned", "not-finalized"])
async def test_a_receipt_that_is_not_final_on_chain_never_clears_the_send_record(monkeypatch, case):
    """A receipt from a block a reorganization dropped (or may still drop) says nothing about the outcome: the
    record stays and no other nonce is signed (review of 58f7221: receipt block 0xaaaa, canonical 0xbbbb)."""
    monkeypatch.setattr(collateral_module, "RECEIPT_TIMEOUT_SEC", 0)
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.mine_sent = False
    client = client_with(provider)
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", lost_receipt)
    with pytest.raises(CollateralOutcomeUnknownError):
        await client.finalize_reclaim(5)
    record = client._read_sent_record(CHAIN_ID)

    provider.mined.add(record["hash"])
    provider.nonce = provider.pending_nonce = NONCE + 1
    if case == "orphaned":
        provider.canonical_hashes[16] = "0x" + "bb" * 32
    else:
        provider.finalized_number = 15
    with pytest.raises(CollateralOutcomeUnknownError) as raised:
        await client.finalize_reclaim(5)
    assert "succeeded" not in str(raised.value)
    assert client._read_sent_record(CHAIN_ID) == record
    assert len(provider.sent) == 1


async def test_a_fresh_send_whose_receipt_block_is_orphaned_keeps_its_record():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    provider.canonical_hashes[16] = "0x" + "bb" * 32
    client = client_with(provider)

    with pytest.raises(CollateralOutcomeUnknownError, match="not the finalized block at that number"):
        await client.finalize_reclaim(5)
    assert client._read_sent_record(CHAIN_ID)["hash"] == sent_hash(provider)


async def test_a_receipt_from_a_lagging_backend_on_another_fork_is_not_final():
    """Backend A finalized block 20 on fork A; a lagging backend B serves its own block 16, the one the receipt
    names, to a read by number (review of 21ec2fd: "orphaned receipt treated finalized: True")."""
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    provider.block_number = provider.finalized_number = 20
    provider.canonical_hashes[16] = "0x" + "aa" * 32
    provider.fork_hashes[16] = BLOCK_HASH
    client = client_with(provider)

    with pytest.raises(CollateralOutcomeUnknownError, match="not the finalized block at that number"):
        await client.finalize_reclaim(5)
    assert client._read_sent_record(CHAIN_ID)["hash"] == sent_hash(provider)
    assert ("eth_getBlockByHash", ["0x" + "aa" * 32, False]) in provider.requests


async def test_a_receipt_on_the_finalized_chain_clears_the_record_even_when_numeric_reads_reach_another_fork():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    provider.block_number = provider.finalized_number = 20
    provider.fork_hashes = {n: "0x" + f"{n:060x}0b0b" for n in range(16, 20)}
    client = client_with(provider)

    await client.finalize_reclaim(5)
    assert client._read_sent_record(CHAIN_ID) is None


async def test_a_receipt_too_far_behind_the_finalized_block_keeps_the_record(monkeypatch):
    monkeypatch.setattr(collateral_module, "FINALITY_CHECK_MAX_BLOCKS", 3)
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    provider.block_number = provider.finalized_number = 20
    client = client_with(provider)

    with pytest.raises(CollateralOutcomeUnknownError, match="too far back"):
        await client.finalize_reclaim(5)
    assert client._read_sent_record(CHAIN_ID)["hash"] == sent_hash(provider)


OLD_CONTRACT = "0x999F9A49A85e9D6E981cad42f197349f50172bEB"


async def test_an_unmined_send_to_the_1_0_0_contract_is_replaced_to_that_contract():
    """The replacement command builds the default 1.0.2 client; a 1.0.0 reclaim is still this key's send."""
    from core import utils

    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    client = utils.get_collateral_contract(miner_key=MINER_KEY)
    client._w3 = AsyncWeb3(provider)
    record = signed_record(to=OLD_CONTRACT)
    client._write_sent_record(CHAIN_ID, record)

    outcome = await client.replace_earlier_send()
    assert "the replacement, succeeded" in outcome
    replacement = decode_legacy(provider.sent[-1])
    assert replacement["to"] == OLD_CONTRACT and replacement["nonce"] == NONCE
    assert replacement["data"] == decode_legacy(record["raw"])["data"]


async def test_a_send_to_an_unconfigured_contract_is_not_replaced():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    client = client_with(provider)
    record = signed_record(to=OLD_CONTRACT)
    client._write_sent_record(CHAIN_ID, record)

    with pytest.raises(CollateralTransactionError, match="not sent to a configured collateral contract"):
        await client.replace_earlier_send()
    assert provider.sent == []


async def test_nothing_is_replaced_without_a_recorded_send():
    provider = FakeProvider()
    with pytest.raises(CollateralTransactionError, match="nothing was replaced"):
        await client_with(provider).replace_earlier_send()
    assert provider.sent == []


async def test_a_mined_reclaim_whose_answer_was_lost_reports_its_request_id(monkeypatch):
    provider = FakeProvider(logs=[started_log(reclaim_request_id=12)])
    client = client_with(provider)
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", lost_receipt)
    with pytest.raises(CollateralOutcomeUnknownError):
        await client.reclaim_collateral(EXECUTOR)

    provider.nonce = provider.pending_nonce = NONCE + 1
    with pytest.raises(CollateralTransactionError, match="succeeded in block 16; it started reclaim request 12;"):
        await client.reclaim_collateral(EXECUTOR)
    assert len(provider.sent) == 1


class Stopped(BaseException):
    """The container stopping: nothing after it runs, no handler catches it."""


def stop_on_outcome_line(monkeypatch) -> SimpleNamespace:
    """Stops the run when the outcome line is about to be logged, while `switch.on`."""
    switch = SimpleNamespace(on=True)
    real_info = collateral_module.logger.info

    def info(message, *args, **kwargs):
        if switch.on and "succeeded in block" in str(message):
            raise Stopped()
        real_info(message, *args, **kwargs)

    monkeypatch.setattr(collateral_module.logger, "info", info)
    return switch


async def test_a_reclaim_stopped_after_finality_keeps_its_record_and_the_retry_reports_its_request_id(monkeypatch):
    """Review of 7cd21ec: the record was cleared before the request ID was decoded and logged, so a stop in between
    left no record and the listing only reaches back RECLAIM_LOOKBACK_BLOCKS blocks."""
    provider = FakeProvider(logs=[started_log(reclaim_request_id=12)])
    client = client_with(provider)
    stop = stop_on_outcome_line(monkeypatch)
    with pytest.raises(Stopped):
        await client.reclaim_collateral(EXECUTOR)
    assert client._read_sent_record(CHAIN_ID)["hash"] == sent_hash(provider)

    stop.on = False
    provider.nonce = provider.pending_nonce = NONCE + 1
    with pytest.raises(CollateralTransactionError, match="succeeded in block 16; it started reclaim request 12;"):
        await client.reclaim_collateral(EXECUTOR)
    assert client._read_sent_record(CHAIN_ID) is None
    assert len(provider.sent) == 1


async def test_a_reclaim_logs_its_request_id_before_its_record_is_cleared(monkeypatch, caplog):
    provider = FakeProvider(logs=[started_log(reclaim_request_id=12)])
    client = client_with(provider)
    logged_at_clear = []
    real_clear = client._clear_sent_record

    def clear(chain_id, tx_hash):
        logged_at_clear.append(caplog.text)
        real_clear(chain_id, tx_hash)

    monkeypatch.setattr(client, "_clear_sent_record", clear)
    with caplog.at_level(logging.INFO, logger=collateral_module.logger.name):
        event = await client.reclaim_collateral(EXECUTOR)

    assert event["args"]["reclaimRequestId"] == 12
    assert len(logged_at_clear) == 1
    assert f"Transaction {sent_hash(provider)} succeeded in block 16; it started reclaim request 12" in logged_at_clear[0]
    assert client._read_sent_record(CHAIN_ID) is None


async def test_a_settled_earlier_reclaim_logs_its_request_id_before_its_record_is_cleared(monkeypatch):
    provider = FakeProvider(logs=[started_log(reclaim_request_id=12)])
    client = client_with(provider)
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", lost_receipt)
    with pytest.raises(CollateralOutcomeUnknownError):
        await client.reclaim_collateral(EXECUTOR)

    provider.nonce = provider.pending_nonce = NONCE + 1
    stop_on_outcome_line(monkeypatch)
    with pytest.raises(Stopped):
        await client.settle_earlier_send()
    assert client._read_sent_record(CHAIN_ID)["hash"] == sent_hash(provider)


async def test_a_retried_finalize_settles_the_earlier_send_before_it_reads_the_request(monkeypatch):
    """A mined finalize closes the request, so the open-request check alone would hide its outcome."""
    reclaims = selector("reclaims(uint256)")
    provider = FakeProvider(calls={reclaims: open_reclaim()}, logs=[reclaimed_log()])
    client = client_with(provider)
    monkeypatch.setattr(client.w3.eth, "wait_for_transaction_receipt", lost_receipt)
    with pytest.raises(CollateralOutcomeUnknownError):
        await client.finalize_reclaim(5)

    provider.calls[reclaims] = open_reclaim(amount=0)
    provider.nonce = provider.pending_nonce = NONCE + 1
    with pytest.raises(CollateralTransactionError, match="sent earlier, succeeded in block 16"):
        await client.finalize_reclaim(5)
    assert collateral_module.SENT_RECORD_PATH.read_text() == "{}"
    assert len(provider.sent) == 1


async def test_a_second_run_while_one_is_sending_sends_nothing():
    import fcntl

    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    path = collateral_module.SENT_RECORD_PATH
    with open(path.with_name(path.name + ".lock"), "a") as other_run:
        fcntl.flock(other_run, fcntl.LOCK_EX)
        with pytest.raises(CollateralTransactionError, match="Another run is sending"):
            await client_with(provider).finalize_reclaim(5)
    assert provider.sent == []
    await client_with(provider).finalize_reclaim(5)
    assert len(provider.sent) == 1


def test_a_stale_clear_leaves_a_newer_send_record():
    client = client_with(FakeProvider())
    old = {"nonce": NONCE, "hash": "0x" + "01" * 32, "raw": "0x"}
    new = {"nonce": NONCE + 1, "hash": "0x" + "02" * 32, "raw": "0x"}
    client._write_sent_record(CHAIN_ID, old)
    client._clear_sent_record(CHAIN_ID, old["hash"])
    client._write_sent_record(CHAIN_ID, new)
    client._clear_sent_record(CHAIN_ID, old["hash"])
    assert client._read_sent_record(CHAIN_ID) == new


async def test_the_record_directory_is_synced_before_the_broadcast(monkeypatch):
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    client = client_with(provider)
    events = []
    real_fsync, real_replace = collateral_module.os.fsync, collateral_module.pathlib.Path.replace

    def fsync(fd):
        kind = "directory" if collateral_module.os.path.isdir(f"/proc/self/fd/{fd}") else "file"
        events.append(f"fsync {kind}")
        real_fsync(fd)

    def replace(self, target):
        events.append("rename")
        return real_replace(self, target)

    original_send = client.w3.eth.send_raw_transaction

    async def send(raw):
        events.append("broadcast")
        return await original_send(raw)

    monkeypatch.setattr(collateral_module.os, "fsync", fsync)
    monkeypatch.setattr(collateral_module.pathlib.Path, "replace", replace)
    monkeypatch.setattr(client.w3.eth, "send_raw_transaction", send)
    await client.finalize_reclaim(5)
    assert events[: events.index("broadcast") + 1] == ["fsync file", "rename", "fsync directory", "broadcast"]


async def test_the_open_reclaim_list_reads_every_request_at_the_finalized_block_hash():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[started_log()])
    provider.block_number = 5000
    client = client_with(provider)

    requests = await client.get_reclaim_events()
    assert [request.reclaim_request_id for request in requests] == [5]
    details = [params for method, params in provider.requests if method == "eth_call"]
    assert details and all(block == {"blockHash": provider.chain_hash(5000)} for _, block in details)


async def test_a_reclaim_request_read_at_a_block_hash_names_that_block():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.block_number = provider.finalized_number = 5000
    client = client_with(provider)

    block_hash = await client.finalized_block_hash()
    reclaim = await client.get_reclaim_request(5, block_hash=block_hash)

    assert reclaim[2] == 10**17
    details = [params for method, params in provider.requests if method == "eth_call"]
    assert [block for _, block in details] == [{"blockHash": provider.chain_hash(5000)}]


FORK_B = "0x" + "0b" * 32


def fork_b_log(url="https://fork-b/reclaim"):
    return {**started_log(url=url), "blockHash": FORK_B}


@pytest.mark.parametrize(
    "logs_by_fork,fork_reads,urls",
    [
        # the first log read reaches backend B, the second backend A: only A's list is kept
        ([[fork_b_log()], [started_log(url="https://fork-a/reclaim")]], False, ["https://fork-a/reclaim"]),
        # every read by number between the finalized block and the log reaches backend B
        ([[started_log(url="https://fork-a/reclaim")]], True, ["https://fork-a/reclaim"]),
        # both: B's log in B's block is still off the finalized chain
        ([[fork_b_log()], [started_log(url="https://fork-a/reclaim")]], True, ["https://fork-a/reclaim"]),
        ([[fork_b_log()]] * collateral_module.RECLAIM_LIST_ATTEMPTS, False, None),
    ],
    ids=["log-from-b", "numeric-reads-from-b", "log-and-numeric-reads-from-b", "b-every-time"],
)
async def test_the_open_reclaim_list_never_mixes_logs_and_state_of_two_forks(logs_by_fork, fork_reads, urls):
    """A load-balanced RPC answers the finalized block from backend A and a read by number from a lagging backend
    B on another fork (review of 21ec2fd: A's log with B's amount). The list is A's, or an error."""
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.block_number = provider.finalized_number = 5000
    provider.logs_by_fork = list(logs_by_fork)
    if fork_reads:
        provider.fork_hashes = {n: "0x" + f"{n:060x}0b0b" for n in range(16, 5000)}
        provider.fork_hashes[16] = FORK_B

    if urls is None:
        with pytest.raises(CollateralTransactionError, match="not on the finalized chain"):
            await client_with(provider).get_reclaim_events()
        assert [method for method, _ in provider.requests].count("eth_call") == 0
        return
    requests = await client_with(provider).get_reclaim_events()

    assert [request.url for request in requests] == urls
    details = [params for method, params in provider.requests if method == "eth_call"]
    assert details and all(block == {"blockHash": provider.chain_hash(5000)} for _, block in details)


async def test_an_empty_log_answer_from_a_lagging_backend_still_lists_the_open_request():
    """Review of 21ec2fd: backend A serves the finalized block, lagging backend B the numeric log range and
    answers []. Block 4500's header, linked to the finalized block, has a bloom that may hold the event, so that
    block's logs are read again by its hash."""
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.block_number = provider.finalized_number = 5000
    provider.logs = [{**started_log(), "blockNumber": hex(4500), "blockHash": provider.chain_hash(4500)}]
    provider.logs_by_fork = [[]]

    requests = await client_with(provider).get_reclaim_events()

    assert [(request.reclaim_request_id, request.block_number) for request in requests] == [(5, 4500)]
    by_hash = [params[0] for method, params in provider.requests if method == "eth_getLogs" and "blockHash" in params[0]]
    assert [log_filter["blockHash"] for log_filter in by_hash] == [provider.chain_hash(4500)]


async def test_a_block_whose_logs_a_backend_cannot_serve_by_hash_is_an_error_not_an_empty_list():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.block_number = provider.finalized_number = 5000
    provider.logs = [{**started_log(), "blockNumber": hex(4500), "blockHash": provider.chain_hash(4500)}]
    provider.logs_by_fork = [[]] * collateral_module.RECLAIM_LIST_ATTEMPTS
    provider.unknown_block_hashes = {provider.chain_hash(4500)}

    with pytest.raises(CollateralTransactionError, match="could not answer a block's logs"):
        await client_with(provider).get_reclaim_events()
    assert [method for method, _ in provider.requests].count("eth_call") == 0


async def test_a_request_in_both_the_range_and_the_by_hash_answer_is_listed_once():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[started_log()])
    provider.block_number = 5000

    requests = await client_with(provider).get_reclaim_events()

    assert [request.reclaim_request_id for request in requests] == [5]


def test_the_bloom_check_matches_a_bloom_built_from_the_log():
    log = started_log()
    bloom = bytes.fromhex(bloom_of([log]).removeprefix("0x"))
    address, topic = bytes.fromhex(CONTRACT.removeprefix("0x")), bytes.fromhex(log["topics"][0].removeprefix("0x"))
    other = bytes.fromhex("12" * 20)

    assert collateral_module.bloom_may_hold(bloom, address, topic)
    assert not collateral_module.bloom_may_hold(bytes(256), address, topic)
    assert not collateral_module.bloom_may_hold(bloom, other, topic)
    # a bloom that proves nothing never hides a block
    assert collateral_module.bloom_may_hold(None, address, topic)
    assert collateral_module.bloom_may_hold(b"\x00" * 10, address, topic)


async def test_the_open_reclaim_list_is_read_at_the_finalized_block_not_the_head():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[started_log()])
    provider.block_number, provider.finalized_number = 5003, 5000

    requests = await client_with(provider).get_reclaim_events()

    assert [request.reclaim_request_id for request in requests] == [5]
    assert [params[0] for method, params in provider.requests if method == "eth_getBlockByNumber"][0] == "finalized"
    (log_filter,) = [
        params[0] for method, params in provider.requests if method == "eth_getLogs" and "blockHash" not in params[0]
    ]
    assert log_filter["toBlock"] == hex(5000)
    details = [params for method, params in provider.requests if method == "eth_call"]
    assert details and all(block == {"blockHash": provider.chain_hash(5000)} for _, block in details)


async def test_a_refused_broadcast_keeps_the_record_and_the_next_run_sends_the_same_bytes():
    """A refusal may be another upstream's answer after one took the bytes, so it proves nothing was sent only
    once the same bytes are broadcast again; no second nonce is signed meanwhile."""
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    provider.send_error = "refused"
    client = client_with(provider)

    with pytest.raises(CollateralOutcomeUnknownError, match=r"answered with an error \(insufficient funds for gas\)"):
        await client.finalize_reclaim(5)
    signed_hash = client._read_sent_record(CHAIN_ID)["hash"]
    provider.send_error = None
    with pytest.raises(CollateralTransactionError, match="sent earlier, succeeded in block 16"):
        await client.finalize_reclaim(5)
    assert [AsyncWeb3.keccak(hexstr=raw).hex() for raw in provider.sent] == [signed_hash]


@pytest.mark.parametrize("network,rpc_chain_id", [("finney", 1), ("test", 964), ("archive", 945)])
async def test_an_rpc_on_another_chain_gets_nothing_signed(network, rpc_chain_id):
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.chain_id = rpc_chain_id
    client = CollateralClient(network=network, contract_address=CONTRACT, miner_key=MINER_KEY)
    client._w3 = AsyncWeb3(provider)

    with pytest.raises(CollateralConfigError, match=f"reports EVM chain {rpc_chain_id}"):
        await client.finalize_reclaim(5)
    assert provider.sent == []
    assert "eth_getTransactionCount" not in [method for method, _ in provider.requests]


@pytest.mark.parametrize("over,sent", [(1, 0), (0, 1)], ids=["above", "at"])
async def test_a_gas_price_quote_above_the_ceiling_signs_nothing_and_one_at_it_is_signed(over, sent):
    ceiling_wei = collateral_module.DEFAULT_MAX_GAS_PRICE_GWEI * 10**9
    provider = FakeProvider(
        calls={selector("reclaims(uint256)"): open_reclaim()},
        logs=[reclaimed_log()],
        gas_price=ceiling_wei + over,
    )
    if over:
        with pytest.raises(CollateralTransactionError, match="COLLATERAL_MAX_GAS_PRICE_GWEI"):
            await client_with(provider).finalize_reclaim(5)
        assert "eth_getTransactionCount" not in [method for method, _ in provider.requests]
    else:
        await client_with(provider).finalize_reclaim(5)
        assert decode_legacy(provider.sent[0])["gasPrice"] == ceiling_wei
    assert len(provider.sent) == sent


async def test_configured_gas_price_ceiling_reaches_the_client(monkeypatch):
    from core import utils
    from core.config import settings

    monkeypatch.setattr(settings, "COLLATERAL_MAX_GAS_PRICE_GWEI", 5)
    client = utils.get_collateral_contract(miner_key=MINER_KEY)
    client._w3 = AsyncWeb3(FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}))
    with pytest.raises(CollateralTransactionError, match="5 gwei ceiling"):
        await client.finalize_reclaim(5)
    assert client._w3.provider.sent == []


@pytest.mark.parametrize("rpc_url", [None, RPC_URL], ids=["no-rpc-url", "rpc-url"])
async def test_contract_call_on_an_unknown_network_needs_and_uses_the_rpc_url(rpc_url, monkeypatch):
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
    client = CollateralClient(network="archive", contract_address=CONTRACT, rpc_url=rpc_url)
    if rpc_url is None:
        with pytest.raises(CollateralConfigError, match="SUBTENSOR_EVM_RPC_URL") as raised:
            await client.get_executor_collateral(EXECUTOR)
        assert "archive" in str(raised.value)
        assert providers == []
    else:
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


@pytest.mark.parametrize(
    "args,exit_code",
    [
        (["current-contract-version"], 0),
        (["reclaim-collateral", "--executor_uuid", EXECUTOR, "--private-key", MINER_KEY], 1),
    ],
    ids=["non-collateral-command", "collateral-command"],
)
def test_a_command_on_an_unknown_network_runs_or_names_the_setting(archive_network, caplog, args, exit_code):
    from cli import cli

    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(cli, args)
    assert result.exit_code == exit_code, result.output
    if exit_code == 0:
        assert CONTRACT in result.output
    else:
        assert isinstance(result.exception, SystemExit)
        assert "SUBTENSOR_EVM_RPC_URL" in caplog.text
    assert MINER_KEY.removeprefix("0x")[:16] not in caplog.text + result.output


@pytest.mark.parametrize(
    "outcome,exit_code",
    [
        ("Transaction 0xab, the replacement, succeeded in block 16; Run the reclaim or finalize again", 0),
        (CollateralOutcomeUnknownError("Transaction 0xab, sent earlier, is not mined yet"), 1),
        (CollateralTransactionError("Transaction 0xab, the replacement, reverted in block 16"), 1),
    ],
    ids=["replacement-succeeded", "outcome-unknown", "replacement-reverted"],
)
def test_replace_collateral_transaction_exits_0_only_on_a_success(archive_network, monkeypatch, caplog, outcome, exit_code):
    import cli as cli_module

    async def replace_earlier_send():
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(
        cli_module, "get_collateral_contract", lambda **_: SimpleNamespace(replace_earlier_send=replace_earlier_send)
    )
    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(cli_module.cli, ["replace-collateral-transaction", "--private-key", MINER_KEY])
    assert result.exit_code == exit_code, result.output
    assert ("✅" in caplog.text) is (exit_code == 0)
    assert MINER_KEY.removeprefix("0x")[:16] not in caplog.text + result.output


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
