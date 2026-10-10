"""CollateralClient against a fake JSON-RPC provider: what it signs and sends, and when it needs an RPC URL."""

import logging
import traceback
from json import JSONDecodeError
from types import SimpleNamespace
from uuid import UUID

import aiohttp
import pytest
import rlp
from click.testing import CliRunner
from eth_abi import encode
from eth_account import Account
from web3 import AsyncWeb3
from web3.providers.async_base import AsyncBaseProvider

from core import collateral as collateral_module
from core.collateral import (
    STORAGE_LAYOUTS,
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
SUBSTRATE_READS = ("chain_getBlockHash", "state_getStorage")


def listed_tx(block_hash: str, tx_hash: str) -> str:
    return "0x" + AsyncWeb3.keccak(hexstr=block_hash + tx_hash.removeprefix("0x")).hex().removeprefix("0x")


def bloom_of(logs) -> str:
    """The 2048-bit logs bloom of a block holding `logs`: three bits from the keccak of each address and topic."""
    bits = 0
    for log in logs:
        for value in (log["address"], *log["topics"]):
            digest = AsyncWeb3.keccak(hexstr=value)
            for i in (0, 2, 4):
                bits |= 1 << (int.from_bytes(digest[i : i + 2], "big") % 2048)
    return "0x" + f"{bits:0512x}"


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
    # the default finney RPC answers a larger batch with one -32010 error object
    BATCH_LIMIT = 50

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
        # the "finalized" tag's number when it trails the head; the chain's block hash at a number, when it is
        # not the one receipts name
        self.finalized_number = None
        self.canonical_hashes = {}
        # a lagging backend on another fork: the hash it answers a read by number with
        self.fork_hashes = {}
        # hashes whose receipt request is answered with TX_HASH's receipt
        self.receipts_of_another = set()
        # blocks below this number are pruned: a read of one answers null
        self.oldest_kept = 0
        # who answers each next batch: "b" a backend on fork B, "lagging" one without these blocks, "429" none,
        # "logs-lagging" a gateway that sends the batch's blocks to A and its receipts to a lagging backend,
        # "logs-empty" one that sends the receipts to a backend that knows each block but answers each with null
        self.batch_backends = []
        self.fork_b_hash = FORK_B
        # transactions of a listed block whose receipt is answered with null, as by a partly synced backend
        self.withheld_receipts = set()
        # a backend whose Substrate block at each number is a sibling of this chain's: its storage answers these
        self.sibling_calls = {}

    async def is_connected(self, show_traceback: bool = False) -> bool:
        return True

    async def make_batch_request(self, requests):
        if len(requests) > self.BATCH_LIMIT:
            error = {"code": -32010, "message": "The batch request was too large", "data": "Exceeded max limit of 50"}
            return {"jsonrpc": "2.0", "id": None, "error": error}
        backend = self.batch_backends.pop(0) if self.batch_backends else "a"
        if backend == "429":
            raise aiohttp.ClientResponseError(None, (), status=429, message="Too Many Requests")
        answers = []
        for i, (method, params) in enumerate(requests):
            logs_elsewhere = backend in ("logs-lagging", "logs-empty")
            if backend == "a" or params[0] == "finalized" or (logs_elsewhere and method != "eth_getTransactionReceipt"):
                answer = await self.make_request(method, params)
            elif backend in ("split", "sibling", "mixed") and method in SUBSTRATE_READS:
                # Substrate reads a gateway sends to a backend without these blocks ("split"), to one whose block at
                # each number is a sibling ("sibling"), or, item by item, to A and then the sibling one ("mixed")
                view = "lagging" if backend == "split" else ("a" if backend == "mixed" and i == 0 else "sibling")
                self.requests.append((method, params))
                answer = self.substrate_answer(method, params, view)
            elif backend in ("split", "sibling", "mixed"):
                answer = await self.make_request(method, params)
            elif backend == "logs-empty":
                self.requests.append((method, params))
                answer = {"result": None}
            elif backend == "logs-lagging":
                self.requests.append((method, params))
                answer = {"error": {"code": -32000, "message": "unknown block"}}
            else:
                self.requests.append((method, params))
                fork_b = {"eth_getLogs": [fork_b_log()], "eth_getBlockByNumber": {**self.block(16), "hash": self.fork_b_hash}}
                answer = {"result": fork_b.get(method) if backend == "b" else ([] if method == "eth_getLogs" else None)}
            answers.append({**answer, "id": i})
        return answers[::-1]

    def chain_hash(self, number: int) -> str:
        if number in self.canonical_hashes:
            return self.canonical_hashes[number]
        return BLOCK_HASH if number == 16 else "0x" + f"{number:064x}"

    def block(self, number: int) -> dict:
        block_hash = self.chain_hash(number)
        logs = [log for log in self.logs if int(log["blockNumber"], 16) == number and log["blockHash"] == block_hash]
        return {
            "number": hex(number),
            "hash": block_hash,
            "parentHash": self.chain_hash(number - 1),
            "timestamp": hex(1_700_000_000 + 12 * number),
            "logsBloom": bloom_of(logs),
            "transactions": list(dict.fromkeys(listed_tx(block_hash, log["transactionHash"]) for log in logs)),
        }

    def listed_receipt(self, tx_hash: str) -> dict | None:
        """The receipt of a transaction a listed block names: its logs are the fake logs of that block and
        transaction. A log's own transactionHash is TX_HASH, the send's, so a listed block names another hash."""
        logs = [log for log in self.logs if listed_tx(log["blockHash"], log["transactionHash"]) == tx_hash]
        if not logs:
            return None
        return {
            "transactionHash": tx_hash,
            "blockHash": logs[0]["blockHash"],
            "blockNumber": logs[0]["blockNumber"],
            "status": "0x1",
            "logs": logs,
        }

    def fork_block(self, number: int) -> dict:
        parent = self.fork_hashes.get(number - 1, self.chain_hash(number - 1))
        return {"number": hex(number), "hash": self.fork_hashes[number], "parentHash": parent}

    @staticmethod
    def substrate_hash(number: int, view: str = "a") -> str:
        return "0x" + ("5c" if view == "sibling" else "5b") + f"{number:062x}"

    @staticmethod
    def storage_of(calls: dict) -> dict[str, str]:
        """The AccountStorages entries of each contract that answer `calls` (by selector) through its getters, for
        EXECUTOR and reclaim ids 0-15. A zero word is an unset slot."""
        words = {}
        for contract, layout in STORAGE_LAYOUTS.items():
            collateral = calls.get(selector("collaterals(bytes16)"))
            if collateral:
                slot = collateral_module.mapping_slot(UUID(EXECUTOR).bytes.ljust(32, b"\0"), layout["collaterals"])
                words[collateral_module.evm_storage_key(contract, slot)] = collateral[2:66]
            reclaim = calls.get(selector("reclaims(uint256)"))
            for reclaim_id in range(16) if reclaim else ():
                base = collateral_module.mapping_slot(reclaim_id.to_bytes(32, "big"), layout["reclaims"])
                fields = [reclaim[2 + 64 * i : 66 + 64 * i] for i in range(4)]
                # a bytes16 getter output is left-aligned; in storage it is the slot's low-order bytes
                fields[0] = fields[0][:32].rjust(64, "0")
                for i, field in enumerate(fields):
                    words[collateral_module.evm_storage_key(contract, base + i)] = field
        return {key: "0x" + word for key, word in words.items() if int(word, 16)}

    def substrate_answer(self, method, params, view: str = "a") -> dict:
        """chain_getBlockHash and state_getStorage as a Subtensor node answers them: a state query at a hash it
        does not have is "UnknownBlock", never another block's state. `view` "lagging" has no block, "sibling" has
        a sibling of each block at its number, with `sibling_calls` state."""
        kept = range(self.oldest_kept, self.block_number + 1) if view != "lagging" else range(0)
        if method == "chain_getBlockHash":
            return {"jsonrpc": "2.0", "id": 1, "result": self.substrate_hash(params[0], view) if params[0] in kept else None}
        key, at = params
        number = next((n for n in kept if self.substrate_hash(n, view) == at), None)
        if number is None:
            return {"jsonrpc": "2.0", "id": 1, "error": {"code": 4003, "message": f"UnknownBlock: {at}"}}
        if key == collateral_module.ethereum_block_hash_key(number):
            return {"jsonrpc": "2.0", "id": 1, "result": "0x" + "5c" * 32 if view == "sibling" else self.chain_hash(number)}
        calls = {**self.calls, **self.sibling_calls} if view == "sibling" else self.calls
        return {"jsonrpc": "2.0", "id": 1, "result": self.storage_of(calls).get(key)}

    async def make_request(self, method, params):
        self.requests.append((method, params))
        if method in SUBSTRATE_READS:
            return self.substrate_answer(method, params)
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
                if number < self.oldest_kept:
                    return {"jsonrpc": "2.0", "id": 1, "result": None}
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
        if method == "eth_getLogs":
            block_hash = params[0].get("blockHash")
            logs = [log for log in self.logs if block_hash in (None, log["blockHash"])]
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
        if method == "eth_getTransactionReceipt" and self.listed_receipt(params[0]) is not None:
            withheld = params[0] in self.withheld_receipts
            return {"jsonrpc": "2.0", "id": 1, "result": None if withheld else self.listed_receipt(params[0])}
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


async def lost_receipt(*_args, **_kwargs):
    raise ConnectionError(f"Could not reach {RPC_URL}/?apikey=secret-rpc-key")


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


OLD_CONTRACT = "0x999F9A49A85e9D6E981cad42f197349f50172bEB"


FORK_B = "0x" + "0b" * 32


def fork_b_log(url="https://fork-b/reclaim"):
    return {**started_log(url=url), "blockHash": FORK_B}


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


@pytest.fixture
def archive_network(monkeypatch):
    from core.config import settings

    wallet = SimpleNamespace(get_hotkey=lambda: SimpleNamespace(ss58_address="5" + "C" * 47))
    monkeypatch.setattr(settings, "BITTENSOR_NETWORK", "archive")
    monkeypatch.setattr(settings, "SUBTENSOR_EVM_RPC_URL", None)
    monkeypatch.setattr(type(settings), "get_bittensor_wallet", lambda self: wallet)
    return settings


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
        (["reclaim-collateral", "--executor_uuid", EXECUTOR], f"{MINER_KEY}\n"),
        (["reclaim-collateral", "--executor_uuid", EXECUTOR, "--contract", "1.0.2"], f"{MINER_KEY}\n"),
        (["finalize-reclaim-request", "--reclaim-request-id", "5"], f"{MINER_KEY}\n"),
        (["reclaim-collateral", "--executor_uuid", EXECUTOR, "--private-key", MINER_KEY], None),
        (["finalize-reclaim-request", "--reclaim-request-id", "5", "--private-key", MINER_KEY], None),
        (["get-balance-of-eth-address", "--private-key", MINER_KEY], None),
        (["finalize-reclaim-request", "--reclaim-request-id", "5", "--contract", "1.0.2"], f"{MINER_KEY}\n"),
        (["get-miner-collateral", "--contract", "1.0.2"], None),
        (["get-executor-collateral", "--address", "192.0.2.10", "--port", "8001"]
         + ["--contract", "1.0.2"], None),
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


@pytest.mark.parametrize(
    "case,error",
    [
        ("orphaned", "not the finalized block at that number"),
        ("final", None),
        # read a of a5079c9: the default finney RPC keeps about 256 blocks and answers HTTP 429 after about 100 reads
        ("pruned", "no longer keeps.*SUBTENSOR_EVM_RPC_URL"),
        ("rate-limited", "could not be read \\(HTTP 429.*SUBTENSOR_EVM_RPC_URL"),
        # review of 7a226f9: a gateway answers the finalized block from fork A and the receipt's block from fork B
        ("split-batch", "from more than one chain"),
        # reads of a7a2820: a receipt 120 blocks below the finalized block is read in batches the RPC accepts
        ("deep", None),
    ],
)
async def test_a_fresh_send_record_clears_only_on_a_receipt_on_the_finalized_chain(monkeypatch, case, error):
    monkeypatch.setattr(collateral_module, "RATE_LIMIT_RETRY_SEC", (0, 0, 0))
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()}, logs=[reclaimed_log()])
    if case == "deep":
        provider.block_number = provider.finalized_number = 16 + 120
    elif case != "orphaned":
        provider.block_number = provider.finalized_number = 20
    if case == "orphaned":
        provider.canonical_hashes[16] = "0x" + "bb" * 32
    elif case == "pruned":
        provider.oldest_kept = 17
    elif case == "rate-limited":
        provider.batch_backends = ["429"] * 4
    elif case == "split-batch":
        provider.canonical_hashes[16] = "0x" + "bb" * 32
        provider.batch_backends, provider.fork_b_hash = ["b", "b"], BLOCK_HASH
    client = client_with(provider)

    if error is None:
        await client.finalize_reclaim(5)
        assert client._read_sent_record(CHAIN_ID) is None
        return
    with pytest.raises(CollateralOutcomeUnknownError, match=error):
        await client.finalize_reclaim(5)
    assert client._read_sent_record(CHAIN_ID)["hash"] == sent_hash(provider)


async def test_a_mined_reclaim_whose_receipt_lacks_its_event_keeps_the_record_until_a_receipt_has_it(monkeypatch):
    """Review of e02f9ff: a status=1 receipt with no ReclaimProcessStarted cleared the record, and a retry reverts
    AmountZero, so the request ID was lost."""
    provider = FakeProvider(logs=[])
    client = client_with(provider)
    with pytest.raises(CollateralOutcomeUnknownError, match="no ReclaimProcessStarted event"):
        await client.reclaim_collateral(EXECUTOR)
    assert client._read_sent_record(CHAIN_ID)["hash"] == sent_hash(provider)

    provider.nonce = provider.pending_nonce = NONCE + 1
    with pytest.raises(CollateralOutcomeUnknownError, match="no ReclaimProcessStarted event"):
        await client.settle_earlier_send()
    assert client._read_sent_record(CHAIN_ID) is not None

    provider.logs = [started_log(reclaim_request_id=12)]
    with pytest.raises(CollateralTransactionError, match="it started reclaim request 12;"):
        await client.settle_earlier_send()
    assert client._read_sent_record(CHAIN_ID) is None
    assert len(provider.sent) == 1
