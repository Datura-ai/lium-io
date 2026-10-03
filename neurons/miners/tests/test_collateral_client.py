"""CollateralClient against a fake JSON-RPC provider: what it signs and sends, and when it needs an RPC URL."""

import logging
import traceback
from decimal import Decimal
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
# finney key suffixes, checked with state_getStorage against the 1.0.2 contract's getters
FINNEY_RECLAIM_22_AMOUNT_KEY = "3a4765a05b7577b434fb25997e2c08b3b8657d180a4d2444fb942e94a4266075e5a1b59d96d88e88cf308d6927f00ff4"
FINNEY_BLOCK_HASH_KEY = (
    "0x2013754dd003840aea66b349f8241e25a44704b568d21667356a5a050c118746"
    "d4adcd30b8b10cced0608c0000000000000000000000000000000000000000000000000000000000"
)
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
    unknown = f"nonce {NONCE} is used, so its outcome is unknown"
    for _ in range(2):
        with pytest.raises(CollateralOutcomeUnknownError, match=unknown) as raised:
            await client.finalize_reclaim(5)
    assert "SUBTENSOR_EVM_RPC_URL set to an RPC that serves its receipt" in str(raised.value)
    assert f"delete {client.sent_record_path}" in str(raised.value)
    assert "no transaction was sent" in str(raised.value)
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


@pytest.mark.parametrize("contract", ["1.0.2", "1.0.0"])
async def test_a_record_this_key_signed_for_the_contract_is_replaced_from_its_verified_bytes(contract):
    """The replacement command builds the default 1.0.2 client; a 1.0.0 reclaim is still this key's send."""
    from core import utils

    to = CONTRACT if contract == "1.0.2" else OLD_CONTRACT
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    client = utils.get_collateral_contract(miner_key=MINER_KEY)
    client._w3 = AsyncWeb3(provider)
    record = signed_record(to=to)
    # loose fields next to the bytes are never what gets signed
    client._write_sent_record(CHAIN_ID, {**record, "to": ATTACKER, "value": 10**18, "data": "0x"})

    outcome = await client.replace_earlier_send()
    assert "the replacement, succeeded" in outcome
    replacement = decode_legacy(provider.sent[-1])
    assert replacement["to"] == to and replacement["value"] == 0 and replacement["nonce"] == NONCE
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


OLD_CONTRACT = "0x999F9A49A85e9D6E981cad42f197349f50172bEB"


@pytest.mark.parametrize(
    "recorded,error",
    [(True, "not sent to a configured collateral contract"), (False, "nothing was replaced")],
    ids=["unconfigured-contract", "no-record"],
)
async def test_nothing_is_replaced_without_a_recorded_send_to_a_configured_contract(recorded, error):
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    client = client_with(provider)
    if recorded:
        client._write_sent_record(CHAIN_ID, signed_record(to=OLD_CONTRACT))

    with pytest.raises(CollateralTransactionError, match=error):
        await client.replace_earlier_send()
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


def storage_reads(provider) -> list:
    return [params for method, params in provider.requests if method == "state_getStorage"]


async def test_a_reclaim_request_read_at_a_block_hash_names_that_block():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.block_number, provider.finalized_number = 5001, 5000
    client = client_with(provider)

    block_hash = await client.finalized_block_hash()
    reclaim = await client.get_reclaim_request(5, block_hash=block_hash)

    assert reclaim == (UUID(EXECUTOR).bytes, MINER, 10**17, 0)
    reads = storage_reads(provider)
    assert [at for _, at in reads] == [provider.substrate_hash(5000)] * 5
    assert reads[0][0] == collateral_module.ethereum_block_hash_key(5000)
    assert "eth_call" not in [method for method, _ in provider.requests]


def test_the_storage_keys_are_the_ones_finney_answers():
    """On finney, at a finalized block, state_getStorage at these keys answered the 1.0.2 contract's open reclaim
    request 22 word for word as reclaims(22) did, and BlockHash[9199824] the EVM block's hash."""
    reclaim = collateral_module.mapping_slot((22).to_bytes(32, "big"), STORAGE_LAYOUTS[CONTRACT]["reclaims"])
    assert collateral_module.evm_storage_key(CONTRACT, reclaim + 2) == (
        "0x1da53b775b270400e7e61ed5cbc5a146ab1160471b1418779239ba8e2b847e42"
        "81e8cde4a494fd9b81dba034e0b8913d8a4023fdd1eaa7b242f3723a7d096b6cc693c7c6" + FINNEY_RECLAIM_22_AMOUNT_KEY
    )
    assert collateral_module.ethereum_block_hash_key(9199824) == FINNEY_BLOCK_HASH_KEY


async def test_a_read_pinned_to_a_block_the_rpc_does_not_have_fails():
    """Review 5395259575 at 0a6fee9: Frontier ignores requireCanonical and answers a call at an unknown hash from
    its pending state. The header read by hash is null there, so the read fails instead of answering."""
    provider = FakeProvider(calls={selector("collaterals(bytes16)"): hex_encode(["uint256"], [10**16])})
    provider.block_number = provider.finalized_number = 5000
    client = client_with(provider)

    with pytest.raises(collateral_module.RpcReadError, match="does not have the block"):
        await client.get_executor_collateral(EXECUTOR, block_hash=bytes.fromhex("ee" * 32))
    assert await client.get_executor_collateral(EXECUTOR, block_hash=await client.head_parent_hash()) == Decimal("0.01")


async def test_a_pinned_read_makes_no_eth_call(monkeypatch):
    """A pinned read is the contract's storage at the Substrate block: no eth_call, so neither Frontier's pending
    fallback nor the EVM creator whitelist nor state override support can change it."""
    from core import utils

    provider = FakeProvider(calls={selector("collaterals(bytes16)"): hex_encode(["uint256"], [10**16])})
    provider.block_number = provider.finalized_number = 5000
    client = client_with(provider)
    monkeypatch.setattr(utils, "get_collateral_contract", lambda version=None: client)
    monkeypatch.setattr(utils.settings, "CONTRACT_VERSIONS", {"1.0.2": CONTRACT})

    assert await client.get_executor_collateral(EXECUTOR, block_hash=await client.head_parent_hash()) == Decimal("0.01")
    assert "eth_call" not in [method for method, _ in provider.requests]
    assert await utils.versions_holding_collateral(EXECUTOR) == ["1.0.2"]


async def test_a_contract_with_no_known_storage_layout_is_not_read_at_a_pinned_block():
    provider = FakeProvider(calls={selector("collaterals(bytes16)"): hex_encode(["uint256"], [10**16])})
    provider.block_number = provider.finalized_number = 5000
    other = CollateralClient(network="finney", contract_address="0x" + "42" * 20)
    other._w3 = AsyncWeb3(provider)

    with pytest.raises(collateral_module.RpcReadError, match="no storage layout"):
        await other.get_executor_collateral(EXECUTOR, block_hash=await other.head_parent_hash())


@pytest.mark.parametrize(
    "word, error",
    [
        ("0x" + "11" * 31, "31 bytes"),
        ("0x" + "11" * 33, "33 bytes"),
        ("0xzz", "no word"),
        (7, "no word"),
    ],
)
def test_a_storage_answer_that_is_not_a_32_byte_word_fails(word, error):
    with pytest.raises(collateral_module.RpcReadError, match=error):
        collateral_module.storage_word(word)


async def test_a_reclaim_word_with_bits_outside_its_field_fails():
    """A word that holds more than its field (a bytes16 slot with high-order bits, say) is not that field: the
    layout is not the one the contract has, and the read fails instead of guessing."""
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.block_number = provider.finalized_number = 5000
    client = client_with(provider)
    storage_of = provider.storage_of

    def wide_executor_id(calls):
        words = storage_of(calls)
        base = collateral_module.mapping_slot((5).to_bytes(32, "big"), STORAGE_LAYOUTS[CONTRACT]["reclaims"])
        words[collateral_module.evm_storage_key(CONTRACT, base)] = "0x" + "ff" * 32
        return words

    provider.storage_of = wide_executor_id
    with pytest.raises(collateral_module.RpcReadError, match="more than the value"):
        await client.get_reclaim_request(5, block_hash=await client.finalized_block_hash())


async def test_remove_executor_fails_when_its_pinned_read_reaches_a_backend_without_the_block(monkeypatch):
    """Review 5397109362: the removal guard's pinned read, sent by a gateway to a backend without the block, fails
    instead of letting the executor's local row go while TAO is still locked."""
    from core import utils

    provider = FakeProvider(calls={selector("collaterals(bytes16)"): hex_encode(["uint256"], [10**16])})
    client = client_with(provider)
    monkeypatch.setattr(utils, "get_collateral_contract", lambda version=None: client)
    monkeypatch.setattr(utils.settings, "CONTRACT_VERSIONS", {"1.0.2": CONTRACT})

    for backends in (["a", "split"], ["a", "a", "split"]):
        provider.batch_backends = list(backends)
        with pytest.raises(collateral_module.RpcReadError):
            await utils.versions_holding_collateral(EXECUTOR)
    assert await utils.versions_holding_collateral(EXECUTOR) == ["1.0.2"]


SPLIT_ROUTES = {
    "number-to-lagging": (["a", "split"], "no Substrate block"),
    "state-to-lagging": (["a", "a", "split"], "an error answer"),
    "number-to-sibling-canonical": (["a", "sibling"], "an error answer"),
    "state-to-sibling-canonical": (["a", "a", "sibling"], "an error answer"),
    "all-to-sibling-canonical": (["a", "sibling", "sibling"], "did not build"),
    "block-hash-to-a-words-to-sibling": (["a", "a", "mixed"], "an error answer"),
    "sibling-hash-then-mixed": (["a", "sibling", "mixed"], "an error answer"),
}


@pytest.mark.parametrize("route", list(SPLIT_ROUTES), ids=list(SPLIT_ROUTES))
@pytest.mark.parametrize(
    "at_block, off_the_block",
    [pytest.param(0, 10**16, id="deposit-only-off-the-block"), pytest.param(10**16, 0, id="reclaim-only-off-the-block")],
)
async def test_state_off_the_pinned_block_is_never_read(route, at_block, off_the_block):
    """Review 5398901003 at 4fa570d: the pinned block has one amount, its sibling (a lagging backend's pending block,
    or the canonical one on another backend) and its child another, and a gateway sends each read to another
    backend. Each storage read names the Substrate block by hash, which a backend answers from that block or
    refuses, and BlockHash[number] read at that hash must be the pinned EVM block's; so every split fails and the
    one-backend read is the block's own amount."""
    key = selector("collaterals(bytes16)")
    provider = FakeProvider(calls={key: hex_encode(["uint256"], [at_block])})
    provider.sibling_calls = {key: hex_encode(["uint256"], [off_the_block])}
    provider.block_number = provider.finalized_number = 5000
    client = client_with(provider)
    block_hash = bytes.fromhex(provider.chain_hash(4999)[2:])

    backends, error = SPLIT_ROUTES[route]
    provider.batch_backends = list(backends)
    with pytest.raises(collateral_module.RpcReadError, match=error):
        await client.get_executor_collateral(EXECUTOR, block_hash=block_hash)
    assert await client.get_executor_collateral(EXECUTOR, block_hash=block_hash) == Decimal(at_block) / 10**18


async def test_a_read_pinned_to_the_head_needs_no_child():
    provider = FakeProvider(calls={selector("collaterals(bytes16)"): hex_encode(["uint256"], [10**16])})
    provider.block_number = provider.finalized_number = 5000
    client = client_with(provider)

    head = bytes.fromhex(provider.chain_hash(5000)[2:])
    assert await client.get_executor_collateral(EXECUTOR, block_hash=head) == Decimal("0.01")


async def test_remove_executor_reads_at_the_head_parent(monkeypatch):
    from core import utils

    provider = FakeProvider(calls={selector("collaterals(bytes16)"): hex_encode(["uint256"], [10**16])})
    provider.block_number = 5000
    client = client_with(provider)
    monkeypatch.setattr(utils, "get_collateral_contract", lambda version=None: client)
    monkeypatch.setattr(utils.settings, "CONTRACT_VERSIONS", {"1.0.2": CONTRACT})

    assert await utils.versions_holding_collateral(EXECUTOR) == ["1.0.2"]
    assert {at for _, at in storage_reads(provider)} == {provider.substrate_hash(4999)}

    provider.calls = {selector("collaterals(bytes16)"): hex_encode(["uint256"], [0])}
    assert await utils.versions_holding_collateral(EXECUTOR) == []


FORK_B = "0x" + "0b" * 32
# the batches of one listing: the headers below the finalized block, then the receipts of the one block with a log
LIST_BATCHES = collateral_module.RECLAIM_LOOKBACK_BLOCKS // collateral_module.CHAIN_READ_BATCH + 1


def fork_b_log(url="https://fork-b/reclaim"):
    return {**started_log(url=url), "blockHash": FORK_B}


@pytest.mark.parametrize(
    "backends,amount,urls",
    [
        # the first batch reaches backend B (its block at the finalized number is B's), the second backend A
        (["b"], 10**17, ["https://fork-a/reclaim"]),
        # a lagging backend has no block at the finalized number and answers the log range with []
        (["lagging"], 10**17, ["https://fork-a/reclaim"]),
        (["b"] * collateral_module.RECLAIM_LIST_ATTEMPTS, 10**17, None),
        # a gateway that splits a batch: the log's amount is not the state at the finalized block
        ([], 2 * 10**17, None),
        # review of 7a226f9: the batch's blocks reach A and its logs a lagging backend, which knows no such block
        (["logs-lagging"] * LIST_BATCHES, 10**17, None),
        # review of 12d1599: the logs reach a backend that knows the block's hash but answers it with no logs
        (["logs-empty"] * LIST_BATCHES * collateral_module.RECLAIM_LIST_ATTEMPTS, 10**17, None),
        # review of e6f4b88: a partly synced backend leaves out the one transaction holding this contract's started
        # log, and the block's other logs (this contract's Reclaimed, the old contract's started log with the same
        # indexed values) set every bloom bit the left-out log sets
        ("partial-covered-bloom", 10**17, None),
    ],
    ids=[
        "b-then-a", "lagging-then-a", "b-every-time", "log-and-state-disagree", "split-empty-logs",
        "known-block-no-logs", "partial-covered-bloom",
    ],
)
async def test_the_open_reclaim_list_never_mixes_logs_and_state_of_two_forks(backends, amount, urls):
    """A load-balanced RPC answers the finalized block from backend A and other reads from a lagging backend B or
    one on another fork (reviews of 21ec2fd: A's log with B's amount; B's empty log answer). The list is A's, or an
    error."""
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim(amount)})
    provider.block_number, provider.finalized_number = 5001, 5000
    at = {"blockNumber": hex(4500), "blockHash": provider.chain_hash(4500)}
    started = {**started_log(url="https://fork-a/reclaim"), **at}
    provider.logs = [started]
    if backends == "partial-covered-bloom":
        others = [
            {**reclaimed_log(), **at, "transactionHash": "0x" + "02" * 32, "logIndex": "0x1"},
            {**started_log(), **at, "address": OLD_CONTRACT, "transactionHash": "0x" + "03" * 32, "logIndex": "0x2"},
        ]
        provider.logs = [started, *others]
        assert collateral_module.logs_bloom(others) == collateral_module.logs_bloom(provider.logs)
        provider.withheld_receipts = {listed_tx(at["blockHash"], started["transactionHash"])}
        backends = []
    provider.batch_backends = list(backends)

    if urls is None:
        with pytest.raises(CollateralTransactionError, match="not on the finalized chain|did not answer"):
            await client_with(provider).get_reclaim_events()
        return
    requests = await client_with(provider).get_reclaim_events()

    assert [(request.url, request.block_number) for request in requests] == [(urls[0], 4500)]
    assert {at for _, at in storage_reads(provider)} == {provider.substrate_hash(5000)}


async def test_a_block_with_events_of_both_contracts_lists_each_contracts_own_requests():
    """Review of a7a2820: a block holds this contract's Reclaimed and the old contract's ReclaimProcessStarted, so
    its bloom holds this address and the started topic, though no log holds both."""
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.block_number, provider.finalized_number = 5001, 5000
    at = {"blockNumber": hex(4500), "blockHash": provider.chain_hash(4500)}
    provider.logs = [{**reclaimed_log(), **at}, {**started_log(), **at, "address": OLD_CONTRACT, "logIndex": "0x1"}]

    assert await client_with(provider).get_reclaim_events() == []
    old = CollateralClient(network="finney", contract_address=OLD_CONTRACT)
    old._w3 = AsyncWeb3(provider)
    assert [request.reclaim_request_id for request in await old.get_reclaim_events()] == [5]


async def test_a_batch_above_the_rpc_limit_fails_the_list(monkeypatch):
    """Reads of a7a2820: the default finney RPC refuses a batch of more than 50 items, and the fake RPC does too."""
    monkeypatch.setattr(collateral_module, "CHAIN_READ_BATCH", FakeProvider.BATCH_LIMIT + 1)
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.block_number = provider.finalized_number = 5000

    with pytest.raises(CollateralTransactionError, match="did not answer the batch"):
        await client_with(provider).get_reclaim_events()


async def test_a_rate_limited_list_backs_off_and_reads_a_bounded_number_of_blocks(monkeypatch, caplog):
    """Read a of a5079c9: the listing read every one of 1000 headers, and the default finney RPC answers HTTP 429
    after about 100 reads and keeps only about 256 blocks."""
    monkeypatch.setattr(collateral_module, "RATE_LIMIT_RETRY_SEC", (0, 0, 0))
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.block_number, provider.finalized_number = 5001, 5000
    provider.logs = [{**started_log(), "blockNumber": hex(4900), "blockHash": provider.chain_hash(4900)}]
    provider.oldest_kept = 5000 - 255
    provider.batch_backends = ["429", "429"]

    requests = await client_with(provider).get_reclaim_events()

    assert [request.reclaim_request_id for request in requests] == [5]
    # the finalized block, batches of headers down to the first pruned one (4744), the logs of the one block whose
    # bloom may hold the event, and the one request's state: BlockHash and its four storage words
    methods = [method for method, _ in provider.requests]
    header_batches = -(-(5000 - provider.oldest_kept + 1) // collateral_module.CHAIN_READ_BATCH)
    assert methods.count("eth_getBlockByNumber") == 1 + header_batches * collateral_module.CHAIN_READ_BATCH
    assert (methods.count("eth_getTransactionReceipt"), methods.count("state_getStorage")) == (1, 5)
    assert "eth_call" not in methods
    assert "eth_getLogs" not in methods
    assert "no longer keeps block 4000" in caplog.text

    provider.batch_backends = ["429"] * 4
    with pytest.raises(CollateralTransactionError, match="HTTP 429.*SUBTENSOR_EVM_RPC_URL"):
        await client_with(provider).get_reclaim_events()


async def test_the_open_reclaim_list_is_read_at_the_finalized_block_not_the_head():
    provider = FakeProvider(calls={selector("reclaims(uint256)"): open_reclaim()})
    provider.block_number, provider.finalized_number = 5003, 5000
    provider.logs = [{**started_log(), "blockNumber": hex(4900), "blockHash": provider.chain_hash(4900)}]

    requests = await client_with(provider).get_reclaim_events()

    assert [request.reclaim_request_id for request in requests] == [5]
    numbers = [params[0] for method, params in provider.requests if method == "eth_getBlockByNumber"]
    assert numbers[0] == "finalized" and max(int(number, 16) for number in numbers[1:]) == 4999
    assert {at for _, at in storage_reads(provider)} == {provider.substrate_hash(5000)}


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

    monkeypatch.setattr(collateral_module, "BatchHTTPProvider", fake_http_provider)
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
        (["reclaim-collateral", "--executor_uuid", EXECUTOR], 1),
    ],
    ids=["non-collateral-command", "collateral-command"],
)
def test_a_command_on_an_unknown_network_runs_or_names_the_setting(archive_network, caplog, args, exit_code):
    from cli import cli

    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(cli, args, input=f"{MINER_KEY}\n")
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
        result = CliRunner().invoke(cli_module.cli, ["replace-collateral-transaction"], input=f"{MINER_KEY}\n")
    assert result.exit_code == exit_code, result.output
    assert ("✅" in caplog.text) is (exit_code == 0)
    assert MINER_KEY.removeprefix("0x")[:16] not in caplog.text + result.output


def test_replace_collateral_transaction_takes_no_key_on_the_command_line(monkeypatch):
    import cli as cli_module

    monkeypatch.setattr(cli_module, "get_collateral_contract", lambda **_: pytest.fail("nothing may be signed"))
    result = CliRunner().invoke(cli_module.cli, ["replace-collateral-transaction", "--private-key", MINER_KEY])
    assert result.exit_code == 2
    assert "No such option" in result.output


@pytest.mark.parametrize(
    "url,origin",
    [
        ("https://user:pw@evm.example.invalid:8443/v2/key?apikey=key", "https://evm.example.invalid:8443"),
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
        ["reclaim-collateral", "--executor_uuid", EXECUTOR],
        ["finalize-reclaim-request", "--reclaim-request-id", "5"],
        # --contract skips detection: the send itself fails, and that must exit 1 too
        ["reclaim-collateral", "--executor_uuid", EXECUTOR, "--contract", "1.0.2"],
        ["finalize-reclaim-request", "--reclaim-request-id", "5", "--contract", "1.0.2"],
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
        input=f"{MINER_KEY}\n",
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
