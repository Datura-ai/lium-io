"""Read and reclaim provider collateral on the Lium collateral contract (Bittensor EVM).

Collateral is optional for providers; this module keeps the withdrawal path for executors
that still hold a deposit. It covers the contract calls the miner CLI makes: balance and
collateral reads, start a reclaim, list open reclaims, finalize a reclaim.
"""

import asyncio
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import pathlib
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit
from uuid import UUID

import aiohttp
import rlp
from bittensor_wallet import Keypair
from eth_account import Account
from eth_utils import event_abi_to_log_topic
from web3 import AsyncHTTPProvider, AsyncWeb3
from web3._utils.method_formatters import log_entry_formatter
from web3._utils.request import async_make_post_request
from web3.exceptions import ContractLogicError, TransactionNotFound
from web3.logs import DISCARD

logger = logging.getLogger(__name__)

ABI_PATH = pathlib.Path(__file__).with_name("collateral_abi.json")

RPC_URLS = {
    "local": "http://127.0.0.1:9944",
    "test": "https://test.finney.opentensor.ai",
    "finney": "https://lite.chain.opentensor.ai",
}
# The Bittensor EVM chain of each network. A transaction is signed for this chain only, so an RPC that reports
# another one (Ethereum mainnet, say) never gets a transaction it could broadcast there.
CHAIN_IDS = {"finney": 964, "archive": 964, "test": 945, "local": 42}
# The signed transaction whose outcome is not known yet, per chain and address. It lives next to the wallets
# (the directory the miner container mounts from the host), so it survives a container rebuild. No other nonce
# is signed while it is there: an unmined one is broadcast again as the same bytes, so at most one transaction
# per nonce can ever be mined.
DEFAULT_SENT_RECORD = "~/.bittensor/wallets/.lium-collateral-sent.json"
SENT_RECORD_PATH = pathlib.Path(DEFAULT_SENT_RECORD).expanduser()
RETRY_IS_SAFE = "Run this again: it reads this transaction's outcome first and sends nothing new until it is known"
# JSON-RPC send refusals and the local text each is reported with. None of them clears the record: a gateway can
# forward the bytes to one upstream, lose its answer and return another upstream's refusal, so every error answer
# to a send is an unknown outcome, and the next run settles it from the same bytes. "nonce too low" is not listed:
# web3's retry middleware and gateways resend eth_sendRawTransaction, and the retry of a transaction that landed
# gets that answer. The RPC's own text never reaches an error: it can echo the keyed RPC URL.
SEND_REFUSALS = {
    "insufficient funds": "insufficient funds for gas",
    "underpriced": "gas price too low",
    "intrinsic gas too low": "gas limit too low",
    "exceeds block gas limit": "gas limit above the block limit",
    "invalid sender": "invalid signature",
}
ALREADY_KNOWN = {"already known", "known transaction"}

RECEIPT_TIMEOUT_SEC = 300
RECEIPT_POLL_SEC = 2
# A replacement at the same nonce must outbid the transaction it replaces; pools refuse a smaller bump.
REPLACEMENT_PRICE_BUMP = (9, 8)
RECLAIM_LIST_ATTEMPTS = 3
# The contract functions this client signs; a recorded transaction that calls anything else is not re-signed.
SENT_FUNCTIONS = ("reclaimCollateral(bytes16,string,bytes16)", "finalizeReclaim(uint256)")

GAS_LIMIT = 200_000
# The RPC quotes the gas price; above this ceiling nothing is signed, so a faulty or hostile RPC
# cannot spend the address balance on fees (at GAS_LIMIT, 100 gwei caps a transaction at 0.02 TAO).
DEFAULT_MAX_GAS_PRICE_GWEI = 100
RECLAIM_LOOKBACK_BLOCKS = 1000
# Reads per JSON-RPC batch. The default finney RPC refuses a batch of more than 50 (-32010).
CHAIN_READ_BATCH = 50
# A pruning RPC keeps at least this many blocks below its finalized one (the default finney RPC about 256); a block
# missing nearer the top is a backend that lags behind, not pruning.
KEPT_BLOCKS_MIN = 128
# The default finney RPC answers HTTP 429 after about 100 reads in 30 s; a batch waits this long before each retry.
RATE_LIMIT_RETRY_SEC = (2, 5, 10)
# An address with no code and no precompile. A pinned read is a call to it with pinned_read_code set as its code
# by eth_call's state override. A contract-creation eth_call would go through the runtime's creator whitelist
# (WhitelistedCreators, DisableWhitelistCheck), which a plain call does not.
PINNED_READ_ADDRESS = "0x00000000000000000000000000000000000c0113"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"

SS58_FORMAT = 42


class CollateralTransactionError(Exception):
    pass


class CollateralConfigError(Exception):
    pass


class CollateralOutcomeUnknownError(CollateralTransactionError):
    """A transaction was broadcast but its receipt was never read: it may still be mined."""


class RpcReadError(Exception):
    """A batch of reads the RPC did not answer; the text is local, never the RPC's (it can echo a keyed URL)."""


class BatchHTTPProvider(AsyncHTTPProvider):
    async def make_batch_request(self, requests: list[tuple[str, list]]):
        body = json.dumps([{"jsonrpc": "2.0", "id": i, "method": m, "params": p} for i, (m, p) in enumerate(requests)])
        return json.loads(await async_make_post_request(self.endpoint_uri, body, **self.get_request_kwargs()))


@dataclass
class ReclaimRequest:
    reclaim_request_id: int
    executor_uuid: str
    miner: str
    amount: float
    expiration_time: str
    url: str
    url_content_md5_checksum: str
    block_number: int


def h160_to_ss58(h160_address: str) -> str:
    """The SS58 account (generic prefix 42) that mirrors an EVM (H160) address on Bittensor.

    Same mapping as opentensor/evm-bittensor examples/address-mapping.js: blake2b-256 of
    b"evm:" + the 20 address bytes, SS58-encoded.
    """
    address_bytes = bytes.fromhex(h160_address.removeprefix("0x"))
    public_key = hashlib.blake2b(b"evm:" + address_bytes, digest_size=32).digest()
    return Keypair(public_key=public_key.hex(), ss58_format=SS58_FORMAT).ss58_address


def rpc_origin(rpc_url: str | None) -> str | None:
    """Scheme and host of an RPC URL, for logs: its path, query and userinfo can carry an API key."""
    if not rpc_url:
        return None
    try:
        parts = urlsplit(rpc_url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return "<unparsed>"
    if not parts.scheme or not host:
        return "<unparsed>"
    return f"{parts.scheme}://{host}" + (f":{port}" if port else "")


def executor_uuid_bytes(executor_uuid: str) -> bytes:
    return UUID(executor_uuid).bytes


def hash_text(value) -> str:
    if isinstance(value, bytes | bytearray):
        value = value.hex()
    return str(value or "").lower().removeprefix("0x")


def same_hash(a, b) -> bool:
    return bool(hash_text(a)) and hash_text(a) == hash_text(b)


def block_number(block) -> int:
    number = block["number"]
    return int(number, 16) if isinstance(number, str) else number


def pinned_read_code(to: str, calls: list[tuple[bytes, int]]) -> str:
    """Code that makes each (calldata, output size) view call to `to` in one EVM run and returns NUMBER,
    BLOCKHASH(NUMBER - 1) and TIMESTAMP, then each call's first `size` bytes. One run reads one state on one
    backend, and the header words prove which block that state is. It is run by pinned_call."""

    def push2(value: int) -> bytes:
        return b"\x61" + value.to_bytes(2, "big")

    scratch = 0x2000
    code = bytearray()
    data_slots: list[int] = []
    revert_slots: list[int] = []
    out = 0x60
    for data, size in calls:
        # CODECOPY(scratch, <data offset>, len)
        code += push2(len(data))
        data_slots.append(len(code) + 1)
        code += push2(0) + push2(scratch) + b"\x39"
        # STATICCALL(gas, to, scratch, len, out, size); revert on failure or a short answer
        code += push2(size) + push2(out) + push2(len(data)) + push2(scratch)
        code += b"\x73" + bytes.fromhex(to.removeprefix("0x")) + b"\x5a\xfa\x15"
        revert_slots.append(len(code) + 1)
        code += push2(0) + b"\x57" + push2(size) + b"\x3d\x10"
        revert_slots.append(len(code) + 1)
        code += push2(0) + b"\x57"
        out += size
    # memory[0:0x60] = NUMBER, BLOCKHASH(NUMBER - 1), TIMESTAMP; RETURN(0, out)
    code += b"\x43\x60\x00\x52" + b"\x60\x01\x43\x03\x40\x60\x20\x52" + b"\x42\x60\x40\x52"
    code += push2(out) + b"\x60\x00\xf3"
    revert_at = len(code)
    code += b"\x5b\x60\x00\x80\xfd"
    for slot in revert_slots:
        code[slot : slot + 2] = revert_at.to_bytes(2, "big")
    offset = len(code)
    for slot, (data, _size) in zip(data_slots, calls):
        code[slot : slot + 2] = offset.to_bytes(2, "big")
        offset += len(data)
    return "0x" + bytes(code).hex() + b"".join(data for data, _ in calls).hex()


def pinned_call(code: str, pin: dict) -> list:
    """eth_call params that run `code` at the block `pin` names, as the code of PINNED_READ_ADDRESS."""
    return [{"to": PINNED_READ_ADDRESS, "data": "0x"}, pin, {PINNED_READ_ADDRESS: {"code": code}}]


def _pinned_answer(result: str, sizes: list[int]) -> bytes:
    if not isinstance(result, str):
        raise RpcReadError("the pinned read has no answer")
    raw = bytes.fromhex(result.removeprefix("0x"))
    if not raw:
        # a call to an address with no code answers nothing: the RPC did not apply the state override
        raise RpcReadError("the pinned read answered 0 bytes; the RPC must support eth_call state overrides")
    if len(raw) != 0x60 + sum(sizes):
        raise RpcReadError(f"the pinned read answered {len(raw)} bytes")
    return raw


def pinned_outputs(result: str, header: dict, sizes: list[int]) -> list[bytes]:
    """The view call outputs of a pinned_read_code answer, once its header words match `header`. Frontier answers
    a block hash it does not know from its pending state, which carries another number, parent or timestamp."""
    raw = _pinned_answer(result, sizes)
    number, parent, timestamp = (raw[i : i + 32] for i in (0, 32, 64))
    if (
        int.from_bytes(number, "big") != block_number(header)
        or not same_hash(parent, header.get("parentHash"))
        or int.from_bytes(timestamp, "big") != int(header["timestamp"], 16)
    ):
        raise RpcReadError("the read did not run on the block it is pinned to")
    outputs, at = [], 0x60
    for size in sizes:
        outputs.append(raw[at : at + size])
        at += size
    return outputs


def child_outputs(result: str, header: dict, sizes: list[int]) -> list[bytes]:
    """The view call outputs of a pinned_read_code answer that ran on a child of `header`: NUMBER is one past it
    and BLOCKHASH(NUMBER - 1) is its hash, which only a chain holding that block can answer."""
    raw = _pinned_answer(result, sizes)
    if int.from_bytes(raw[0:32], "big") != block_number(header) + 1 or not same_hash(raw[32:64], header.get("hash")):
        raise RpcReadError("the read did not run on a child of the block it is pinned to")
    outputs, at = [], 0x60
    for size in sizes:
        outputs.append(raw[at : at + size])
        at += size
    return outputs


def bloom_may_hold(bloom, *values: bytes) -> bool:
    """Whether a block's 2048-bit logs bloom may hold a log with every one of `values` (address, topics): False
    only when the bloom proves it does not. A missing or malformed bloom proves nothing."""
    if isinstance(bloom, str):
        try:
            bloom = bytes.fromhex(bloom.removeprefix("0x"))
        except ValueError:
            return True
    if not isinstance(bloom, bytes | bytearray) or len(bloom) != 256:
        return True
    return all(bloom[255 - bit // 8] & (1 << (bit % 8)) for value in values for bit in bloom_bits(value))


def bloom_bits(value: bytes):
    digest = AsyncWeb3.keccak(value)
    for i in (0, 2, 4):
        yield int.from_bytes(digest[i : i + 2], "big") & 2047


def logs_bloom(logs) -> str:
    """The logs bloom of a block that holds exactly `logs`, as the header's hex text."""
    bloom = bytearray(256)
    for log in logs:
        for value in (log["address"], *log["topics"]):
            for bit in bloom_bits(bytes.fromhex(hash_text(value))):
                bloom[255 - bit // 8] |= 1 << (bit % 8)
    return bloom.hex()


def record_hashes(record: dict) -> list[str]:
    """Every signed transaction of a record at its nonce: the first send and each replacement."""
    return list(record.get("hashes") or [record["hash"]])


def decode_legacy_transaction(raw: str) -> dict:
    nonce, gas_price, gas, to, value, data, *_ = rlp.decode(bytes.fromhex(raw.removeprefix("0x")))
    return {
        "nonce": int.from_bytes(nonce, "big"),
        "gasPrice": int.from_bytes(gas_price, "big"),
        "gas": int.from_bytes(gas, "big"),
        "to": AsyncWeb3.to_checksum_address(to),
        "value": int.from_bytes(value, "big"),
        "data": AsyncWeb3.to_hex(data),
    }


class CollateralClient:
    def __init__(
        self,
        network: str,
        contract_address: str,
        rpc_url: str | None = None,
        miner_key: str | None = None,
        max_gas_price_gwei: float = DEFAULT_MAX_GAS_PRICE_GWEI,
        sent_record_path: str | os.PathLike | None = None,
        other_contract_addresses: tuple[str, ...] = (),
    ):
        self.network = network
        self.sent_record_path = pathlib.Path(sent_record_path or SENT_RECORD_PATH).expanduser()
        self.max_gas_price_gwei = max_gas_price_gwei
        self.rpc_url = rpc_url or RPC_URLS.get(network)
        self.contract_address = AsyncWeb3.to_checksum_address(contract_address)
        # the record is per key, not per contract: a replacement may be for a send to another contract version
        self.replaceable_contract_addresses = {
            self.contract_address, *(AsyncWeb3.to_checksum_address(address) for address in other_contract_addresses)
        }
        self.miner_account = Account.from_key(miner_key) if miner_key else None
        self.miner_address = self.miner_account.address if self.miner_account else None
        self._w3 = None
        self._contract = None

    @property
    def w3(self) -> AsyncWeb3:
        # Built on first contract call so that CLI commands which never touch the
        # contract still run on a network without a known EVM RPC endpoint.
        if self._w3 is None:
            if not self.rpc_url:
                raise CollateralConfigError(
                    f"No EVM RPC endpoint is known for BITTENSOR_NETWORK={self.network!r}; "
                    "set SUBTENSOR_EVM_RPC_URL to call the collateral contract"
                )
            self._w3 = AsyncWeb3(BatchHTTPProvider(self.rpc_url))
        return self._w3

    async def _read_together(self, *requests: tuple[str, list]) -> list:
        """The results of JSON-RPC reads sent as one batch, in request order. A gateway may send each item of a
        batch to another backend, so the answers are not taken to come from one view of the chain."""
        for delay in (*RATE_LIMIT_RETRY_SEC, None):
            try:
                answers = await self.w3.provider.make_batch_request(list(requests))
                break
            except aiohttp.ClientResponseError as error:
                if error.status != 429:
                    raise RpcReadError(f"HTTP {error.status}") from error
                if delay is None:
                    raise RpcReadError(f"HTTP 429, too many requests, {len(RATE_LIMIT_RETRY_SEC) + 1} times") from error
                await asyncio.sleep(delay)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
                raise RpcReadError(type(error).__name__) from error
        if not isinstance(answers, list) or len(answers) != len(requests) or not all(isinstance(a, dict) for a in answers):
            raise RpcReadError("the RPC did not answer the batch")
        answers = sorted(answers, key=lambda answer: answer.get("id", -1))
        if any("error" in answer or "result" not in answer for answer in answers):
            raise RpcReadError("an error answer")
        return [answer["result"] for answer in answers]

    @property
    def contract(self):
        if self._contract is None:
            self._contract = self.w3.eth.contract(
                address=self.contract_address, abi=json.loads(ABI_PATH.read_text())
            )
        return self._contract

    async def get_balance(self, address: str):
        balance = await self.w3.eth.get_balance(AsyncWeb3.to_checksum_address(address))
        return AsyncWeb3.from_wei(balance, "ether")

    async def get_executor_collateral(self, executor_uuid: str, block_hash=None):
        """The executor's collateral in TAO, at the block named by `block_hash` when one is given, else at latest."""
        function = self.contract.functions.collaterals(executor_uuid_bytes(executor_uuid))
        if block_hash is None:
            amount = await function.call()
        else:
            amount = await self._call_at_block_hash(function, block_hash)
        return AsyncWeb3.from_wei(amount, "ether")

    async def get_reclaim_request(self, reclaim_request_id: int, block_hash=None) -> tuple:
        """(executorId, miner, amount in wei, denyTimeout) of a reclaim request; amount 0 once it is closed.
        Read at the block named by `block_hash` when one is given, else at the latest block."""
        if block_hash is not None:
            return await self._reclaim_at_block_hash(reclaim_request_id, block_hash)
        return tuple(await self.contract.functions.reclaims(reclaim_request_id).call())

    async def finalized_block_hash(self):
        return (await self.w3.eth.get_block("finalized"))["hash"]

    async def head_parent_hash(self):
        """The newest block a pinned read can run on: the head's parent. `_call_at_block_hash` also runs the read on
        the pinned block's child, and no node has a child of the head yet ("header not found")."""
        return (await self.w3.eth.get_block("latest"))["parentHash"]

    async def _pinned_chain_id(self) -> int:
        if self.miner_account is None:
            raise CollateralTransactionError(
                "An Ethereum private key is required to send this transaction"
            )
        chain_id = CHAIN_IDS.get(self.network)
        if chain_id is None:
            raise CollateralConfigError(
                f"No EVM chain ID is known for BITTENSOR_NETWORK={self.network!r}; no transaction was signed"
            )
        rpc_chain_id = await self.w3.eth.chain_id
        if rpc_chain_id != chain_id:
            raise CollateralConfigError(
                f"The RPC reports EVM chain {rpc_chain_id}, not {chain_id} ({self.network}); "
                "no transaction was signed. Check SUBTENSOR_EVM_RPC_URL"
            )
        return chain_id

    async def settle_earlier_send(self) -> None:
        """Raise while this key has a recorded send whose outcome is not known, after reporting what it can.

        Run before any precheck that reads contract state: a mined reclaim or finalize whose answer was lost
        leaves nothing open, and that must be reported as its outcome, not as "nothing to do"."""
        chain_id = await self._pinned_chain_id()
        with self._send_lock():
            await self._settle_earlier_send(chain_id)

    async def _send(self, function_call) -> dict:
        chain_id = await self._pinned_chain_id()
        gas_price = await self.w3.eth.gas_price
        max_gas_price = AsyncWeb3.to_wei(self.max_gas_price_gwei, "gwei")
        if gas_price > max_gas_price:
            raise CollateralTransactionError(
                f"The RPC quoted a gas price of {AsyncWeb3.from_wei(gas_price, 'gwei')} gwei, above the "
                f"{self.max_gas_price_gwei} gwei ceiling (COLLATERAL_MAX_GAS_PRICE_GWEI); no transaction was sent"
            )
        with self._send_lock():
            return await self._send_locked(function_call, chain_id, gas_price)

    async def _send_locked(self, function_call, chain_id: int, gas_price: int) -> dict:
        await self._settle_earlier_send(chain_id)
        # An earlier send still pending blocks this one, and a mined one leaves a call the contract now rejects,
        # which the simulation catches unsent.
        nonce = await self.w3.eth.get_transaction_count(self.miner_address, "latest")
        pending_nonce = await self.w3.eth.get_transaction_count(self.miner_address, "pending")
        if pending_nonce > nonce:
            raise CollateralTransactionError(
                f"An earlier transaction from {self.miner_address} (nonce {nonce}) is still pending; "
                "no transaction was sent. Check it on the explorer and run this again once it is mined"
            )
        try:
            await function_call.call({"from": self.miner_address})
        except ContractLogicError as error:
            data = error.data if isinstance(error.data, str) else ""
            reason = self._custom_error_name(data) or "execution reverted"
            raise CollateralTransactionError(
                f"The contract rejects this call ({reason}); no transaction was sent"
            ) from error
        transaction = await function_call.build_transaction(
            {
                "from": self.miner_address,
                "nonce": nonce,
                "gas": GAS_LIMIT,
                "gasPrice": gas_price,
                "chainId": chain_id,
            }
        )
        signed = self.miner_account.sign_transaction(transaction)
        raw_transaction = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        signed_hash = signed.hash.hex()
        self._write_sent_record(
            chain_id,
            {"nonce": nonce, "hash": signed_hash, "raw": AsyncWeb3.to_hex(raw_transaction), "hashes": [signed_hash]},
        )
        try:
            answered_hash = await self.w3.eth.send_raw_transaction(raw_transaction)
        except Exception as error:
            answer = self._send_answer(error)
            raise CollateralOutcomeUnknownError(
                f"Transaction {signed_hash} may have been sent: the RPC answered with an error ({answer}), which "
                f"does not prove that no node took it, so its outcome is unknown. {RETRY_IS_SAFE}"
            ) from error
        # only the hash of the signed bytes names what was sent: a gateway can answer with any hash
        if not same_hash(answered_hash, signed_hash):
            logger.warning(
                "The RPC answered transaction %s with another hash; waiting for the signed one's receipt", signed_hash
            )
        logger.info("Sent transaction %s; waiting for its receipt", signed_hash)
        try:
            receipt = await self.w3.eth.wait_for_transaction_receipt(
                signed_hash, timeout=RECEIPT_TIMEOUT_SEC, poll_latency=RECEIPT_POLL_SEC
            )
        except Exception as error:
            # the class name only: a transport error's text can carry the RPC URL and its API key
            raise CollateralOutcomeUnknownError(
                f"Transaction {signed_hash} was sent but its receipt could not be read "
                f"({type(error).__name__}); its outcome is unknown. {RETRY_IS_SAFE}"
            ) from error
        self._check_receipt_hash(receipt, signed_hash)
        if not await self._finalized_on_chain(signed_hash, receipt):
            raise self._orphaned(signed_hash, receipt)
        if receipt["status"] == 0:
            self._clear_sent_record(chain_id, signed_hash)
            reason = await self._revert_reason(transaction, receipt["blockNumber"])
            message = f"Transaction {signed_hash} reverted"
            raise CollateralTransactionError(f"{message}: {reason}" if reason else message)
        self._clear_after_logging(
            chain_id, signed_hash, f"Transaction {signed_hash} succeeded in block {receipt['blockNumber']}"
            f"{self._started_request(receipt)}"
        )
        return receipt

    def _clear_after_logging(self, chain_id: int, record_hash: str, outcome: str) -> None:
        # The listing only searches the last RECLAIM_LOOKBACK_BLOCKS blocks, so this log line may be the only place
        # a started request's ID reaches the person. A stop before it leaves the record, and the next run reads the
        # receipt again and reports the ID.
        logger.info(outcome)
        self._clear_sent_record(chain_id, record_hash)

    async def _settle_earlier_send(self, chain_id: int) -> None:
        """Settle the recorded send whose outcome is not known. Raises while there is one: the run that learns
        its outcome reports it and sends nothing new."""
        record = self._read_sent_record(chain_id)
        if record is None:
            return
        found = await self._receipt_of_earlier_send(record)
        if found is None:
            await self._raise_if_nonce_used(record)
            found = await self._broadcast_again(record)
        self._report_settled(chain_id, *found)

    async def _raise_if_nonce_used(self, record: dict) -> None:
        nonce = await self.w3.eth.get_transaction_count(self.miner_address, "latest")
        if nonce > record["nonce"]:
            # the nonce is used, but that does not say by which transaction or with what outcome, so the record
            # stays until a receipt says so. The receipt may be late (a lagging RPC) or gone for good (an RPC
            # that prunes old receipts): another RPC, or the person after checking the explorer, settles it
            raise CollateralOutcomeUnknownError(
                f"Transaction {record['hash']}, sent earlier, has no receipt on this RPC and nonce {record['nonce']} "
                "is used, so its outcome is unknown; no transaction was sent. Run this again with "
                "SUBTENSOR_EVM_RPC_URL set to an RPC that serves its receipt, or look the transaction up on the "
                f"explorer and, once you know its outcome, delete {self.sent_record_path} and run this again "
                "if it still needs doing"
            )

    def _report_settled(self, chain_id: int, tx_hash: str, receipt, replaced: bool = False) -> None:
        message = self._settled_message(tx_hash, receipt, replaced)
        self._clear_after_logging(chain_id, self._read_sent_record(chain_id)["hash"], message)
        raise CollateralTransactionError(message)

    def _started_request(self, receipt) -> str:
        if receipt["status"] != 1:
            return ""
        started = self.contract.events.ReclaimProcessStarted().process_receipt(receipt, errors=DISCARD)
        return f"; it started reclaim request {started[0]['args']['reclaimRequestId']}" if started else ""

    def _settled_message(self, tx_hash: str, receipt, replaced: bool) -> str:
        outcome = "succeeded" if receipt["status"] == 1 else "reverted"
        request = self._started_request(receipt)
        sent = "the replacement" if replaced else "sent earlier"
        tail = (
            "Run the reclaim or finalize again if it still needs doing"
            if replaced
            else "no new transaction was sent. Run this again if it still needs doing"
        )
        return f"Transaction {tx_hash}, {sent}, {outcome} in block {receipt['blockNumber']}{request}; {tail}"

    async def _receipt_of_earlier_send(self, record: dict):
        """(hash, receipt) of whichever signed transaction of the record has a receipt, or None."""
        for tx_hash in record_hashes(record):
            try:
                receipt = await self.w3.eth.get_transaction_receipt(tx_hash)
            except TransactionNotFound:
                continue
            except Exception as error:
                raise CollateralOutcomeUnknownError(
                    f"The receipt of transaction {tx_hash}, sent earlier, could not be read "
                    f"({type(error).__name__}); no transaction was sent. {RETRY_IS_SAFE}"
                ) from error
            if receipt is None:
                continue
            self._check_receipt_hash(receipt, tx_hash)
            # a receipt from a block a reorganization dropped says nothing: the transaction may be back in a pool
            if not await self._finalized_on_chain(tx_hash, receipt):
                continue
            return tx_hash, receipt
        return None

    @staticmethod
    def _check_receipt_hash(receipt, tx_hash: str) -> None:
        if not same_hash(receipt.get("transactionHash"), tx_hash):
            raise CollateralOutcomeUnknownError(
                f"The RPC answered the receipt of transaction {tx_hash} with another transaction's receipt, so its "
                f"outcome is unknown; no transaction was sent. {RETRY_IS_SAFE}"
            )

    async def _finalized_on_chain(self, tx_hash: str, receipt) -> bool:
        """Whether the receipt's block is an ancestor of the finalized block. Waits for finality; raises while it
        has not come, so no record is cleared on a receipt a reorganization can still drop."""
        number = receipt["blockNumber"]
        deadline = asyncio.get_running_loop().time() + RECEIPT_TIMEOUT_SEC
        while True:
            try:
                (finalized,) = await self._read_together(("eth_getBlockByNumber", ["finalized", False]))
                chain = (
                    await self._chain_down_to(finalized, number)
                    if finalized is not None and block_number(finalized) >= number
                    else None
                )
            except RpcReadError as error:
                raise CollateralOutcomeUnknownError(
                    f"The finalized block and block {number} of transaction {tx_hash} could not be read ({error}), "
                    f"so its outcome is unknown; no transaction was sent. {RETRY_IS_SAFE}. If the RPC keeps "
                    "refusing, set SUBTENSOR_EVM_RPC_URL to another RPC, or look the transaction up on the explorer"
                ) from error
            if chain is not None:
                break
            if finalized is not None and block_number(finalized) >= number:
                raise CollateralOutcomeUnknownError(
                    f"The RPC answered the blocks from the finalized block down to block {number} of transaction "
                    f"{tx_hash} from more than one chain, so its outcome is unknown; no transaction was sent. "
                    f"{RETRY_IS_SAFE}"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise CollateralOutcomeUnknownError(
                    f"Transaction {tx_hash} is in block {number}, which is not finalized yet, so its outcome is "
                    f"not settled; no transaction was sent. {RETRY_IS_SAFE}"
                )
            await asyncio.sleep(RECEIPT_POLL_SEC)
        block = chain.get(number)
        if block is None:
            raise CollateralOutcomeUnknownError(
                f"Transaction {tx_hash} is in block {number}, which this RPC no longer keeps, so its outcome cannot "
                "be checked against the finalized chain; no transaction was sent. Run this again with "
                "SUBTENSOR_EVM_RPC_URL set to an RPC that keeps that block, or look the transaction up on the "
                f"explorer and, once you know its outcome, delete {self.sent_record_path} and run this again "
                "if it still needs doing"
            )
        return same_hash(block["hash"], receipt["blockHash"])

    async def _chain_down_to(self, finalized, lowest: int) -> dict[int, dict] | None:
        """The blocks from `lowest` up to the finalized block, read by number and kept only when each one's hash is
        the parent hash of the block above. A gateway can send each item of a batch to another backend, so a read
        by number can be another fork's block; one linked by parent hashes to the finalized block is its ancestor
        whoever answered. Stops below a block the RPC does not answer; None when the hashes do not link."""
        top = block_number(finalized)
        chain = {top: finalized}
        parent = finalized["parentHash"]
        for high in range(top - 1, lowest - 1, -CHAIN_READ_BATCH):
            numbers = range(high, max(high - CHAIN_READ_BATCH, lowest - 1), -1)
            blocks = await self._read_together(*(("eth_getBlockByNumber", [hex(n), False]) for n in numbers))
            for number, block in zip(numbers, blocks):
                if block is None:
                    return chain
                if not same_hash(block["hash"], parent) or block_number(block) != number:
                    return None
                chain[number] = block
                parent = block["parentHash"]
        return chain

    @staticmethod
    def _orphaned(tx_hash: str, receipt) -> CollateralOutcomeUnknownError:
        return CollateralOutcomeUnknownError(
            f"The receipt of transaction {tx_hash} names block {receipt['blockNumber']} {receipt['blockHash']}, "
            f"which is not the finalized block at that number, so its outcome is unknown; no transaction was sent. "
            f"{RETRY_IS_SAFE}"
        )

    async def _wait_for_earlier_send(self, record: dict):
        hashes = record_hashes(record)
        if len(hashes) == 1:
            receipt = await self.w3.eth.wait_for_transaction_receipt(
                hashes[0], timeout=RECEIPT_TIMEOUT_SEC, poll_latency=RECEIPT_POLL_SEC
            )
            self._check_receipt_hash(receipt, hashes[0])
            if not await self._finalized_on_chain(hashes[0], receipt):
                raise self._orphaned(hashes[0], receipt)
            return hashes[0], receipt
        deadline = asyncio.get_running_loop().time() + RECEIPT_TIMEOUT_SEC
        while True:
            found = await self._receipt_of_earlier_send(record)
            if found is not None:
                return found
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError()
            await asyncio.sleep(RECEIPT_POLL_SEC)

    async def _broadcast_again(self, record: dict):
        """Broadcast an unmined earlier send again as the same bytes (a mempool may have dropped it) and wait
        for a receipt of it or of a transaction it replaced. Its nonce is unused, so only one of them can be mined."""
        tx_hash = record["hash"]
        try:
            await self.w3.eth.send_raw_transaction(record["raw"])
        except Exception as error:
            answer = self._send_answer(error)
            if answer != "already known":
                raise CollateralOutcomeUnknownError(
                    f"Transaction {tx_hash}, sent earlier, is not mined and broadcasting it again failed "
                    f"({answer}); no transaction was sent. {self._if_it_stays_unmined(record)}"
                ) from error
        logger.info("Broadcast transaction %s, sent earlier, again; waiting for its receipt", tx_hash)
        try:
            return await self._wait_for_earlier_send(record)
        except CollateralOutcomeUnknownError:
            raise
        except Exception as error:
            raise CollateralOutcomeUnknownError(
                f"Transaction {tx_hash}, sent earlier, is not mined yet; it was broadcast again and no new "
                f"transaction was sent ({type(error).__name__}). {RETRY_IS_SAFE}. "
                f"{self._if_it_stays_unmined(record)}"
            ) from error

    def _if_it_stays_unmined(self, record: dict) -> str:
        # no receipt on this RPC does not prove it can't be mined, so the record is never dropped for that
        return (
            "If it stays unmined (a gas price the chain no longer takes), run `replace-collateral-transaction`: it "
            f"signs a replacement at the same nonce {record['nonce']} with a higher gas price, so only one of the "
            "two can be mined"
        )

    async def replace_earlier_send(self) -> str:
        """Replace this key's unmined recorded send with the same call at the same nonce and a higher gas price.

        The replacement is recorded next to the transaction it replaces before it is broadcast, and a receipt of
        either settles the record; no other nonce is signed meanwhile. Returns the outcome when a transaction of
        the record succeeded; raises with it otherwise, like a settle."""
        chain_id = await self._pinned_chain_id()
        gas_quote = await self.w3.eth.gas_price
        with self._send_lock():
            record = self._read_sent_record(chain_id)
            if record is None:
                raise CollateralTransactionError(
                    "No collateral transaction from this key is waiting for its outcome; nothing was replaced"
                )
            earlier = self._verified_earlier_send(record, chain_id)
            found = await self._receipt_of_earlier_send(record)
            if found is not None:
                self._report_settled(chain_id, *found)
            await self._raise_if_nonce_used(record)
            bump, base = REPLACEMENT_PRICE_BUMP
            gas_price = max(gas_quote, -(-earlier["gasPrice"] * bump // base))
            if gas_price > AsyncWeb3.to_wei(self.max_gas_price_gwei, "gwei"):
                raise CollateralTransactionError(
                    f"A replacement needs a gas price of {AsyncWeb3.from_wei(gas_price, 'gwei')} gwei, above the "
                    f"{self.max_gas_price_gwei} gwei ceiling (COLLATERAL_MAX_GAS_PRICE_GWEI); nothing was replaced"
                )
            signed = self.miner_account.sign_transaction(
                {**{k: earlier[k] for k in ("to", "value", "data", "gas")}, "nonce": record["nonce"],
                 "gasPrice": gas_price, "chainId": chain_id}
            )
            raw_transaction = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            signed_hash = signed.hash.hex()
            record = {
                "nonce": record["nonce"],
                "hash": signed_hash,
                "raw": AsyncWeb3.to_hex(raw_transaction),
                "hashes": [*record_hashes(record), signed_hash],
            }
            self._write_sent_record(chain_id, record)
            logger.info(
                "Replacing the transaction at nonce %s with %s at %s gwei",
                record["nonce"], signed_hash, AsyncWeb3.from_wei(gas_price, "gwei"),
            )
            tx_hash, receipt = await self._broadcast_again(record)
            if receipt["status"] != 1:
                self._report_settled(chain_id, tx_hash, receipt, replaced=same_hash(tx_hash, signed_hash))
            message = self._settled_message(tx_hash, receipt, replaced=same_hash(tx_hash, signed_hash))
            self._clear_after_logging(chain_id, record["hash"], message)
            return message

    def _verified_earlier_send(self, record: dict, chain_id: int) -> dict:
        """The recorded transaction's fields, once its bytes prove this key signed them for a collateral call.

        The record sits in a directory the miner service can write, so a replacement never copies a field from it
        unchecked: the bytes must hash to the recorded hash, recover to this key's address, carry the recorded
        nonce and this chain, and call a function this client sends on a configured collateral contract with no
        value."""

        def refuse(why: str) -> CollateralTransactionError:
            return CollateralTransactionError(
                f"The recorded transaction in {self.sent_record_path} is not a collateral call this key signed "
                f"({why}); nothing was signed or broadcast. Check that file before running this again"
            )

        try:
            raw = bytes.fromhex(str(record.get("raw", "")).removeprefix("0x"))
            fields = rlp.decode(raw)
            earlier = decode_legacy_transaction(record["raw"])
            signer = Account.recover_transaction(raw)
        except Exception as error:
            raise refuse(f"its bytes do not decode as a signed legacy transaction ({type(error).__name__})") from error
        if len(fields) != 9:
            raise refuse("its bytes do not decode as a signed legacy transaction")
        if not same_hash(AsyncWeb3.keccak(raw), record.get("hash")):
            raise refuse("its bytes do not hash to the recorded hash")
        if signer != self.miner_address:
            raise refuse("another key signed it")
        if earlier["nonce"] != record.get("nonce"):
            raise refuse("its nonce is not the recorded one")
        v = int.from_bytes(fields[6], "big")
        if v < 35 or (v - 35) // 2 != chain_id:
            raise refuse(f"it is not signed for chain {chain_id}")
        if earlier["to"] not in self.replaceable_contract_addresses:
            raise refuse("it is not sent to a configured collateral contract")
        if earlier["value"] != 0:
            raise refuse("it sends value")
        if earlier["gas"] > GAS_LIMIT:
            raise refuse("its gas limit is above the one this client signs")
        sent = {AsyncWeb3.keccak(text=signature)[:4].hex().removeprefix("0x") for signature in SENT_FUNCTIONS}
        if earlier["data"].removeprefix("0x")[:8].lower() not in sent:
            raise refuse("it calls a function this client does not send")
        return earlier

    def _sent_record_key(self, chain_id: int) -> str:
        return f"{chain_id}:{self.miner_address}"

    @contextlib.contextmanager
    def _send_lock(self):
        """Hold the record file's lock from settling an earlier send until this one's record is cleared or
        left, so two runs never both find no record and each send on a new nonce."""
        lock_path = self.sent_record_path.with_name(self.sent_record_path.name + ".lock")
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(lock_path, "a")
        except OSError as error:
            raise CollateralTransactionError(
                f"The lock of the record of earlier sends ({lock_path}) could not be opened "
                f"({type(error).__name__}); no transaction was sent. Set COLLATERAL_SENT_RECORD to a writable path"
            ) from error
        with handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CollateralTransactionError(
                    "Another run is sending a collateral transaction from this machine; no transaction was sent. "
                    "Run this again once it has finished"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _read_sent_record(self, chain_id: int) -> dict | None:
        try:
            records = json.loads(self.sent_record_path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            raise CollateralTransactionError(
                f"The record of earlier sends ({self.sent_record_path}) could not be read ({type(error).__name__}); "
                "no transaction was sent"
            ) from error
        return records.get(self._sent_record_key(chain_id))

    def _write_sent_record(self, chain_id: int, record: dict) -> None:
        try:
            self._replace_sent_records(lambda records: {**records, self._sent_record_key(chain_id): record})
        except (OSError, ValueError) as error:
            raise CollateralTransactionError(
                f"The record of earlier sends ({self.sent_record_path}) could not be written ({type(error).__name__}); "
                "no transaction was sent. Set COLLATERAL_SENT_RECORD to a writable path"
            ) from error

    def _clear_sent_record(self, chain_id: int, tx_hash: str) -> None:
        """Drop the record, only while it is still the one of tx_hash: a newer send's record stays."""
        key = self._sent_record_key(chain_id)

        def without(records: dict) -> dict:
            if (records.get(key) or {}).get("hash") != tx_hash:
                return records
            return {k: v for k, v in records.items() if k != key}

        try:
            self._replace_sent_records(without)
        except (OSError, ValueError) as error:
            # the outcome is already known; the next run reads it again and reports it
            logger.warning(
                "Could not clear the record of earlier sends (%s): %s", self.sent_record_path, type(error).__name__
            )

    def _replace_sent_records(self, change) -> None:
        path = self.sent_record_path
        records = json.loads(path.read_text()) if path.exists() else {}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with open(temporary, "w") as handle:
            json.dump(change(records), handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        # the rename reaches the disk only once the directory is synced; before that a crash can lose the record
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @staticmethod
    def _send_answer(error: Exception) -> str:
        """Local text for a send's error answer; the RPC's own text is only matched, never passed on."""
        if not isinstance(error, ValueError) or isinstance(error, json.JSONDecodeError):
            return type(error).__name__
        detail = error.args[0] if error.args else None
        if not isinstance(detail, dict):
            return type(error).__name__
        message = str(detail.get("message", "")).lower().strip()
        if message in ALREADY_KNOWN:
            return "already known"
        for phrase, answer in SEND_REFUSALS.items():
            if phrase in message:
                return answer
        return "an error answer that is not a known refusal"

    async def _revert_reason(self, transaction: dict, block_number: int) -> str | None:
        """Replay a reverted transaction as an eth_call at its block and name the revert."""
        call = {
            key: transaction[key] for key in ("from", "to", "data", "value") if key in transaction
        }
        try:
            await self.w3.eth.call(call, block_identifier=block_number)
        except ContractLogicError as error:
            data = (
                error.data
                if isinstance(error.data, str)
                else str(error.args[0] if error.args else "")
            )
            return self._custom_error_name(data) or "execution reverted"
        except Exception as error:
            # the class name only: a transport error's text can carry the RPC URL and its API key
            logger.warning(
                "Could not replay the reverted transaction at block %s to read its revert reason: %s",
                block_number,
                type(error).__name__,
            )
            return None
        return None

    def _custom_error_name(self, data: str) -> str | None:
        selector = data.removeprefix("0x")[:8].lower()
        if len(selector) != 8:
            return None
        for entry in self.contract.abi:
            if entry.get("type") != "error":
                continue
            signature = f"{entry['name']}({','.join(i['type'] for i in entry['inputs'])})"
            if AsyncWeb3.keccak(text=signature)[:4].hex().removeprefix("0x") == selector:
                return entry["name"]
        return None

    async def reclaim_collateral(self, executor_uuid: str, url: str = "Manual reclaim"):
        """Start a reclaim of the executor's full collateral; returns its ReclaimProcessStarted."""
        receipt = await self._send(
            self.contract.functions.reclaimCollateral(
                executor_uuid_bytes(executor_uuid), url, bytes(16)
            )
        )
        return self.contract.events.ReclaimProcessStarted().process_receipt(receipt)[0]

    async def finalize_reclaim(self, reclaim_request_id: int):
        """Pay out a reclaim after its deny window; returns its Reclaimed event or None."""
        await self.settle_earlier_send()
        reclaim = await self.get_reclaim_request(reclaim_request_id)
        if reclaim[2] == 0:
            raise CollateralTransactionError(
                f"No open reclaim request {reclaim_request_id} on this contract "
                "(never opened, or already finalized or denied)"
            )
        receipt = await self._send(self.contract.functions.finalizeReclaim(reclaim_request_id))
        events = self.contract.events.Reclaimed().process_receipt(receipt)
        return events[0] if events else None

    async def get_reclaim_events(self) -> list[ReclaimRequest]:
        """Open reclaim requests started in the last RECLAIM_LOOKBACK_BLOCKS finalized blocks.

        A load-balanced RPC can answer any read from a backend on another fork or one that lags behind, so the
        blocks are kept only when they link by parent hashes to the finalized block, and their logs are read from
        the receipts of their transactions (see `_reclaim_events_at`). Each request's state is read at the finalized block's hash and must
        match its log. A request started after the finalized block is listed once its block is final."""
        for _ in range(RECLAIM_LIST_ATTEMPTS):
            finalized = await self.w3.eth.get_block("finalized")
            requests = await self._reclaim_events_at(finalized)
            if requests is not None:
                return requests
        raise CollateralTransactionError(
            f"The RPC answered the open reclaim requests from blocks that are not on the finalized chain, or with "
            f"only part of a block's logs, {RECLAIM_LIST_ATTEMPTS} times; run this again"
        )

    async def _reclaim_at_block_hash(self, reclaim_request_id: int, block_hash) -> tuple:
        return await self._call_at_block_hash(self.contract.functions.reclaims(reclaim_request_id), block_hash)

    async def _call_at_block_hash(self, function, block_hash):
        """A view function's result at a block named by its hash. A contract function's call(block_identifier=<hash>)
        looks the hash up and sends the call by number; here the header is read by hash first.

        Frontier ignores requireCanonical and answers a hash it does not know from its pending state, which can
        carry the block's number, parent and timestamp; a call pinned by number can reach a sibling block with the
        same three. So it runs inside pinned_read_code twice: pinned by hash, where the header words must match the
        header, and on the block's child, where BLOCKHASH(NUMBER - 1) must be the block's hash, which only a chain
        holding the block answers. The child's run sees the child's state, so a third run pinned by the block's
        number, which a backend without the block refuses, must agree as well. The answers must agree; no child yet
        fails closed. Both runs are plain calls
        (pinned_call), so the creator whitelist does not apply. Every output of the contract's view functions is a
        static type."""
        hash_hex = AsyncWeb3.to_hex(block_hash)
        outputs = [output["type"] for output in function.abi["outputs"]]
        size = 32 * len(outputs)
        code = pinned_read_code(self.contract_address, [(bytes.fromhex(function._encode_transaction_data()[2:]), size)])
        (block,) = await self._read_together(("eth_getBlockByHash", [hash_hex, False]))
        if not isinstance(block, dict) or block.get("hash") != hash_hex:
            raise RpcReadError("the RPC does not have the block the read is pinned to")
        by_hash, by_number, at_child = await self._read_together(
            ("eth_call", pinned_call(code, {"blockHash": hash_hex, "requireCanonical": True})),
            ("eth_call", pinned_call(code, {"blockNumber": hex(block_number(block))})),
            ("eth_call", pinned_call(code, {"blockNumber": hex(block_number(block) + 1)})),
        )
        [result] = pinned_outputs(by_hash, block, [size])
        if not pinned_outputs(by_number, block, [size]) == child_outputs(at_child, block, [size]) == [result]:
            raise RpcReadError("the reads at the block and at its child disagree")
        decoded = self.w3.codec.decode(outputs, result)
        return decoded[0] if len(outputs) == 1 else tuple(decoded)

    async def _reclaim_events_at(self, finalized) -> list[ReclaimRequest] | None:
        """The open requests at the finalized block, or None when a block or log is not on its chain, or a block's
        logs are answered only in part.

        A range eth_getLogs can reach a lagging backend whose empty answer looks like no request. So every log of
        each block on the finalized chain whose logs bloom may hold this contract's ReclaimProcessStarted is read
        from the receipts of all of that block's transactions, and kept only when every receipt names that block and
        their logs make up the header's bloom exactly."""
        top = finalized["number"]
        lowest = max(top - RECLAIM_LOOKBACK_BLOCKS, 0)
        event = self.contract.events.ReclaimProcessStarted()
        address = bytes.fromhex(self.contract_address.removeprefix("0x"))
        topic = event_abi_to_log_topic(event.abi)
        try:
            chain = await self._chain_down_to(finalized, lowest)
            if chain is None or (min(chain) > lowest and top - min(chain) < KEPT_BLOCKS_MIN):
                return None
            candidates = [block for block in chain.values() if bloom_may_hold(block.get("logsBloom"), address, topic)]
            # Blooms OR the bits of every address and topic, so an answer that makes up the header's bloom can still
            # leave out a log whose bits other logs set. Each candidate block's logs come from the receipt of every
            # transaction the block names, and a missing receipt makes the list indeterminate.
            receipts_of = []
            for block in candidates:
                transactions = block.get("transactions")
                if transactions is None:
                    return None
                receipts_of.extend((block, hash_text(tx)) for tx in transactions)
            logs_of = {hash_text(block["hash"]): [] for block in candidates}
            for start in range(0, len(receipts_of), CHAIN_READ_BATCH):
                batch = receipts_of[start : start + CHAIN_READ_BATCH]
                answers = await self._read_together(
                    *(("eth_getTransactionReceipt", ["0x" + tx]) for _, tx in batch)
                )
                for (block, tx), receipt in zip(batch, answers):
                    if (
                        not isinstance(receipt, dict)
                        or not same_hash(receipt.get("transactionHash"), tx)
                        or not same_hash(receipt.get("blockHash"), block["hash"])
                    ):
                        return None
                    logs_of[hash_text(block["hash"])].extend(receipt.get("logs") or [])
            raw_logs = []
            for block in candidates:
                answer = logs_of[hash_text(block["hash"])]
                if any(log.get("removed") or not same_hash(log["blockHash"], block["hash"]) for log in answer):
                    return None
                if logs_bloom(answer) != hash_text(block.get("logsBloom")):
                    return None
                raw_logs.extend(
                    log for log in answer
                    if same_hash(log["address"], address) and log["topics"] and same_hash(log["topics"][0], topic)
                )
        except RpcReadError as error:
            raise CollateralTransactionError(
                f"The RPC did not answer the reclaim request list ({error}); run this again, or set "
                "SUBTENSOR_EVM_RPC_URL to another RPC"
            ) from error
        if min(chain) > lowest:
            # a pruning RPC (the default finney one keeps about 256 blocks) answers the range from what it keeps
            logger.warning(
                "The RPC %s no longer keeps block %s, so a reclaim request started before the blocks it keeps is "
                "not listed. The reclaim command that started it printed its ID ('it started reclaim request <id>'); "
                "set SUBTENSOR_EVM_RPC_URL to an RPC that keeps older blocks to list it",
                rpc_origin(self.rpc_url), lowest,
            )
        logs = [event.process_log(log_entry_formatter(log)) for log in raw_logs]
        requests = []
        for log in sorted(logs, key=lambda log: (log["blockNumber"], log["logIndex"])):
            args = log["args"]
            reclaim_request_id = args["reclaimRequestId"]
            executor_id, miner, amount, expiration_time = (
                await self._reclaim_at_block_hash(reclaim_request_id, finalized["hash"])
            )[:4]
            if amount == 0:
                continue
            if (executor_id, AsyncWeb3.to_checksum_address(miner), amount, expiration_time) != (
                args["executorId"], AsyncWeb3.to_checksum_address(args["miner"]), args["amount"], args["expirationTime"]
            ):
                return None
            requests.append(
                ReclaimRequest(
                    reclaim_request_id=reclaim_request_id,
                    executor_uuid=str(UUID(bytes=executor_id)),
                    miner=miner,
                    amount=float(AsyncWeb3.from_wei(amount, "ether")),
                    expiration_time=datetime.fromtimestamp(expiration_time, UTC).strftime(
                        DATETIME_FORMAT
                    ),
                    url=log["args"]["url"],
                    url_content_md5_checksum=log["args"]["urlContentMd5Checksum"].hex(),
                    block_number=log["blockNumber"],
                )
            )
        return requests
