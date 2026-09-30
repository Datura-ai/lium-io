"""Read and reclaim provider collateral on the Lium collateral contract (Bittensor EVM).

Collateral is optional for providers; this module keeps the withdrawal path for executors
that still hold a deposit. It covers the contract calls the miner CLI makes: balance and
collateral reads, start a reclaim, list open reclaims, finalize a reclaim.
"""

import hashlib
import json
import logging
import os
import pathlib
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit
from uuid import UUID

from bittensor_wallet import Keypair
from eth_account import Account
from web3 import AsyncHTTPProvider, AsyncWeb3
from web3.exceptions import ContractLogicError, TransactionNotFound

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
SENT_RECORD_PATH = pathlib.Path(
    os.environ.get("COLLATERAL_SENT_RECORD", "~/.bittensor/wallets/.lium-collateral-sent.json")
).expanduser()
RETRY_IS_SAFE = "Run this again: it reads this transaction's outcome first and sends nothing new until it is known"

GAS_LIMIT = 200_000
# The RPC quotes the gas price; above this ceiling nothing is signed, so a faulty or hostile RPC
# cannot spend the address balance on fees (at GAS_LIMIT, 100 gwei caps a transaction at 0.02 TAO).
DEFAULT_MAX_GAS_PRICE_GWEI = 100
RECLAIM_LOOKBACK_BLOCKS = 1000
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"

SS58_FORMAT = 42


class CollateralTransactionError(Exception):
    pass


class CollateralConfigError(Exception):
    pass


class CollateralOutcomeUnknownError(CollateralTransactionError):
    """A transaction was broadcast but its receipt was never read: it may still be mined."""


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
    try:
        raw = UUID(executor_uuid).bytes
    except ValueError:
        raw = bytes.fromhex(executor_uuid.removeprefix("0x"))
    return raw[:16].ljust(16, b"\0")


class CollateralClient:
    def __init__(
        self,
        network: str,
        contract_address: str,
        rpc_url: str | None = None,
        miner_key: str | None = None,
        max_gas_price_gwei: float = DEFAULT_MAX_GAS_PRICE_GWEI,
    ):
        self.network = network
        self.max_gas_price_gwei = max_gas_price_gwei
        self.rpc_url = rpc_url or RPC_URLS.get(network)
        self.contract_address = AsyncWeb3.to_checksum_address(contract_address)
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
            self._w3 = AsyncWeb3(AsyncHTTPProvider(self.rpc_url))
        return self._w3

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

    async def get_executor_collateral(self, executor_uuid: str):
        amount = await self.contract.functions.collaterals(
            executor_uuid_bytes(executor_uuid)
        ).call()
        return AsyncWeb3.from_wei(amount, "ether")

    async def get_reclaim_request(self, reclaim_request_id: int) -> tuple:
        """(executorId, miner, amount in wei, denyTimeout) of a reclaim request; amount 0 once it is closed."""
        return tuple(await self.contract.functions.reclaims(reclaim_request_id).call())

    async def _send(self, function_call) -> dict:
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
        gas_price = await self.w3.eth.gas_price
        max_gas_price = AsyncWeb3.to_wei(self.max_gas_price_gwei, "gwei")
        if gas_price > max_gas_price:
            raise CollateralTransactionError(
                f"The RPC quoted a gas price of {AsyncWeb3.from_wei(gas_price, 'gwei')} gwei, above the "
                f"{self.max_gas_price_gwei} gwei ceiling (COLLATERAL_MAX_GAS_PRICE_GWEI); no transaction was sent"
            )
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
            reason = self._custom_error_name(data) or error.message or "execution reverted"
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
        self._write_sent_record(
            chain_id, {"nonce": nonce, "hash": signed.hash.hex(), "raw": AsyncWeb3.to_hex(raw_transaction)}
        )
        try:
            tx_hash = await self.w3.eth.send_raw_transaction(raw_transaction)
        except Exception as error:
            message = self._rpc_error_message(error)
            if message is not None and "known" not in message.lower():
                self._write_sent_record(chain_id, None)
                raise CollateralTransactionError(
                    f"The RPC refused the transaction ({message}); no transaction was sent"
                ) from error
            # "already known", a reply that is not a JSON-RPC error (malformed JSON) or no reply at all: the node
            # may hold the transaction, so it may be mined
            answer = message or type(error).__name__
            raise CollateralOutcomeUnknownError(
                f"Transaction {signed.hash.hex()} may have been sent, but the RPC's answer was not a clear "
                f"refusal ({answer}); its outcome is unknown. {RETRY_IS_SAFE}"
            ) from error
        logger.info("Sent transaction %s; waiting for its receipt", tx_hash.hex())
        try:
            receipt = await self.w3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=300, poll_latency=2
            )
        except Exception as error:
            # the class name only: a transport error's text can carry the RPC URL and its API key
            raise CollateralOutcomeUnknownError(
                f"Transaction {tx_hash.hex()} was sent but its receipt could not be read "
                f"({type(error).__name__}); its outcome is unknown. {RETRY_IS_SAFE}"
            ) from error
        self._write_sent_record(chain_id, None)
        if receipt["status"] == 0:
            reason = await self._revert_reason(transaction, receipt["blockNumber"])
            message = f"Transaction {tx_hash.hex()} reverted"
            raise CollateralTransactionError(f"{message}: {reason}" if reason else message)
        return receipt

    async def _settle_earlier_send(self, chain_id: int) -> None:
        """Settle the recorded send whose outcome is not known. Raises while there is one: the run that learns
        its outcome reports it and sends nothing new."""
        record = self._read_sent_record(chain_id)
        if record is None:
            return
        tx_hash = record["hash"]
        receipt = await self._receipt_of_earlier_send(tx_hash)
        if receipt is None:
            nonce = await self.w3.eth.get_transaction_count(self.miner_address, "latest")
            if nonce > record["nonce"]:
                # a lagging RPC can show the nonce used before it serves the receipt, so this is not proof that
                # the transaction was dropped
                raise CollateralOutcomeUnknownError(
                    f"Transaction {tx_hash}, sent earlier, has no receipt yet but nonce {record['nonce']} is used; "
                    f"no transaction was sent. Run this again in a few minutes. If the explorer shows another "
                    f"transaction took nonce {record['nonce']}, delete {SENT_RECORD_PATH} and run this again"
                )
            receipt = await self._broadcast_again(record)
        self._write_sent_record(chain_id, None)
        outcome = "succeeded" if receipt["status"] == 1 else "reverted"
        raise CollateralTransactionError(
            f"Transaction {tx_hash}, sent earlier, {outcome} in block {receipt['blockNumber']}; "
            "no new transaction was sent. Run this again if it still needs doing"
        )

    async def _receipt_of_earlier_send(self, tx_hash: str):
        try:
            return await self.w3.eth.get_transaction_receipt(tx_hash)
        except TransactionNotFound:
            return None
        except Exception as error:
            raise CollateralOutcomeUnknownError(
                f"The receipt of transaction {tx_hash}, sent earlier, could not be read ({type(error).__name__}); "
                f"no transaction was sent. {RETRY_IS_SAFE}"
            ) from error

    async def _broadcast_again(self, record: dict):
        """Broadcast an unmined earlier send again as the same bytes (a mempool may have dropped it) and wait
        for its receipt. Its nonce is unused, so it and anything else on that nonce can't both be mined."""
        tx_hash = record["hash"]
        refusal = None
        try:
            await self.w3.eth.send_raw_transaction(record["raw"])
        except Exception as error:
            message = self._rpc_error_message(error)
            if message is not None and "known" not in message.lower():
                refusal = message
        if refusal is not None:
            raise CollateralOutcomeUnknownError(
                f"Transaction {tx_hash}, sent earlier, is not mined and the RPC refused to broadcast it again "
                f"({refusal}); no transaction was sent. If it stays unmined, delete {SENT_RECORD_PATH} and run "
                f"this again: the next transaction reuses nonce {record['nonce']}, so only one of the two can "
                "be mined"
            )
        logger.info("Broadcast transaction %s, sent earlier, again; waiting for its receipt", tx_hash)
        try:
            return await self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300, poll_latency=2)
        except Exception as error:
            raise CollateralOutcomeUnknownError(
                f"Transaction {tx_hash}, sent earlier, is not mined yet; it was broadcast again and no new "
                f"transaction was sent ({type(error).__name__}). {RETRY_IS_SAFE}"
            ) from error

    def _sent_record_key(self, chain_id: int) -> str:
        return f"{chain_id}:{self.miner_address}"

    def _read_sent_record(self, chain_id: int) -> dict | None:
        try:
            records = json.loads(SENT_RECORD_PATH.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            raise CollateralTransactionError(
                f"The record of earlier sends ({SENT_RECORD_PATH}) could not be read ({type(error).__name__}); "
                "no transaction was sent"
            ) from error
        return records.get(self._sent_record_key(chain_id))

    def _write_sent_record(self, chain_id: int, record: dict | None) -> None:
        try:
            records = json.loads(SENT_RECORD_PATH.read_text()) if SENT_RECORD_PATH.exists() else {}
            if record is None:
                records.pop(self._sent_record_key(chain_id), None)
            else:
                records[self._sent_record_key(chain_id)] = record
            SENT_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = SENT_RECORD_PATH.with_suffix(".tmp")
            with open(temporary, "w") as handle:
                json.dump(records, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(SENT_RECORD_PATH)
        except (OSError, ValueError) as error:
            if record is None:
                # the outcome is already known; the next run reads it again and reports it
                logger.warning(
                    "Could not clear the record of earlier sends (%s): %s", SENT_RECORD_PATH, type(error).__name__
                )
                return
            raise CollateralTransactionError(
                f"The record of earlier sends ({SENT_RECORD_PATH}) could not be written ({type(error).__name__}); "
                "no transaction was sent"
            ) from error

    @staticmethod
    def _rpc_error_message(error: Exception) -> str | None:
        """The message of a JSON-RPC error answer; None for anything else (no answer, or one that isn't JSON)."""
        if not isinstance(error, ValueError) or isinstance(error, json.JSONDecodeError):
            return None
        detail = error.args[0] if error.args else None
        return str(detail.get("message", "error")) if isinstance(detail, dict) else None

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
            return self._custom_error_name(data) or error.message or data or None
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
        """Open reclaim requests started in the last RECLAIM_LOOKBACK_BLOCKS blocks."""
        latest_block = await self.w3.eth.block_number
        logs = await self.contract.events.ReclaimProcessStarted().get_logs(
            fromBlock=max(latest_block - RECLAIM_LOOKBACK_BLOCKS, 0), toBlock=latest_block
        )
        requests = []
        for log in logs:
            reclaim_request_id = log["args"]["reclaimRequestId"]
            executor_id, miner, amount, expiration_time = (
                await self.contract.functions.reclaims(reclaim_request_id).call()
            )[:4]
            if amount == 0:
                continue
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
