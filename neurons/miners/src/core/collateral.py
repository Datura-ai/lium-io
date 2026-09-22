"""Read and reclaim provider collateral on the Lium collateral contract (Bittensor EVM).

Collateral is optional for providers; this module keeps the withdrawal path for executors
that still hold a deposit. It covers the contract calls the miner CLI makes: balance and
collateral reads, start a reclaim, list open reclaims, finalize a reclaim.
"""

import hashlib
import json
import pathlib
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from eth_account import Account
from web3 import AsyncHTTPProvider, AsyncWeb3

ABI_PATH = pathlib.Path(__file__).with_name("collateral_abi.json")

RPC_URLS = {
    "local": "http://127.0.0.1:9944",
    "test": "https://test.finney.opentensor.ai",
    "finney": "https://lite.chain.opentensor.ai",
}

GAS_LIMIT = 200_000
RECLAIM_LOOKBACK_BLOCKS = 1000
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"

SS58_FORMAT = 42
_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_SS58_PREFIX = b"SS58PRE"


class CollateralTransactionError(Exception):
    pass


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


def _base58_encode(data: bytes) -> str:
    number = int.from_bytes(data, "big")
    encoded = ""
    while number:
        number, remainder = divmod(number, 58)
        encoded = _BASE58_ALPHABET[remainder] + encoded
    leading_zeros = len(data) - len(data.lstrip(b"\0"))
    return "1" * leading_zeros + encoded


def h160_to_ss58(h160_address: str) -> str:
    """The SS58 account (generic prefix 42) that mirrors an EVM (H160) address on Bittensor.

    Same mapping as opentensor/evm-bittensor examples/address-mapping.js: blake2b-256 of
    b"evm:" + the 20 address bytes, SS58-encoded.
    """
    address_bytes = bytes.fromhex(h160_address.removeprefix("0x"))
    public_key = hashlib.blake2b(b"evm:" + address_bytes, digest_size=32).digest()
    payload = bytes([SS58_FORMAT]) + public_key
    checksum = hashlib.blake2b(_SS58_PREFIX + payload).digest()[:2]
    return _base58_encode(payload + checksum)


def executor_uuid_bytes(executor_uuid: str | UUID) -> bytes:
    if isinstance(executor_uuid, UUID):
        return executor_uuid.bytes
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
    ):
        self.w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url or RPC_URLS[network]))
        self.contract_address = AsyncWeb3.to_checksum_address(contract_address)
        self.contract = self.w3.eth.contract(
            address=self.contract_address, abi=json.loads(ABI_PATH.read_text())
        )
        self.miner_account = Account.from_key(miner_key) if miner_key else None
        self.miner_address = self.miner_account.address if self.miner_account else None

    async def get_balance(self, address: str):
        balance = await self.w3.eth.get_balance(AsyncWeb3.to_checksum_address(address))
        return self.w3.from_wei(balance, "ether")

    async def get_executor_collateral(self, executor_uuid: str | UUID):
        amount = await self.contract.functions.collaterals(
            executor_uuid_bytes(executor_uuid)
        ).call()
        return self.w3.from_wei(amount, "ether")

    async def _send(self, function_call) -> dict:
        if self.miner_account is None:
            raise CollateralTransactionError(
                "An Ethereum private key is required to send this transaction"
            )
        transaction = await function_call.build_transaction(
            {
                "from": self.miner_address,
                "nonce": await self.w3.eth.get_transaction_count(self.miner_address),
                "gas": GAS_LIMIT,
                "gasPrice": await self.w3.eth.gas_price,
                "chainId": await self.w3.eth.chain_id,
            }
        )
        signed = self.miner_account.sign_transaction(transaction)
        raw_transaction = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = await self.w3.eth.send_raw_transaction(raw_transaction)
        receipt = await self.w3.eth.wait_for_transaction_receipt(
            tx_hash, timeout=300, poll_latency=2
        )
        if receipt["status"] == 0:
            raise CollateralTransactionError(f"Transaction {tx_hash.hex()} reverted")
        return receipt

    async def reclaim_collateral(self, executor_uuid: str | UUID, url: str = "Manual reclaim"):
        """Start a reclaim of the executor's full collateral; returns its ReclaimProcessStarted."""
        receipt = await self._send(
            self.contract.functions.reclaimCollateral(
                executor_uuid_bytes(executor_uuid), url, bytes(16)
            )
        )
        return self.contract.events.ReclaimProcessStarted().process_receipt(receipt)[0]

    async def finalize_reclaim(self, reclaim_request_id: int):
        """Pay out a reclaim after its deny window; returns its Reclaimed event or None."""
        reclaim = await self.contract.functions.reclaims(reclaim_request_id).call()
        if reclaim[2] == 0:
            raise CollateralTransactionError(
                f"Reclaim request {reclaim_request_id} has already been finalized"
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
                    amount=float(self.w3.from_wei(amount, "ether")),
                    expiration_time=datetime.fromtimestamp(expiration_time, UTC).strftime(
                        DATETIME_FORMAT
                    ),
                    url=log["args"]["url"],
                    url_content_md5_checksum=log["args"]["urlContentMd5Checksum"].hex(),
                    block_number=log["blockNumber"],
                )
            )
        return requests
