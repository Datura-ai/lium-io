"""collateral_deposited, read without celium-collateral.

The backend's provider statistics, its GET /executors filter and the support board read
collateral_deposited as "this executor has its collateral on the contract". The value keeps the
meaning CollateralContractService gave it: the miner hotkey's associated EVM address owns the
executor on the contract, and the executor's collateral covers
required_deposit_amount[gpu_model] × gpu_count × COLLATERAL_DAYS. It has no score effect.

The finalized Substrate block hash is read first. Then the contract's two storage words for the
executor (`executorToMiner[executorId]`, `collaterals[executorId]`) are read from pallet-evm's
AccountStorages with `state_getStorage` at that hash. A Substrate node answers a state query at a
hash only from that block's own state, and a node without the block answers an error, so each
answer is the block's value whichever backend serves it. An eth_call cannot give that: Frontier
answers an unknown block hash from its pending state, and a number can reach a sibling block. A read
is kept per executor for COLLATERAL_STATUS_CACHE_SECONDS; a failed read reports the last answer for
that executor when there is one.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Awaitable, Callable
from uuid import UUID

import aiohttp
import xxhash
from Crypto.Hash import keccak

from core.config import settings, shared_client

RPC_URLS = {
    "local": "http://127.0.0.1:9944",
    "test": "https://test.finney.opentensor.ai",
    "finney": "https://lite.chain.opentensor.ai",
}

# Storage slots of the 1.0.2 contract's mappings (celium-collateral-contracts src/Collateral.sol: NETUID and
# TRUSTEE share slot 0, BURN_ADDRESS and DECISION_TIMEOUT slot 1, MIN_COLLATERAL_INCREASE slot 2). Checked on
# finney against the contract's getters for every open reclaim request.
EXECUTOR_TO_MINER_SLOT = 3  # mapping(bytes16 => address)
COLLATERALS_SLOT = 4  # mapping(bytes16 => uint256)

WEI_PER_TAO = Decimal(10) ** 18

# The RPC is peer-controlled, so its answer is read up to this size before anything is decoded. The answers
# are a block hash and two 32-byte storage words.
MAX_RPC_ANSWER_BYTES = 512 * 1024

RpcBatch = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]


@dataclass(frozen=True)
class CollateralStatus:
    deposited: bool
    contract_version: str | None
    error_message: str | None = None
    collateral_tao: Decimal | None = None
    required_tao: Decimal | None = None
    read_failed: bool = False


def _twox128(data: bytes) -> bytes:
    return b"".join(xxhash.xxh64(data, seed=seed).intdigest().to_bytes(8, "little") for seed in (0, 1))


def _blake2_128_concat(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=16).digest() + data


def mapping_slot(executor_uuid: str, slot: int) -> bytes:
    """The storage slot of `mapping(bytes16 => ...)` at `slot` for the executor; a bytes16 key is left-aligned."""
    key = UUID(executor_uuid).bytes.ljust(32, b"\0")
    return keccak.new(data=key + slot.to_bytes(32, "big"), digest_bits=256).digest()


def evm_storage_key(contract: str, slot: bytes) -> str:
    """The Substrate key of pallet-evm's AccountStorages[contract][slot]."""
    address = bytes.fromhex(contract.removeprefix("0x"))
    key = _twox128(b"EVM") + _twox128(b"AccountStorages") + _blake2_128_concat(address) + _blake2_128_concat(slot)
    return "0x" + key.hex()


def storage_word(result: Any) -> str:
    """A state_getStorage answer as a 32-byte word: an unset slot answers null and reads as zero."""
    if result is None:
        return "00" * 32
    if not isinstance(result, str) or len(result.removeprefix("0x")) != 64:
        raise ValueError("state_getStorage did not answer a 32-byte word")
    return result.removeprefix("0x")


def required_deposit_tao(gpu_model: str | None, gpu_count: int) -> Decimal | None:
    unit = shared_client.config.required_deposit_amount.get(gpu_model) if gpu_model else None
    if unit is None:
        return None
    return Decimal(str(round(unit * gpu_count * settings.COLLATERAL_DAYS, 6)))


def decide(
    *,
    executor_uuid: str,
    evm_address: str | None,
    owner_word: str,
    collateral_word: str,
    required_tao: Decimal | None,
) -> CollateralStatus:
    version = settings.COLLATERAL_CONTRACT_VERSION
    owner = "0x" + _word(owner_word)[-40:]
    collateral_tao = Decimal(int(_word(collateral_word), 16)) / WEI_PER_TAO
    if int(owner, 16) == 0:
        return CollateralStatus(
            False, None, f"No miner address found on contract for executor {executor_uuid}", collateral_tao, required_tao
        )
    if evm_address is None or owner.lower() != evm_address.lower():
        return CollateralStatus(
            False,
            None,
            f"Miner address on contract ({owner}) does not match EVM address ({evm_address}) for executor {executor_uuid}",
            collateral_tao,
            required_tao,
        )
    if required_tao is None:
        return CollateralStatus(False, None, "No required deposit amount found for this GPU model", collateral_tao)
    if collateral_tao < required_tao:
        return CollateralStatus(
            False,
            None,
            f"requires {required_tao} TAO, but the executor has only {collateral_tao} TAO deposited",
            collateral_tao,
            required_tao,
        )
    return CollateralStatus(True, version, None, collateral_tao, required_tao)


def _word(result: str) -> str:
    hex_part = result[2:] if result.startswith("0x") else result
    if len(hex_part) < 64:
        raise ValueError(f"eth_call answered {len(hex_part) // 2} bytes, expected 32")
    return hex_part[:64]


async def _read_bounded(response: aiohttp.ClientResponse, limit: int) -> bytes:
    """The body up to `limit` bytes; one byte more marks it oversized (the caller checks the length)."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.content.iter_chunked(64 * 1024):
        chunks.append(chunk)
        size += len(chunk)
        if size > limit:
            break
    return b"".join(chunks)


async def _post_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    url = settings.SUBTENSOR_EVM_RPC_URL or RPC_URLS[settings.BITTENSOR_NETWORK]
    timeout = aiohttp.ClientTimeout(total=settings.COLLATERAL_STATUS_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=batch) as response:
            response.raise_for_status()
            body = await _read_bounded(response, MAX_RPC_ANSWER_BYTES)
    if len(body) > MAX_RPC_ANSWER_BYTES:
        raise ValueError(f"JSON-RPC answer longer than {MAX_RPC_ANSWER_BYTES} bytes")
    return json.loads(body)


def _evm_address_for_hotkey(hotkey: str) -> str | None:
    from clients.subtensor_client import SubtensorClient

    return SubtensorClient.get_instance().get_evm_address_for_hotkey(hotkey)


class CollateralStatusReader:
    def __init__(
        self,
        rpc: RpcBatch | None = None,
        evm_address_for_hotkey: Callable[[str], str | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._rpc = rpc or _post_batch
        self._evm_address_for_hotkey = evm_address_for_hotkey or _evm_address_for_hotkey
        self._clock = clock
        self._cache: dict[tuple[str, str, str, str | None, int], tuple[float, CollateralStatus]] = {}

    async def status(
        self, *, miner_hotkey: str, executor_uuid: str, gpu_model: str | None, gpu_count: int
    ) -> tuple[CollateralStatus, bool]:
        """The executor's status and whether it came from the cache."""
        evm_address = self._evm_address_for_hotkey(miner_hotkey)
        if evm_address is None:
            # not cached: the miner can associate an address at any time
            return CollateralStatus(
                False, None, f"No evm address found that is associated to this miner hotkey {miner_hotkey} in subnet"
            ), False

        key = (miner_hotkey, executor_uuid, evm_address, gpu_model, gpu_count)
        cached = self._cache.get(key)
        now = self._clock()
        if cached and now - cached[0] < settings.COLLATERAL_STATUS_CACHE_SECONDS:
            return cached[1], True

        try:
            status = await asyncio.wait_for(
                self._read(executor_uuid, evm_address, required_deposit_tao(gpu_model, gpu_count)),
                timeout=settings.COLLATERAL_STATUS_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - any read failure keeps the last answer
            # class name only: an aiohttp error's text can carry the RPC URL, and its path can carry a key
            error = f"Collateral read failed: {type(exc).__name__}"
            if cached:
                stale = cached[1]
                return CollateralStatus(
                    stale.deposited,
                    stale.contract_version,
                    error,
                    stale.collateral_tao,
                    stale.required_tao,
                    read_failed=True,
                ), True
            return CollateralStatus(False, None, error, read_failed=True), False

        self._cache[key] = (now, status)
        return status, False

    async def _read(self, executor_uuid: str, evm_address: str, required_tao: Decimal | None) -> CollateralStatus:
        to = settings.COLLATERAL_CONTRACT_ADDRESS
        [head] = await self._rpc([{"jsonrpc": "2.0", "id": 0, "method": "chain_getFinalizedHead", "params": []}])
        block_hash = head.get("result") if isinstance(head, dict) else None
        if not isinstance(block_hash, str) or len(block_hash.removeprefix("0x")) != 64:
            raise ValueError("chain_getFinalizedHead has no block hash")
        slots = (mapping_slot(executor_uuid, EXECUTOR_TO_MINER_SLOT), mapping_slot(executor_uuid, COLLATERALS_SLOT))
        answers = await self._rpc(
            [
                {"jsonrpc": "2.0", "id": i, "method": "state_getStorage", "params": [evm_storage_key(to, slot), block_hash]}
                for i, slot in enumerate(slots, start=1)
            ]
        )
        if not isinstance(answers, list) or len(answers) != len(slots):
            raise ValueError("state_getStorage has no result")
        by_id = {answer.get("id"): answer for answer in answers if isinstance(answer, dict)}
        words = []
        for i in range(1, len(slots) + 1):
            answer = by_id.get(i)
            if answer is None or "error" in answer or "result" not in answer:
                raise ValueError("state_getStorage has no result")
            words.append(storage_word(answer["result"]))
        owner, collateral = words
        return decide(
            executor_uuid=executor_uuid,
            evm_address=evm_address,
            owner_word=owner,
            collateral_word=collateral,
            required_tao=required_tao,
        )


_reader: CollateralStatusReader | None = None


def get_collateral_status_reader() -> CollateralStatusReader:
    global _reader
    if _reader is None:
        _reader = CollateralStatusReader()
    return _reader
