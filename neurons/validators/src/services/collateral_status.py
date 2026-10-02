"""collateral_deposited, read without celium-collateral.

The backend's provider statistics, its GET /executors filter and the support board read
collateral_deposited as "this executor has its collateral on the contract". The value keeps the
meaning CollateralContractService gave it: the miner hotkey's associated EVM address owns the
executor on the contract, and the executor's collateral covers
required_deposit_amount[gpu_model] × gpu_count × COLLATERAL_DAYS. It has no score effect.

Two view calls (`executorToMiner(bytes16)`, `collaterals(bytes16)`) go out as one JSON-RPC batch
of eth_call. A read is kept per executor for COLLATERAL_STATUS_CACHE_SECONDS; a failed read
reports the last answer for that executor when there is one.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Awaitable, Callable
from uuid import UUID

import aiohttp

from core.config import settings, shared_client

RPC_URLS = {
    "local": "http://127.0.0.1:9944",
    "test": "https://test.finney.opentensor.ai",
    "finney": "https://lite.chain.opentensor.ai",
}

# keccak256 of the signature, first 4 bytes
EXECUTOR_TO_MINER_SELECTOR = "f44e1119"  # executorToMiner(bytes16) -> address
COLLATERALS_SELECTOR = "fdda13a1"  # collaterals(bytes16) -> uint256

WEI_PER_TAO = Decimal(10) ** 18

RpcBatch = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]


@dataclass(frozen=True)
class CollateralStatus:
    deposited: bool
    contract_version: str | None
    error_message: str | None = None
    collateral_tao: Decimal | None = None
    required_tao: Decimal | None = None
    read_failed: bool = False


def executor_call_data(selector: str, executor_uuid: str) -> str:
    # bytes16 is left-aligned in its 32-byte word
    return "0x" + selector + UUID(executor_uuid).bytes.hex().ljust(64, "0")


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


async def _post_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    url = settings.SUBTENSOR_EVM_RPC_URL or RPC_URLS[settings.BITTENSOR_NETWORK]
    timeout = aiohttp.ClientTimeout(total=settings.COLLATERAL_STATUS_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=batch) as response:
            response.raise_for_status()
            return await response.json(content_type=None)


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
        # both calls at one block hash (EIP-1898): "latest" twice can answer from two blocks or forks,
        # pairing an old owner with a new owner's deposit
        [head] = await self._rpc(
            [{"jsonrpc": "2.0", "id": 0, "method": "eth_getBlockByNumber", "params": ["latest", False]}]
        )
        block_hash = (head.get("result") or {}).get("hash") if isinstance(head, dict) else None
        if not block_hash:
            raise ValueError("eth_getBlockByNumber has no block hash")
        batch = [
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "eth_call",
                "params": [
                    {"to": to, "data": executor_call_data(selector, executor_uuid)},
                    {"blockHash": block_hash, "requireCanonical": True},
                ],
            }
            for request_id, selector in ((1, EXECUTOR_TO_MINER_SELECTOR), (2, COLLATERALS_SELECTOR))
        ]
        answers = await self._rpc(batch)
        if not isinstance(answers, list):
            raise ValueError("JSON-RPC batch answered with a single object")
        by_id = {answer.get("id"): answer for answer in answers if isinstance(answer, dict)}
        results = []
        for request_id in (1, 2):
            answer = by_id.get(request_id)
            if answer is None or "error" in answer or "result" not in answer:
                raise ValueError(f"eth_call {request_id} has no result")
            results.append(answer["result"])
        return decide(
            executor_uuid=executor_uuid,
            evm_address=evm_address,
            owner_word=results[0],
            collateral_word=results[1],
            required_tao=required_tao,
        )


_reader: CollateralStatusReader | None = None


def get_collateral_status_reader() -> CollateralStatusReader:
    global _reader
    if _reader is None:
        _reader = CollateralStatusReader()
    return _reader
