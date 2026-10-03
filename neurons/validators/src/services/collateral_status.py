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
import json
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

# The RPC is peer-controlled, so its answer is read up to this size before anything is decoded. The largest
# answer is the latest header, which lists one ~70-byte hash per transaction, so this holds a block of over
# 7,000 transactions; the two eth_call answers are ~200 bytes.
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


def executor_call_data(selector: str, executor_uuid: str) -> str:
    # bytes16 is left-aligned in its 32-byte word
    return "0x" + selector + UUID(executor_uuid).bytes.hex().ljust(64, "0")


def pinned_read_code(to: str, calls: list[tuple[bytes, int]]) -> str:
    """Init code for a contract-creation eth_call that makes each (calldata, output size) view call to `to` in one
    EVM run and returns NUMBER, BLOCKHASH(NUMBER - 1) and TIMESTAMP, then each call's first `size` bytes. One run
    reads one state on one backend, and the three header words prove which block that state is."""
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


def pinned_outputs(result: str, header: dict[str, Any], sizes: list[int]) -> list[bytes]:
    """The view call outputs of a pinned_read_code answer, once its header words match `header`. Frontier answers
    a block hash it does not know from its pending state, which carries another number, parent or timestamp."""
    if not isinstance(result, str):
        raise ValueError("pinned read has no answer")
    raw = bytes.fromhex(result.removeprefix("0x"))
    if len(raw) != 0x60 + sum(sizes):
        raise ValueError(f"pinned read answered {len(raw)} bytes")
    number, parent, timestamp = (raw[i : i + 32] for i in (0, 32, 64))
    if (
        int.from_bytes(number, "big") != int(header["number"], 16)
        or "0x" + parent.hex() != str(header["parentHash"]).lower()
        or int.from_bytes(timestamp, "big") != int(header["timestamp"], 16)
    ):
        raise ValueError("the read did not run on the block it is pinned to")
    outputs, at = [], 0x60
    for size in sizes:
        outputs.append(raw[at : at + size])
        at += size
    return outputs


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
        # owner and amount come from one EVM run (pinned_read_code), so one backend reads both from one state; a
        # gateway that splits a batch across backends cannot pair an old owner with a new owner's deposit
        [head] = await self._rpc(
            [{"jsonrpc": "2.0", "id": 0, "method": "eth_getBlockByNumber", "params": ["finalized", False]}]
        )
        header = head.get("result") if isinstance(head, dict) else None
        if not isinstance(header, dict) or not all(header.get(k) for k in ("hash", "number", "parentHash", "timestamp")):
            raise ValueError("eth_getBlockByNumber has no block header")
        calls = [
            (bytes.fromhex(executor_call_data(selector, executor_uuid)[2:]), 32)
            for selector in (EXECUTOR_TO_MINER_SELECTOR, COLLATERALS_SELECTOR)
        ]
        # Neither pin names the block alone: an unknown hash runs on pending state that can carry the block's
        # number, parent and timestamp, and a number can reach a sibling block with the same three words. So the
        # read runs under both, and its answer counts only when both runs match the header and agree.
        code = pinned_read_code(to, calls)
        pins = ({"blockHash": header["hash"], "requireCanonical": True}, {"blockNumber": header["number"]})
        answers = await self._rpc(
            [
                {"jsonrpc": "2.0", "id": i, "method": "eth_call", "params": [{"data": code}, pin]}
                for i, pin in enumerate(pins, start=1)
            ]
        )
        if not isinstance(answers, list) or len(answers) != len(pins):
            raise ValueError("eth_call has no result")
        runs = []
        for answer in sorted(answers, key=lambda a: a.get("id", -1) if isinstance(a, dict) else -1):
            if not isinstance(answer, dict) or "error" in answer or "result" not in answer:
                raise ValueError("eth_call has no result")
            runs.append(pinned_outputs(answer["result"], header, [32, 32]))
        if runs[0] != runs[1]:
            raise ValueError("the reads pinned by hash and by number disagree")
        owner, collateral = runs[0]
        return decide(
            executor_uuid=executor_uuid,
            evm_address=evm_address,
            owner_word=owner.hex(),
            collateral_word=collateral.hex(),
            required_tao=required_tao,
        )


_reader: CollateralStatusReader | None = None


def get_collateral_status_reader() -> CollateralStatusReader:
    global _reader
    if _reader is None:
        _reader = CollateralStatusReader()
    return _reader
