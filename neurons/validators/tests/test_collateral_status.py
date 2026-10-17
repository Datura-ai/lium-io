"""collateral_deposited keeps the meaning its readers (the backend's provider statistics, the GET
/executors filter and the support board) give it, read without celium-collateral, and has no
score effect."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

import pytest
from services.collateral_status import (
    COLLATERALS_SLOT,
    EXECUTOR_TO_MINER_SLOT,
    CollateralStatusReader,
    evm_storage_key,
    mapping_slot,
)

from core.config import settings

EXECUTOR_UUID = "0b9d2f4e-6c1a-4f3b-9e7d-2a5c8b1d4e6f"
MINER_EVM = "0x1111111111111111111111111111111111111111"
OTHER_EVM = "0x2222222222222222222222222222222222222222"
NOBODY = "0x" + "0" * 40
GPU_MODEL = "NVIDIA H100 80GB HBM3"
GPU_COUNT = 8
BLOCK_HASH = "0x" + "ab" * 32
# blocks other than the finalized one: a sibling at its height, and its child
SIBLING_HASH = "0x" + "cc" * 32
CHILD_HASH = "0x" + "dd" * 32
# 0.103 TAO per H100 × 8 cards × COLLATERAL_DAYS (7)
REQUIRED_TAO = Decimal("5.768")


def _address_word(address: str) -> str | None:
    return None if int(address, 16) == 0 else "0x" + address[2:].rjust(64, "0")


def _tao_word(tao: Decimal) -> str | None:
    return None if tao == 0 else "0x" + format(int(tao * 10**18), "x").rjust(64, "0")


def _storage(owner: str, tao: Decimal) -> dict[str, str | None]:
    """One block's AccountStorages entries for the executor; an unset slot is absent and answers null."""
    contract = settings.COLLATERAL_CONTRACT_ADDRESS
    return {
        evm_storage_key(contract, mapping_slot(EXECUTOR_UUID, EXECUTOR_TO_MINER_SLOT)): _address_word(owner),
        evm_storage_key(contract, mapping_slot(EXECUTOR_UUID, COLLATERALS_SLOT)): _tao_word(tao),
    }


class FakeRpc:
    """A gateway in front of Substrate backends. Each backend holds the state of the blocks it has imported and
    answers state_getStorage at a hash only from that block, or "UnknownBlock" when it does not have it, as a
    Subtensor node does. `route` picks the backend for each request of a batch."""

    def __init__(self, owner: str, collateral_tao: Decimal, *, fail: Exception | None = None):
        self.fail = fail
        self.backends: list[dict[str, tuple[str, Decimal]]] = [{BLOCK_HASH: (owner, collateral_tao)}]
        self.route: Callable[[int, dict], int] = lambda _i, _req: 0
        self.batches: list[list[dict]] = []
        self.heads = 0

    def set_state(self, owner: str, collateral_tao: Decimal) -> None:
        self.backends[0][BLOCK_HASH] = (owner, collateral_tao)

    async def __call__(self, batch):
        if self.fail:
            raise self.fail
        if batch[0]["method"] == "chain_getFinalizedHead":
            self.heads += 1
            return [{"jsonrpc": "2.0", "id": batch[0]["id"], "result": BLOCK_HASH}]
        self.batches.append(batch)
        return [self._answer(self.backends[self.route(i, req)], req) for i, req in enumerate(batch)]

    @staticmethod
    def _answer(backend, req):
        assert req["method"] == "state_getStorage"
        key, at = req["params"]
        if at not in backend:
            return {"jsonrpc": "2.0", "id": req["id"], "error": {"code": 4003, "message": f"UnknownBlock: {at}"}}
        return {"jsonrpc": "2.0", "id": req["id"], "result": _storage(*backend[at]).get(key)}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _reader(rpc, evm=MINER_EVM, clock=None):
    return CollateralStatusReader(rpc=rpc, evm_address_for_hotkey=lambda _hotkey: evm, clock=clock or Clock())


async def _status(reader, gpu_model=GPU_MODEL, gpu_count=GPU_COUNT):
    return await reader.status(
        miner_hotkey="miner-hotkey", executor_uuid=EXECUTOR_UUID, gpu_model=gpu_model, gpu_count=gpu_count
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner, evm, collateral, gpu_model, deposited, error_part",
    [
        pytest.param(MINER_EVM, MINER_EVM, REQUIRED_TAO, GPU_MODEL, True, None, id="exactly-required"),
        pytest.param(MINER_EVM, MINER_EVM, Decimal(9), GPU_MODEL, True, None, id="above"),
        pytest.param(MINER_EVM, MINER_EVM, Decimal("5.767"), GPU_MODEL, False, "requires 5.768 TAO", id="below"),
        pytest.param(MINER_EVM, MINER_EVM, Decimal(0), GPU_MODEL, False, "requires", id="nothing-deposited"),
        pytest.param(NOBODY, MINER_EVM, Decimal(0), GPU_MODEL, False, "No miner address", id="no-owner"),
        pytest.param(OTHER_EVM, MINER_EVM, Decimal(9), GPU_MODEL, False, "does not match", id="other-owner"),
        pytest.param(MINER_EVM, MINER_EVM, Decimal(9), "Unknown GPU", False, "No required deposit", id="no-rate"),
    ],
)
async def test_deposited_means_owned_by_the_miner_and_covers_the_requirement(
    owner, evm, collateral, gpu_model, deposited, error_part
):
    rpc = FakeRpc(owner, collateral)
    status, cached = await _status(_reader(rpc, evm=evm), gpu_model=gpu_model)

    assert status.deposited is deposited
    assert status.contract_version == ("1.0.2" if deposited else None)
    assert cached is False
    assert (status.error_message is None) if error_part is None else (error_part in status.error_message)
    contract = settings.COLLATERAL_CONTRACT_ADDRESS
    [batch] = rpc.batches
    assert [(req["method"], req["params"]) for req in batch] == [
        ("state_getStorage", [evm_storage_key(contract, mapping_slot(EXECUTOR_UUID, EXECUTOR_TO_MINER_SLOT)), BLOCK_HASH]),
        ("state_getStorage", [evm_storage_key(contract, mapping_slot(EXECUTOR_UUID, COLLATERALS_SLOT)), BLOCK_HASH]),
    ]
    assert rpc.heads == 1


ROUTES = {
    "one-backend": lambda i, _req: 0,
    "owner-to-lagging": lambda i, _req: 1 if i == 0 else 0,
    "amount-to-lagging": lambda i, _req: 1 if i == 1 else 0,
    "owner-to-sibling-canonical": lambda i, _req: 2 if i == 0 else 0,
    "split-lagging-and-sibling": lambda i, _req: 1 if i == 0 else 2,
    "all-to-sibling-canonical": lambda i, _req: 2,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", list(ROUTES), ids=list(ROUTES))
@pytest.mark.parametrize(
    "at_block, in_sibling_and_child",
    [
        pytest.param((NOBODY, Decimal(0)), (MINER_EVM, Decimal(9)), id="deposit-only-off-the-block"),
        pytest.param((MINER_EVM, Decimal(9)), (NOBODY, Decimal(0)), id="reclaim-only-off-the-block"),
    ],
)
async def test_state_off_the_finalized_block_is_never_read(route, at_block, in_sibling_and_child):
    """Review 5398901003 at 4fa570d: the finalized block H has no deposit, its pending sibling S and H's child C
    both carry one, and the gateway sends each read to another backend: one a block behind, whose pending block
    is S, and one where S is canonical. Each read names H by its Substrate hash, which a backend answers from H's
    own state or refuses, so the answer is H's or the read fails; S's and C's state is never reported."""
    rpc = FakeRpc(*at_block)
    rpc.backends[0][CHILD_HASH] = in_sibling_and_child
    rpc.backends.append({SIBLING_HASH: in_sibling_and_child})
    rpc.backends.append({SIBLING_HASH: in_sibling_and_child, CHILD_HASH: in_sibling_and_child})
    rpc.route = ROUTES[route]
    status, cached = await _status(_reader(rpc))

    expected = (at_block[0] == MINER_EVM, False) if route == "one-backend" else (False, True)
    assert (status.deposited, status.read_failed) == expected
    assert cached is False


@pytest.mark.asyncio
async def test_a_failed_first_read_reports_not_deposited():
    status, cached = await _status(_reader(FakeRpc(MINER_EVM, Decimal(9), fail=TimeoutError())))

    assert (status.deposited, status.read_failed, cached) == (False, True, False)


