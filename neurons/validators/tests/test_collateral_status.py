"""collateral_deposited keeps the meaning its readers (the backend's provider statistics, the GET
/executors filter and the support board) give it, read without celium-collateral, and has no
score effect."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from decimal import Decimal
from types import SimpleNamespace

import pytest
from aiohttp import web
from datura.requests.miner_requests import ExecutorSSHInfo
from neurons.validators.src.services.task.checks import CollateralStatusCheck
from neurons.validators.src.services.task.result_handler import ResultHandler
from services.collateral_status import (
    COLLATERALS_SLOT,
    EXECUTOR_TO_MINER_SLOT,
    CollateralStatusReader,
    evm_storage_key,
    mapping_slot,
)

from core.config import settings
from services import collateral_status
from tests.helpers import build_state

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


def test_the_storage_keys_are_the_ones_finney_answers_for_the_1_0_2_contract():
    """Executor 65af4497-… has open reclaim request 22 on the 1.0.2 contract. On finney, state_getStorage at these
    keys answered the same words as executorToMiner() and collaterals() for it."""
    contract = "0x8A4023FdD1eaA7b242F3723a7d096B6CC693c7C6"
    executor = "65af4497-335c-4528-8523-0441735e665d"
    prefix = (
        "0x1da53b775b270400e7e61ed5cbc5a146ab1160471b1418779239ba8e2b847e42"
        "81e8cde4a494fd9b81dba034e0b8913d8a4023fdd1eaa7b242f3723a7d096b6cc693c7c6"
    )
    assert evm_storage_key(contract, mapping_slot(executor, EXECUTOR_TO_MINER_SLOT)) == prefix + (
        "cc6a004345680850ecc4e3472c55acf28c408095c99a96eaa467d8913efe65457673a96dd0a6537c0319b06a05e04cdd"
    )
    assert evm_storage_key(contract, mapping_slot(executor, COLLATERALS_SLOT)) == prefix + (
        "f7079d893e7f92eba7c2eaf578e3e6109d265339201d214e81869ea6bc9dae1f3d0dd728b994e3535a7e56a2b22da043"
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


@pytest.mark.asyncio
async def test_a_read_at_a_block_the_backend_does_not_have_is_a_failed_read():
    rpc = FakeRpc(MINER_EVM, Decimal(9))
    rpc.backends.append({SIBLING_HASH: (MINER_EVM, Decimal(9))})
    rpc.route = lambda _i, _req: 1
    status, cached = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed, cached) == (False, True, False)


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
@pytest.mark.parametrize(
    "word",
    ["0x" + "11" * 31, "0x" + "11" * 33, 7, {"value": "0x00"}],
    ids=["short", "long", "number", "object"],
)
async def test_a_storage_answer_that_is_not_a_32_byte_word_is_a_failed_read(word):
    async def rpc(batch):
        if batch[0]["method"] == "chain_getFinalizedHead":
            return [{"jsonrpc": "2.0", "id": 0, "result": BLOCK_HASH}]
        return [{"jsonrpc": "2.0", "id": req["id"], "result": word} for req in batch]

    status, _ = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed) == (False, True)


@pytest.mark.asyncio
async def test_two_answers_with_one_id_are_a_failed_read():
    fake = FakeRpc(MINER_EVM, Decimal(9))

    async def rpc(batch):
        answers = await fake(batch)
        if batch[0]["method"] == "state_getStorage":
            answers = [answers[1], dict(answers[1])]
        return answers

    status, _ = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed) == (False, True)


@pytest.mark.asyncio
async def test_a_hotkey_with_no_evm_address_is_not_deposited_and_no_rpc_is_made():
    rpc = FakeRpc(MINER_EVM, Decimal(9))
    status, _ = await _status(_reader(rpc, evm=None))

    assert status.deposited is False
    assert "No evm address" in status.error_message
    assert (rpc.batches, rpc.heads) == ([], 0)


@pytest.mark.asyncio
async def test_a_read_is_reused_until_the_cache_expires():
    clock = Clock()
    rpc = FakeRpc(MINER_EVM, Decimal(9))
    reader = _reader(rpc, clock=clock)

    assert (await _status(reader))[0].deposited is True
    clock.now += settings.COLLATERAL_STATUS_CACHE_SECONDS - 1
    status, cached = await _status(reader)
    assert (status.deposited, cached, len(rpc.batches)) == (True, True, 1)

    clock.now += 2
    rpc.set_state(MINER_EVM, Decimal(0))
    status, cached = await _status(reader)
    assert (status.deposited, cached, len(rpc.batches)) == (False, False, 2)


@pytest.mark.asyncio
async def test_a_missing_evm_address_is_not_cached_once_the_miner_associates_one():
    clock = Clock()
    address = [None]
    reader = CollateralStatusReader(
        rpc=FakeRpc(MINER_EVM, Decimal(9)),
        evm_address_for_hotkey=lambda _: address[0],
        clock=clock,
    )
    assert (await _status(reader))[0].deposited is False
    address[0] = MINER_EVM
    clock.now += 121
    assert (await _status(reader))[0].deposited is True


@pytest.mark.asyncio
async def test_a_failed_read_keeps_the_last_answer_and_names_only_the_error_class():
    clock = Clock()
    rpc = FakeRpc(MINER_EVM, Decimal(9))
    reader = _reader(rpc, clock=clock)
    await _status(reader)

    clock.now += settings.COLLATERAL_STATUS_CACHE_SECONDS + 1
    rpc.fail = ConnectionError("https://rpc.example/key-in-path")
    status, cached = await _status(reader)

    assert (status.deposited, status.contract_version, status.read_failed, cached) == (True, "1.0.2", True, True)
    assert status.error_message == "Collateral read failed: ConnectionError"


@pytest.mark.asyncio
async def test_a_failed_first_read_reports_not_deposited():
    status, cached = await _status(_reader(FakeRpc(MINER_EVM, Decimal(9), fail=TimeoutError())))

    assert (status.deposited, status.read_failed, cached) == (False, True, False)


@pytest.mark.asyncio
async def test_an_rpc_error_answer_is_a_failed_read():
    async def rpc(batch):
        return [{"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32000, "message": "x"}} for req in batch]

    status, _ = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed) == (False, True)


@pytest.mark.asyncio
async def test_a_head_with_no_block_hash_is_a_failed_read_and_makes_no_storage_read():
    calls = []

    async def rpc(batch):
        calls.append(batch[0]["method"])
        return [{"jsonrpc": "2.0", "id": batch[0]["id"], "result": None}]

    status, _ = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed, calls) == (False, True, ["chain_getFinalizedHead"])


def _context(context_factory):
    executor = ExecutorSSHInfo(
        uuid=EXECUTOR_UUID,
        address="127.0.0.1",
        port=22,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python",
        root_dir="/root/app",
        price_per_gpu=0.5,
    )
    state = build_state(
        specs={"gpu": {"count": GPU_COUNT, "details": [{"name": GPU_MODEL}] * GPU_COUNT}}, gpu_count=GPU_COUNT
    )
    return context_factory(
        executor=executor,
        state=state,
        tdx_attestation_passed=False,
        score=1.0,
        job_score=1.0,
        ssh_pub_keys=[],
        rented=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner, collateral, expected",
    [
        pytest.param(MINER_EVM, REQUIRED_TAO, True, id="funded"),
        pytest.param(MINER_EVM, Decimal("0.01"), False, id="underfunded"),
        pytest.param(NOBODY, Decimal(0), False, id="no-deposit"),
    ],
)
async def test_the_check_puts_the_contract_answer_in_the_published_job_result(
    context_factory, monkeypatch, owner, collateral, expected
):
    monkeypatch.setattr(collateral_status, "_reader", _reader(FakeRpc(owner, collateral)))
    ctx = _context(context_factory)

    result = await CollateralStatusCheck().run(ctx)

    assert result.passed is True
    assert CollateralStatusCheck.fatal is False
    assert result.updates["collateral_deposited"] is expected
    assert result.event.reason_code == ("COLLATERAL_OK" if expected else "COLLATERAL_MISSING")
    assert result.event.severity == "info"

    ctx = ctx.model_copy(update=result.updates)
    job = await ResultHandler(redis_service=None, dry_run=True).handle_result(
        context=ctx,
        miner_info=SimpleNamespace(miner_hotkey="miner-hotkey", job_batch_id="batch-1"),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )
    assert job.collateral_deposited is expected
    assert (job.score, job.job_score) == (1.0, 1.0)


@contextlib.asynccontextmanager
async def _rpc_server(monkeypatch, handler):
    app = web.Application()
    app.router.add_post("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setattr(settings, "SUBTENSOR_EVM_RPC_URL", f"http://127.0.0.1:{port}/")
    try:
        yield
    finally:
        await runner.cleanup()


def _http_reader():
    return CollateralStatusReader(evm_address_for_hotkey=lambda _hotkey: MINER_EVM, clock=Clock())


@pytest.mark.asyncio
async def test_the_rpc_answer_is_read_over_http_and_decoded(monkeypatch):
    fake = FakeRpc(MINER_EVM, Decimal(9))

    async def handler(request):
        return web.Response(body=json.dumps(await fake(await request.json())), content_type="application/json")

    async with _rpc_server(monkeypatch, handler):
        status, _ = await _status(_http_reader())

    assert (status.deposited, status.read_failed) == (True, False)


@pytest.mark.asyncio
async def test_an_rpc_answer_over_the_size_cap_is_a_failed_read_and_is_not_buffered(monkeypatch):
    """Review comment 4168134745 at 4ac762f: a broken or hostile RPC can answer a body of any size inside the
    timeout. This one streams valid JSON with no end; the read stops one byte past the cap.

    Read b at 0a6fee9: a read-all-then-check client raises the same error, so the bytes the client takes off the
    socket are counted too. The server offers 64 MiB; the client must stop near the cap."""
    import aiohttp

    received = []
    feed_data = aiohttp.StreamReader.feed_data

    def counting_feed_data(self, data, *args, **kwargs):
        received.append(len(data))
        return feed_data(self, data, *args, **kwargs)

    monkeypatch.setattr(aiohttp.StreamReader, "feed_data", counting_feed_data)

    async def handler(request):
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b'[{"jsonrpc": "2.0", "id": 0, "result": "' + BLOCK_HASH.encode() + b'", "pad": "')
        try:
            for _ in range(1024):
                await response.write(b"x" * 64 * 1024)
        except (ConnectionError, RuntimeError):
            pass
        return response

    async with _rpc_server(monkeypatch, handler):
        with pytest.raises(ValueError, match="longer than"):
            await collateral_status._post_batch([{"jsonrpc": "2.0", "id": 0, "method": "eth_blockNumber"}])
        # the cap, one chunk past it, and what flow control lets in before the read stops
        assert 0 < sum(received) <= collateral_status.MAX_RPC_ANSWER_BYTES + 2 * 1024 * 1024
        status, cached = await _status(_http_reader())

    assert (status.deposited, status.read_failed, cached) == (False, True, False)
    assert status.error_message == "Collateral read failed: ValueError"


@pytest.mark.asyncio
async def test_a_slow_drip_rpc_answer_is_cut_at_the_timeout(monkeypatch):
    monkeypatch.setattr(settings, "COLLATERAL_STATUS_TIMEOUT_SECONDS", 0.5)

    async def handler(request):
        response = web.StreamResponse()
        await response.prepare(request)
        try:
            await response.write(b"[")
            for _ in range(100):
                await asyncio.sleep(0.1)
                await response.write(b" ")
        except (ConnectionError, RuntimeError):
            pass
        return response

    async with _rpc_server(monkeypatch, handler):
        started = time.monotonic()
        status, _ = await _status(_http_reader())
        elapsed = time.monotonic() - started

    assert (status.deposited, status.read_failed) == (False, True)
    assert "TimeoutError" in status.error_message
    assert elapsed < 3
