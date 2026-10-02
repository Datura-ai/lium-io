"""collateral_deposited keeps the meaning its readers (the backend's provider statistics, the GET
/executors filter and the support board) give it, read without celium-collateral, and has no
score effect."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from decimal import Decimal
from types import SimpleNamespace

import pytest
from aiohttp import web
from datura.requests.miner_requests import ExecutorSSHInfo
from neurons.validators.src.services.task.checks import CollateralStatusCheck
from neurons.validators.src.services.task.result_handler import ResultHandler
from services import collateral_status
from services.collateral_status import (
    COLLATERALS_SELECTOR,
    EXECUTOR_TO_MINER_SELECTOR,
    CollateralStatusReader,
    executor_call_data,
)

from core.config import settings
from tests.helpers import build_state

EXECUTOR_UUID = "0b9d2f4e-6c1a-4f3b-9e7d-2a5c8b1d4e6f"
MINER_EVM = "0x1111111111111111111111111111111111111111"
OTHER_EVM = "0x2222222222222222222222222222222222222222"
GPU_MODEL = "NVIDIA H100 80GB HBM3"
GPU_COUNT = 8
BLOCK_HASH = "0x" + "ab" * 32
# 0.103 TAO per H100 × 8 cards × COLLATERAL_DAYS (7)
REQUIRED_TAO = Decimal("5.768")


def _address_word(address: str) -> str:
    return "0x" + address[2:].rjust(64, "0")


def _tao_word(tao: Decimal) -> str:
    return "0x" + format(int(tao * 10**18), "x").rjust(64, "0")


class FakeRpc:
    def __init__(self, owner: str, collateral_tao: Decimal, *, fail: Exception | None = None):
        self.owner, self.collateral_tao, self.fail = owner, collateral_tao, fail
        self.batches: list[list[dict]] = []

    async def __call__(self, batch):
        if self.fail:
            raise self.fail
        if batch[0]["method"] == "eth_getBlockByNumber":
            return [{"jsonrpc": "2.0", "id": batch[0]["id"], "result": {"hash": BLOCK_HASH}}]
        self.batches.append(batch)
        results = {
            executor_call_data(EXECUTOR_TO_MINER_SELECTOR, EXECUTOR_UUID): _address_word(self.owner),
            executor_call_data(COLLATERALS_SELECTOR, EXECUTOR_UUID): _tao_word(self.collateral_tao),
        }
        # answered out of order, as a batch may be
        return [
            {"jsonrpc": "2.0", "id": req["id"], "result": results[req["params"][0]["data"]]} for req in reversed(batch)
        ]


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


def test_call_data_is_the_selector_and_the_left_aligned_uuid():
    data = executor_call_data(COLLATERALS_SELECTOR, EXECUTOR_UUID)
    assert data == "0xfdda13a1" + EXECUTOR_UUID.replace("-", "") + "0" * 32


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner, evm, collateral, gpu_model, deposited, error_part",
    [
        pytest.param(MINER_EVM, MINER_EVM, REQUIRED_TAO, GPU_MODEL, True, None, id="exactly-required"),
        pytest.param(MINER_EVM, MINER_EVM, Decimal(9), GPU_MODEL, True, None, id="above"),
        pytest.param(MINER_EVM, MINER_EVM, Decimal("5.767"), GPU_MODEL, False, "requires 5.768 TAO", id="below"),
        pytest.param(MINER_EVM, MINER_EVM, Decimal(0), GPU_MODEL, False, "requires", id="nothing-deposited"),
        pytest.param("0x" + "0" * 40, MINER_EVM, Decimal(0), GPU_MODEL, False, "No miner address", id="no-owner"),
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
    [batch] = rpc.batches
    assert [req["params"][0]["to"] for req in batch] == [settings.COLLATERAL_CONTRACT_ADDRESS] * 2
    assert [req["params"][1] for req in batch] == [{"blockHash": BLOCK_HASH, "requireCanonical": True}] * 2


@pytest.mark.asyncio
async def test_a_hotkey_with_no_evm_address_is_not_deposited_and_no_rpc_is_made():
    rpc = FakeRpc(MINER_EVM, Decimal(9))
    status, _ = await _status(_reader(rpc, evm=None))

    assert status.deposited is False
    assert "No evm address" in status.error_message
    assert rpc.batches == []


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
    rpc.collateral_tao = Decimal(0)
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
async def test_a_head_with_no_block_hash_is_a_failed_read_and_makes_no_eth_call():
    calls = []

    async def rpc(batch):
        calls.append(batch[0]["method"])
        return [{"jsonrpc": "2.0", "id": batch[0]["id"], "result": None}]

    status, _ = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed, calls) == (False, True, ["eth_getBlockByNumber"])


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
        pytest.param("0x" + "0" * 40, Decimal(0), False, id="no-deposit"),
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
        await response.write(b'[{"jsonrpc": "2.0", "id": 0, "result": {"hash": "' + BLOCK_HASH.encode() + b'", "pad": "')
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
