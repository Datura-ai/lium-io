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


HEADER = {"hash": BLOCK_HASH, "number": hex(16), "parentHash": "0x" + "aa" * 32, "timestamp": hex(1_700_000_000)}
CHILD = {"number": hex(17), "parentHash": BLOCK_HASH, "timestamp": hex(1_700_000_012)}
SIBLING_CHILD = {"number": hex(17), "parentHash": "0x" + "cc" * 32, "timestamp": hex(1_700_000_012)}
# the pending state a backend without BLOCK_HASH runs a call on, as Frontier does
PENDING = {"number": hex(17), "parentHash": "0x" + "bb" * 32, "timestamp": hex(1_700_000_012)}


def _pinned_inner_calls(code: bytes) -> list[tuple[str, int]]:
    """The (calldata, output size) of each view call a pinned_read_code runs, in order."""
    calls, at = [], 0
    while code[at] == 0x61 and code[at + 9] == 0x39:
        length, offset = int.from_bytes(code[at + 1 : at + 3], "big"), int.from_bytes(code[at + 4 : at + 6], "big")
        calls.append(("0x" + code[offset : offset + length].hex(), int.from_bytes(code[at + 11 : at + 13], "big")))
        at += 59
    return calls


class FakeRpc:
    def __init__(self, owner: str, collateral_tao: Decimal, *, fail: Exception | None = None):
        self.owner, self.collateral_tao, self.fail = owner, collateral_tao, fail
        # False: the backend that runs the call does not have BLOCK_HASH and answers from PENDING
        self.has_block = True
        self.pending_owner, self.pending_collateral_tao = owner, collateral_tao
        self.batches: list[list[dict]] = []
        self.head_tags: list[str] = []
        self.pending = PENDING
        self.sibling: tuple[str, Decimal] | None = None
        # True: a gateway sends the hash-pinned call to a backend one block behind, which answers from PENDING
        self.hash_lags = False
        # the state after HEADER's child, when the child changed the executor's owner or collateral
        self.child_state: tuple[str, Decimal] | None = None
        # False: the backend has not imported a child of HEADER yet
        self.has_child = True
        # True: EVM::DisableWhitelistCheck is off and WhitelistedCreators is empty, so a contract-creation call fails
        self.creator_whitelist = False
        # False: the RPC drops eth_call's state override, so a call to an address with no code answers "0x"
        self.honours_overrides = True

    async def __call__(self, batch):
        if self.fail:
            raise self.fail
        if batch[0]["method"] == "eth_getBlockByNumber":
            self.head_tags.append(batch[0]["params"][0])
            return [{"jsonrpc": "2.0", "id": batch[0]["id"], "result": HEADER}]
        self.batches.append(batch)
        return [self._run(req) for req in batch]

    def _code(self, params) -> str | None:
        """The code an eth_call runs: a creation call's data (None while the creator whitelist refuses it), or the
        code a state override sets at the called address ("0x" when there is none)."""
        if "to" not in params[0]:
            return None if self.creator_whitelist else params[0]["data"]
        overrides = params[2] if len(params) > 2 and self.honours_overrides else {}
        return overrides.get(params[0]["to"], {}).get("code", "0x")

    def _run(self, req):
        at = req["params"][1]
        code = self._code(req["params"])
        if code is None:
            return {"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32603, "message": "evm error: NotAllowed"}}
        if code == "0x":
            return {"jsonrpc": "2.0", "id": req["id"], "result": "0x"}
        # as Finney's Frontier answers: an unknown number is "header not found", an unknown hash runs on pending
        if "blockNumber" in at:
            known = self.has_block or self.sibling is not None
            if known and at["blockNumber"] == HEADER["number"]:
                header = HEADER
            elif known and self.has_child and at["blockNumber"] == CHILD["number"]:
                # a backend where a sibling is canonical runs on the sibling's child, whose parent hash is the sibling's
                header = SIBLING_CHILD if self.sibling is not None else CHILD
            else:
                return {"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32000, "message": "header not found"}}
        else:
            known = self.has_block and not self.hash_lags
            header = HEADER if known and at.get("blockHash") == BLOCK_HASH else self.pending
        if header is CHILD and self.child_state is not None:
            owner, tao = self.child_state
        elif header is HEADER or header is CHILD:
            owner, tao = self.owner, self.collateral_tao
        else:
            owner, tao = self.pending_owner, self.pending_collateral_tao
        if "blockNumber" in at and self.sibling is not None:
            # a backend whose block at that height is a sibling with the same number, parent and timestamp
            owner, tao = self.sibling
        results = {
            executor_call_data(EXECUTOR_TO_MINER_SELECTOR, EXECUTOR_UUID): _address_word(owner),
            executor_call_data(COLLATERALS_SELECTOR, EXECUTOR_UUID): _tao_word(tao),
        }
        out = int(header["number"], 16).to_bytes(32, "big") + bytes.fromhex(header["parentHash"][2:])
        out += int(header["timestamp"], 16).to_bytes(32, "big")
        for data, size in _pinned_inner_calls(bytes.fromhex(code[2:])):
            out += bytes.fromhex(results[data][2:])[:size]
        return {"jsonrpc": "2.0", "id": req["id"], "result": "0x" + out.hex()}


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
    [[req, by_number, at_child]] = rpc.batches
    address = collateral_status.PINNED_READ_ADDRESS
    assert req["params"][0] == {"to": address, "data": "0x"}
    assert req["params"][1] == {"blockHash": BLOCK_HASH, "requireCanonical": True}
    assert by_number["params"] == [req["params"][0], {"blockNumber": HEADER["number"]}, req["params"][2]]
    assert at_child["params"] == [req["params"][0], {"blockNumber": CHILD["number"]}, req["params"][2]]
    assert rpc.head_tags == ["finalized"]
    contract = settings.COLLATERAL_CONTRACT_ADDRESS[2:].lower()
    code = req["params"][2][address]["code"]
    assert code.count("73" + contract) == 2
    assert _pinned_inner_calls(bytes.fromhex(code[2:])) == [
        (executor_call_data(EXECUTOR_TO_MINER_SELECTOR, EXECUTOR_UUID), 32),
        (executor_call_data(COLLATERALS_SELECTOR, EXECUTOR_UUID), 32),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("creator_whitelist", [False, True], ids=["whitelist-check-disabled", "whitelist-check-on"])
async def test_the_pinned_read_does_not_depend_on_the_evm_creator_whitelist(creator_whitelist):
    """Review 5397591319 at 78c163c: Subtensor runs a contract-creation eth_call through WhitelistedCreators unless
    EVM::DisableWhitelistCheck is set. The pinned read is a plain call to PINNED_READ_ADDRESS with its code set by
    the state override, so it reads the same answer with the check on or off."""
    rpc = FakeRpc(MINER_EVM, Decimal(9))
    rpc.creator_whitelist = creator_whitelist
    status, cached = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed, cached) == (True, False, False)
    assert status.collateral_tao == Decimal(9)


@pytest.mark.asyncio
async def test_an_rpc_that_drops_the_state_override_is_a_failed_read():
    rpc = FakeRpc(MINER_EVM, Decimal(9))
    rpc.honours_overrides = False
    status, cached = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed, cached) == (False, True, False)
    with pytest.raises(ValueError, match="state overrides"):
        collateral_status.pinned_outputs("0x", HEADER, [32, 32])


@pytest.mark.asyncio
async def test_a_read_pinned_to_a_block_the_rpc_does_not_have_is_a_failed_read():
    rpc = FakeRpc(MINER_EVM, Decimal(9))
    rpc.has_block = False
    status, cached = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed, cached) == (False, True, False)


@pytest.mark.asyncio
async def test_a_read_a_gateway_sends_to_a_backend_without_the_block_is_not_published_as_deposited():
    """Review 5397109362 at 0132b5c: the header comes from a backend that has the block, the call from one that
    does not and answers from its pending state, where the miner owns a funded executor. The call's own header
    words name another block, so the read fails instead of caching collateral_deposited=True."""
    rpc = FakeRpc("0x" + "0" * 40, Decimal(0))
    rpc.has_block = False
    rpc.pending_owner, rpc.pending_collateral_tao = MINER_EVM, Decimal(9)
    status, cached = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed, cached) == (False, True, False)
    rpc.has_block = True
    status, _ = await _status(_reader(rpc))
    assert (status.deposited, status.read_failed) == (False, False)


@pytest.mark.asyncio
async def test_a_pending_block_with_the_pinned_number_parent_and_timestamp_is_never_read():
    """Review at 78c163c: a backend one block behind can answer an unknown hash from a pending block with the
    pinned header's number, parent and timestamp but other collateral. The read is pinned by number, which such a
    backend does not have, so it fails instead of reporting the pending state."""
    rpc = FakeRpc("0x" + "0" * 40, Decimal(0))
    rpc.has_block = False
    rpc.pending = {k: HEADER[k] for k in ("number", "parentHash", "timestamp")}
    rpc.pending_owner, rpc.pending_collateral_tao = MINER_EVM, Decimal(9)
    status, cached = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed, cached) == (False, True, False)


@pytest.mark.asyncio
async def test_a_sibling_block_with_the_pinned_number_parent_and_timestamp_is_never_read():
    """Review at aca7529: the numbered call can reach a backend whose block at that height is a sibling of the
    finalized one, with the same number, parent and timestamp but the miner's deposit. The read pinned by hash runs
    on the finalized block itself, the two disagree, and the read fails instead of caching collateral_deposited."""
    rpc = FakeRpc("0x" + "0" * 40, Decimal(0))
    rpc.sibling = (MINER_EVM, Decimal(9))
    status, cached = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed, cached) == (False, True, False)
    rpc.sibling = None
    status, _ = await _status(_reader(rpc))
    assert (status.deposited, status.read_failed) == (False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "at_block, in_sibling_and_child, deposited",
    [
        pytest.param(("0x" + "0" * 40, Decimal(0)), (MINER_EVM, Decimal(9)), False, id="deposit-after-the-block"),
        pytest.param((MINER_EVM, Decimal(9)), ("0x" + "0" * 40, Decimal(0)), True, id="reclaim-after-the-block"),
    ],
)
async def test_a_pending_sibling_that_agrees_with_the_child_is_never_read(at_block, in_sibling_and_child, deposited):
    """Review 5398736862 at d391eb0: the hash-pinned call reaches a backend one block behind, whose pending sibling
    carries the block's number, parent and timestamp, and the child changed the executor the same way the sibling
    did. The sibling's and the child's runs agree; the run pinned by the block's number reads the block's own state,
    so the read fails instead of caching either answer."""
    rpc = FakeRpc(*at_block)
    rpc.hash_lags = True
    rpc.pending = {k: HEADER[k] for k in ("number", "parentHash", "timestamp")}
    rpc.pending_owner, rpc.pending_collateral_tao = in_sibling_and_child
    rpc.child_state = in_sibling_and_child
    status, cached = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed, cached) == (False, True, False)
    rpc.hash_lags = False
    rpc.child_state = None
    status, _ = await _status(_reader(rpc))
    assert (status.deposited, status.read_failed) == (deposited, False)


@pytest.mark.asyncio
async def test_a_lagging_pending_sibling_and_a_backend_where_the_sibling_is_canonical_are_never_read():
    """Review at 8a84360: the hash-pinned call reaches a backend one block behind, which runs on its pending block,
    a sibling with the finalized number, parent and timestamp; the other call reaches a backend where that sibling
    is canonical. Both see the sibling's deposit. The second run is on a child, where BLOCKHASH(NUMBER - 1) is the
    sibling's hash rather than the finalized one, so the read fails instead of caching collateral_deposited."""
    rpc = FakeRpc("0x" + "0" * 40, Decimal(0))
    rpc.has_block = False
    rpc.pending = {k: HEADER[k] for k in ("number", "parentHash", "timestamp")}
    rpc.pending_owner, rpc.pending_collateral_tao = MINER_EVM, Decimal(9)
    rpc.sibling = (MINER_EVM, Decimal(9))
    status, cached = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed, cached) == (False, True, False)


@pytest.mark.asyncio
async def test_a_finalized_block_with_no_child_yet_is_a_failed_read():
    rpc = FakeRpc(MINER_EVM, Decimal(9))
    rpc.has_child = False
    status, cached = await _status(_reader(rpc))

    assert (status.deposited, status.read_failed, cached) == (False, True, False)


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
