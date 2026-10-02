"""`LiumdExecClient`'s exit-status mapping, bounds and exec request,
against a scripted asyncssh process. The binary itself answers in test_liumd_exec_e2e.py."""

from __future__ import annotations

import asyncio
import json

import asyncssh
import bittensor
import pytest
from datura.requests.validator_requests import VerifyXStep
from services.liumd_exec_client import (
    LIUMD_COMMAND,
    REFUSAL_ERRORS_BY_EXIT,
    LiumdExecClient,
    LiumdRefusal,
)
from services.local_verify_client import (
    MAX_ANSWER_BYTES,
    SCHEMA,
    LocalVerifyAnswer,
    LocalVerifyUnavailable,
    build_intent,
)


class _Reader:
    def __init__(self, data: bytes, *, hang: bool = False):
        self.data = data
        self.hang = hang
        self.reads = 0

    async def read(self, n: int) -> bytes:
        self.reads += 1
        if not self.data:
            if self.hang:
                await asyncio.Event().wait()
            return b""
        chunk, self.data = self.data[:n], self.data[n:]
        return chunk


class _Writer:
    def __init__(self):
        self.data = b""
        self.eof = False

    def write(self, data: bytes) -> None:
        self.data += data

    def write_eof(self) -> None:
        self.eof = True


class FakeProcess:
    def __init__(self, stdout: bytes, exit_status: int | None, *, stderr=b"", hang=False):
        self.stdin = _Writer()
        self.stdout = _Reader(stdout, hang=hang)
        self.stderr = _Reader(stderr)
        self.exit_status = exit_status
        self.closed = False

    async def wait_closed(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, process: FakeProcess | None = None, *, raises: Exception | None = None):
        self.process = process
        self.raises = raises
        self.calls: list[tuple[str, dict]] = []

    async def create_process(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if self.raises is not None:
            raise self.raises
        return self.process


@pytest.fixture
def keypair():
    return bittensor.Keypair.create_from_uri("//LiumdClientTest")


@pytest.fixture
def intent():
    return build_intent(
        executor_uuid="exec-1",
        miner_hotkey="5Miner",
        matmul=None,
        verifyx=VerifyXStep(seed=5, cipher_text="vx"),
        parallel_gpu=False,
        deadline_s=60,
    )


def _answer(intent) -> bytes:
    return json.dumps(
        {
            "schema": SCHEMA,
            "nonce": intent["nonce"],
            "executor_uuid": intent["executor_uuid"],
            "executor_version": "liumd/0.1.0",
            "elapsed_ms": 12,
            "deadline_hit": False,
            "steps": {"verifyx": {"status": "ok", "ms": 10, "stdout": "vx-ok", "exit_status": 0}},
        }
    ).encode()


def _refusal(intent, error: str, *, echo: bool = True) -> bytes:
    doc = {"schema": SCHEMA, "error": error, "detail": "why " * 200}
    if echo:
        doc.update(nonce=intent["nonce"], executor_uuid=intent["executor_uuid"])
    return json.dumps(doc).encode()


async def _run(keypair, intent, conn, timeout_s=5):
    return await LiumdExecClient(keypair, timeout_s=timeout_s).run(conn, intent)


@pytest.mark.asyncio
async def test_exit_0_is_the_verify_result_through_parse_answer(keypair, intent):
    process = FakeProcess(_answer(intent), 0, stderr=b"liumd run: admitted\n")
    conn = FakeConnection(process)

    answer = await _run(keypair, intent, conn)

    assert isinstance(answer, LocalVerifyAnswer)
    assert answer.executor_version == "liumd/0.1.0"
    assert answer.step("verifyx").stdout == "vx-ok"
    sent = json.loads(process.stdin.data)
    assert sent["nonce"] == intent["nonce"] and sent["signature"].startswith("0x")
    assert process.stdin.eof and process.closed


@pytest.mark.asyncio
async def test_the_exec_request_is_the_command_alone(keypair, intent):
    conn = FakeConnection(FakeProcess(_answer(intent), 0))

    await _run(keypair, intent, conn)

    [(command, kwargs)] = conn.calls
    assert command == LIUMD_COMMAND == "/usr/local/bin/liumd run"
    # {} and [] replace the connection's defaults; () would inherit them.
    assert kwargs["env"] == {} and kwargs["send_env"] == []
    assert kwargs["request_pty"] is False
    assert "LIUMD_MINER_HOTKEY" not in json.dumps(kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_status,error",
    [(s, e) for s, errors in sorted(REFUSAL_ERRORS_BY_EXIT.items()) for e in sorted(errors)],
)
async def test_exits_2_4_5_6_are_the_refusal_document(keypair, intent, exit_status, error):
    conn = FakeConnection(FakeProcess(_refusal(intent, error), exit_status))

    refusal = await _run(keypair, intent, conn)

    assert isinstance(refusal, LiumdRefusal)
    assert (refusal.exit_status, refusal.error, refusal.echoed) == (exit_status, error, True)
    assert len(refusal.detail) == 300


@pytest.mark.asyncio
async def test_a_refusal_code_outside_its_exit_status_is_not_echoed_into_labels(keypair, intent):
    conn = FakeConnection(FakeProcess(_refusal(intent, "busy", echo=False), 4))

    refusal = await _run(keypair, intent, conn)

    assert (refusal.error, refusal.echoed) == ("unexpected_error", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", [{"nonce": "other"}, {"executor_uuid": "other"}])
async def test_a_refusal_for_another_intent_is_not_echoed(keypair, intent, echo):
    doc = {**json.loads(_refusal(intent, "busy")), **echo}
    conn = FakeConnection(FakeProcess(json.dumps(doc).encode(), 5))

    refusal = await _run(keypair, intent, conn)

    assert (refusal.error, refusal.echoed) == ("busy", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_status", [0, 2, 6])
async def test_a_document_that_is_not_json_is_malformed(keypair, intent, exit_status):
    conn = FakeConnection(FakeProcess(b"{not json", exit_status))

    with pytest.raises(LocalVerifyUnavailable) as err:
        await _run(keypair, intent, conn)

    assert err.value.reason == "malformed"


@pytest.mark.asyncio
async def test_a_refusal_that_is_not_an_object_is_malformed(keypair, intent):
    conn = FakeConnection(FakeProcess(b"[1, 2]", 5))

    with pytest.raises(LocalVerifyUnavailable) as err:
        await _run(keypair, intent, conn)

    assert err.value.reason == "malformed"


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_status", [126, 127])
async def test_126_and_127_mean_no_liumd_here(keypair, intent, exit_status):
    conn = FakeConnection(FakeProcess(b"", exit_status, stderr=b"sh: liumd: not found\n"))

    with pytest.raises(LocalVerifyUnavailable) as err:
        await _run(keypair, intent, conn)

    assert err.value.reason == "not_supported"


@pytest.mark.asyncio
async def test_a_channel_that_does_not_open_means_no_liumd_here(keypair, intent):
    conn = FakeConnection(raises=asyncssh.ChannelOpenError(4, "Session refused"))

    with pytest.raises(LocalVerifyUnavailable) as err:
        await _run(keypair, intent, conn)

    assert err.value.reason == "not_supported"


@pytest.mark.asyncio
async def test_a_dead_session_is_a_transport_error(keypair, intent):
    conn = FakeConnection(raises=asyncssh.ConnectionLost("gone"))

    with pytest.raises(LocalVerifyUnavailable) as err:
        await _run(keypair, intent, conn)

    assert err.value.reason == "transport"


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_status,reason", [(1, "unexpected_exit"), (None, "transport")])
async def test_any_other_exit_is_not_an_answer(keypair, intent, exit_status, reason):
    conn = FakeConnection(FakeProcess(_answer(intent), exit_status, stderr=b"panicked\n"))

    with pytest.raises(LocalVerifyUnavailable) as err:
        await _run(keypair, intent, conn)

    assert err.value.reason == reason


@pytest.mark.asyncio
async def test_an_answer_of_exactly_the_cap_is_read(keypair, intent):
    body = _answer(intent)
    padded = body[:-1] + b" " * (MAX_ANSWER_BYTES - len(body)) + b"}"
    assert len(padded) == MAX_ANSWER_BYTES

    answer = await _run(keypair, intent, FakeConnection(FakeProcess(padded, 0)))

    assert isinstance(answer, LocalVerifyAnswer)


@pytest.mark.asyncio
async def test_one_byte_over_the_cap_stops_reading_and_closes_the_channel(keypair, intent):
    body = _answer(intent)
    oversized = body[:-1] + b" " * (MAX_ANSWER_BYTES + 1 - len(body)) + b"}" + b" " * (1 << 22)
    process = FakeProcess(oversized, 0, hang=True)

    with pytest.raises(LocalVerifyUnavailable) as err:
        await _run(keypair, intent, FakeConnection(process))

    assert err.value.reason == "malformed"
    assert process.closed
    # Read in 64 KiB chunks until one byte past the cap, never the 4 MiB behind it.
    assert process.stdout.reads == MAX_ANSWER_BYTES // (64 * 1024) + 1


@pytest.mark.asyncio
async def test_a_host_that_never_answers_times_out_and_closes_the_channel(keypair, intent):
    process = FakeProcess(b"", None, hang=True)

    with pytest.raises(LocalVerifyUnavailable) as err:
        await asyncio.wait_for(_run(keypair, intent, FakeConnection(process), timeout_s=0.05), 5)

    assert err.value.reason == "timeout"
    assert process.closed


@pytest.mark.asyncio
async def test_a_channel_open_that_hangs_times_out(keypair, intent):
    class _Hanging(FakeConnection):
        async def create_process(self, command, **kwargs):
            await asyncio.Event().wait()

    with pytest.raises(LocalVerifyUnavailable) as err:
        await asyncio.wait_for(_run(keypair, intent, _Hanging(), timeout_s=0.05), 5)

    assert err.value.reason == "timeout"
