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
    MAX_STDERR_BYTES,
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
@pytest.mark.parametrize(
    "doc,exit_status,expected",
    [
        ({"error": "busy"}, 4, "unexpected_error"),
        ({"nonce": "other"}, 5, "busy"),
        ({"executor_uuid": "other"}, 5, "busy"),
    ],
    ids=["code-outside-its-exit", "other-nonce", "other-executor"],
)
async def test_a_refusal_not_for_this_intent_is_not_echoed(
    keypair, intent, doc, exit_status, expected
):
    echo = "error" in doc  # an unechoed document carries no nonce at all
    body = {**json.loads(_refusal(intent, "busy", echo=not echo)), **doc}
    conn = FakeConnection(FakeProcess(json.dumps(body).encode(), exit_status))

    refusal = await _run(keypair, intent, conn)

    assert (refusal.error, refusal.echoed) == (expected, False)


def _unavailable_cases(intent):
    return [
        ("malformed", FakeConnection(FakeProcess(b"{not json", 0))),
        ("malformed", FakeConnection(FakeProcess(b"{not json", 2))),
        ("malformed", FakeConnection(FakeProcess(b"[1, 2]", 5))),
        ("not_supported", FakeConnection(FakeProcess(b"", 126, stderr=b"sh: liumd: not found\n"))),
        ("not_supported", FakeConnection(FakeProcess(b"", 127, stderr=b"sh: liumd: not found\n"))),
        ("not_supported", FakeConnection(raises=asyncssh.ChannelOpenError(4, "Session refused"))),
        ("transport", FakeConnection(raises=asyncssh.ConnectionLost("gone"))),
        ("unexpected_exit", FakeConnection(FakeProcess(_answer(intent), 1, stderr=b"panicked\n"))),
        ("transport", FakeConnection(FakeProcess(_answer(intent), None, stderr=b"panicked\n"))),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", range(9))
async def test_anything_but_an_answer_or_refusal_is_unavailable(keypair, intent, case):
    reason, conn = _unavailable_cases(intent)[case]

    with pytest.raises(LocalVerifyUnavailable) as err:
        await _run(keypair, intent, conn)

    assert err.value.reason == reason


@pytest.mark.asyncio
@pytest.mark.parametrize("over", [0, 1], ids=["exactly-the-cap", "one-byte-over"])
async def test_an_answer_is_read_up_to_the_cap_and_no_further(keypair, intent, over):
    body = _answer(intent)
    padded = body[:-1] + b" " * (MAX_ANSWER_BYTES + over - len(body)) + b"}"
    # Past the cap there is 4 MiB more, behind a stream that then hangs.
    process = FakeProcess(padded + b" " * (1 << 22 if over else 0), 0, hang=bool(over))

    if not over:
        assert isinstance(await _run(keypair, intent, FakeConnection(process)), LocalVerifyAnswer)
        return
    with pytest.raises(LocalVerifyUnavailable) as err:
        await _run(keypair, intent, FakeConnection(process))
    assert err.value.reason == "malformed"
    assert process.closed
    # Read in 64 KiB chunks until one byte past the cap, never the 4 MiB behind it.
    assert process.stdout.reads == MAX_ANSWER_BYTES // (64 * 1024) + 1


@pytest.mark.asyncio
async def test_endless_stderr_is_cut_off_and_closes_the_channel(keypair, intent):
    process = FakeProcess(_answer(intent), 0, stderr=b"x" * (MAX_STDERR_BYTES + 1))

    with pytest.raises(LocalVerifyUnavailable) as err:
        await _run(keypair, intent, FakeConnection(process))

    assert err.value.reason == "malformed"
    assert process.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("hang_at", ["answer", "channel-open"])
async def test_a_host_that_never_answers_times_out_and_closes_the_channel(keypair, intent, hang_at):
    process = FakeProcess(b"", None, hang=True)
    conn = FakeConnection(process)
    if hang_at == "channel-open":

        async def hang(command, **kwargs):
            await asyncio.Event().wait()

        conn.create_process = hang

    with pytest.raises(LocalVerifyUnavailable) as err:
        await asyncio.wait_for(_run(keypair, intent, conn, timeout_s=0.05), 5)

    assert err.value.reason == "timeout"
    assert process.closed == (hang_at == "answer")
