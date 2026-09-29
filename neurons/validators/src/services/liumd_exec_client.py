"""Validator side of `liumd run` over an SSH exec channel.

The executor image ships `liumd` at `LIUMD_COMMAND`. On the SSH session the pipeline already holds
(`Context.ssh`), this client opens one exec channel running `/usr/local/bin/liumd run` — no
arguments, no environment, no pty — writes the signed intent (`build_intent` + `sign_intent`, the
same document `POST /verify` takes) to its stdin, closes stdin, and reads stdout to EOF, at most
`MAX_ANSWER_BYTES`, plus the exit status. stderr carries log lines only and is drained, not parsed.

The exit status says what stdout is:

- 0: the `VerifyResult` document, read by `parse_answer` exactly as the HTTP path reads it;
- 2, 4, 5, 6: one refusal document (`LiumdRefusal`): 2 = invalid_json | invalid_intent,
  4 = bad_signature | intent_refused, 5 = nonce_replayed | busy, 6 = agent_error;
- 126, 127, or a channel that does not open: no liumd on this host (`not_supported`), so the
  caller keeps the SSH path.

Anything else raises `LocalVerifyUnavailable(reason)`. The host's miner hotkey is the host's own
setting (`/etc/liumd/miner_hotkey`); the validator never sends `LIUMD_MINER_HOTKEY` or any other
variable, so nothing on this side can change what the host checks the intent against.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import asyncssh

from services.local_verify_client import (
    DETAIL_MAX_CHARS,
    MAX_ANSWER_BYTES,
    LocalVerifyAnswer,
    LocalVerifyUnavailable,
    parse_answer,
    sign_intent,
)

LIUMD_COMMAND = "/usr/local/bin/liumd run"
# The refusal codes each exit status may carry (lium.local_verify/1's ErrorCode). Anything else in
# a refusal document is reported as `unexpected_error`, so a log label never carries peer text.
REFUSAL_ERRORS_BY_EXIT: dict[int, frozenset[str]] = {
    2: frozenset({"invalid_json", "invalid_intent"}),
    4: frozenset({"bad_signature", "intent_refused"}),
    5: frozenset({"nonce_replayed", "busy"}),
    6: frozenset({"agent_error"}),
}
NOT_SUPPORTED_EXITS = frozenset({126, 127})
STDERR_TAIL_BYTES = 4 * 1024
READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class LiumdRefusal:
    """`liumd run` refused the intent: which exit status, which code, and its capped detail."""

    exit_status: int
    error: str
    detail: str
    # Whether the document echoed this intent's nonce and executor (liumd echoes them whenever
    # the intent carried both, even for a refusal).
    echoed: bool
    round_trip_ms: int


@dataclass(frozen=True)
class _Finished:
    exit_status: int | None
    stdout: bytes
    oversized: bool
    stderr_tail: bytes


async def _read_capped(reader, limit: int) -> tuple[bytes, bool]:
    """Up to `limit` bytes of `reader` and whether it had more. Stops reading past the cap."""
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await reader.read(READ_CHUNK_BYTES)
        if not chunk:
            return b"".join(chunks), False
        chunks.append(chunk)
        size += len(chunk)
        if size > limit:
            return b"".join(chunks)[:limit], True


async def _drain_tail(reader, keep: int) -> bytes:
    """Read `reader` to EOF, keeping the last `keep` bytes: the channel's flow control stalls the
    remote side when a stream nobody reads fills its window."""
    tail = b""
    while True:
        chunk = await reader.read(READ_CHUNK_BYTES)
        if not chunk:
            return tail
        tail = (tail + chunk)[-keep:]


def _text(data: bytes, limit: int = DETAIL_MAX_CHARS) -> str:
    return data.decode("utf-8", errors="replace")[:limit]


class LiumdExecClient:
    """One `liumd run` per call on the pipeline's SSH connection."""

    def __init__(self, keypair, *, timeout_s: float):
        self.keypair = keypair
        self.timeout_s = timeout_s

    async def run(
        self, ssh: asyncssh.SSHClientConnection, intent: dict[str, Any]
    ) -> LocalVerifyAnswer | LiumdRefusal:
        signed = sign_intent(intent, self.keypair)
        started = time.perf_counter()
        try:
            finished = await asyncio.wait_for(self._exchange(ssh, signed), self.timeout_s)
        except TimeoutError:
            raise LocalVerifyUnavailable("timeout", f"no exit status within {self.timeout_s}s")
        round_trip_ms = int((time.perf_counter() - started) * 1000)
        return self._interpret(finished, intent, round_trip_ms)

    async def _exchange(
        self, ssh: asyncssh.SSHClientConnection, signed: dict[str, Any]
    ) -> _Finished:
        try:
            # env={} and send_env=[] override whatever the connection's options would send;
            # () would mean "the connection's defaults".
            process = await ssh.create_process(
                LIUMD_COMMAND,
                env={},
                send_env=[],
                request_pty=False,
                encoding=None,
            )
        except asyncssh.ChannelOpenError as exc:
            raise LocalVerifyUnavailable("not_supported", f"exec channel refused: {exc.reason}")
        except (asyncssh.Error, OSError) as exc:
            raise LocalVerifyUnavailable("transport", f"{type(exc).__name__}: {exc}")
        try:
            process.stdin.write(json.dumps(signed).encode())
            process.stdin.write_eof()
            stderr_task = asyncio.ensure_future(_drain_tail(process.stderr, STDERR_TAIL_BYTES))
            try:
                stdout, oversized = await _read_capped(process.stdout, MAX_ANSWER_BYTES)
                if oversized:
                    return _Finished(None, stdout, True, b"")
                await process.wait_closed()
                stderr_tail = await stderr_task
            finally:
                stderr_task.cancel()
            return _Finished(process.exit_status, stdout, False, stderr_tail)
        except (asyncssh.Error, OSError, BrokenPipeError) as exc:
            raise LocalVerifyUnavailable("transport", f"{type(exc).__name__}: {exc}")
        finally:
            # Closing the channel on every way out: an oversized or timed-out answer must not keep
            # it (or the remote process's stdout) open.
            process.close()

    @staticmethod
    def _interpret(
        finished: _Finished, intent: dict[str, Any], round_trip_ms: int
    ) -> LocalVerifyAnswer | LiumdRefusal:
        if finished.oversized:
            raise LocalVerifyUnavailable(
                "malformed", f"answer longer than {MAX_ANSWER_BYTES} bytes"
            )
        status = finished.exit_status
        if status is None:
            raise LocalVerifyUnavailable("transport", "channel closed without an exit status")
        if status in NOT_SUPPORTED_EXITS:
            raise LocalVerifyUnavailable("not_supported", f"exit {status}: no liumd on this host")
        if status == 0:
            try:
                raw = json.loads(finished.stdout)
            except ValueError:
                raise LocalVerifyUnavailable("malformed", "answer is not JSON")
            return parse_answer(raw, intent=intent, round_trip_ms=round_trip_ms)
        if status in REFUSAL_ERRORS_BY_EXIT:
            try:
                doc = json.loads(finished.stdout)
            except ValueError:
                raise LocalVerifyUnavailable("malformed", f"exit {status}: refusal is not JSON")
            if not isinstance(doc, dict):
                raise LocalVerifyUnavailable(
                    "malformed", f"exit {status}: refusal is not an object"
                )
            error = doc.get("error")
            detail = doc.get("detail")
            return LiumdRefusal(
                exit_status=status,
                error=error if error in REFUSAL_ERRORS_BY_EXIT[status] else "unexpected_error",
                detail=detail[:DETAIL_MAX_CHARS] if isinstance(detail, str) else "",
                echoed=(
                    doc.get("nonce") == intent["nonce"]
                    and doc.get("executor_uuid") == intent["executor_uuid"]
                ),
                round_trip_ms=round_trip_ms,
            )
        raise LocalVerifyUnavailable(
            "unexpected_exit", f"exit {status}: {_text(finished.stderr_tail)}"
        )
