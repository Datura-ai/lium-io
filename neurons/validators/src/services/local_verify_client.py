"""Validator side of the executor's one-call verification (`POST /verify`, liumd phase 1).

The executor that advertises `local_verify/1` in `GET /version` accepts one intent signed by this
validator's hotkey — nonce'd and time-windowed — and runs the matmul and VerifyX scripts locally,
side by side at first-pass sizes, returning their raw output in one document. This client signs,
sends, and checks that what came back is the answer to what was asked (schema, nonce, executor);
judging the steps is the checks' job, with the same functions the SSH path uses.

Anything that is not a well-formed answer raises `LocalVerifyUnavailable(reason)`; the caller
falls back to the SSH path and logs the reason. Nothing here can fail a node.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from datura.requests.validator_requests import (
    LOCAL_VERIFY_CAPABILITY,
    LOCAL_VERIFY_SCHEMA,
    MatmulStep,
    VerifyXStep,
    local_verify_signing_blob,
)

# The judged results are the SSH path's own types. Imported at runtime, not under TYPE_CHECKING:
# `Context` (pipeline.py) is a pydantic model and resolves `LocalVerifyOutcome`'s annotations
# when it is built — a name only the type checker sees leaves every cycle unbuildable
# (test_rented_machine_check.test_context_annotations_resolve_at_runtime).
from services.matrix_validation_service import ValidationResult
from services.verifyx_validation_service import VerifyXResponse

# One definition for both sides (datura, #1339): the executor's payloads/verify.py and
# local_verify_service.py import the same names, so the two ends cannot drift apart.
SCHEMA = LOCAL_VERIFY_SCHEMA
CAPABILITY = LOCAL_VERIFY_CAPABILITY
# The executor refuses an intent whose issued_at is more than its window (120 s default) from its
# clock and whose expiry is further than about two windows out; stay inside both.
INTENT_TTL_SECONDS = 120
# The executor's deadline clock starts after connect + signature check and its answer travels back
# after the deadline; the intent's `deadline_s` is therefore the client's whole-call timeout minus
# this margin, so a `deadline_hit` answer (finished steps inside) arrives before the client gives up.
EXECUTOR_DEADLINE_MARGIN_SECONDS = 30
# `VerifyIntentBody.deadline_s` on the executor: ge=5, le=3600 — anything outside is a 422.
EXECUTOR_DEADLINE_MIN_SECONDS = 5
EXECUTOR_DEADLINE_MAX_SECONDS = 3600


def executor_deadline_s(timeout_s: int) -> int:
    """The `deadline_s` to put in the intent for a client whole-call timeout of `timeout_s`,
    clamped to what the executor accepts."""
    wanted = int(timeout_s) - EXECUTOR_DEADLINE_MARGIN_SECONDS
    return max(EXECUTOR_DEADLINE_MIN_SECONDS, min(EXECUTOR_DEADLINE_MAX_SECONDS, wanted))


def _wire_int(value: Any) -> int:
    """An executor-reported count; anything that is not a plain int reads as 0 (the answer stays
    parseable — a bad number is not worth losing the other steps' evidence for)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


# The step names the intent can ask for; anything else in an answer is dropped, not echoed.
STEP_NAMES = ("matmul", "verifyx", "docker", "ports", "inspector")
# The statuses the executor's `StepResult` emits. Any other string is `malformed` here, so every
# `reason` label and event field built from a status comes from this closed set (PR_PROCESS §5:
# peer-controlled strings never reach a log label uncapped).
STEP_STATUSES = frozenset({"ok", "failed", "timeout", "skipped"})
EXECUTOR_VERSION_MAX_CHARS = 64
# Bounds on what is read from the executor before anything is parsed: the answer (five stdouts of
# ≤ 256 KB each on the executor's side) and the `/version` capability list.
MAX_ANSWER_BYTES = 2 * 1024 * 1024
MAX_CAPABILITIES = 32
MAX_CAPABILITY_CHARS = 64
DETAIL_MAX_CHARS = 300


async def _read_bounded(response, limit: int) -> bytes:
    """The body up to `limit` bytes; one byte more marks it oversized (the caller checks the
    length). `StreamReader.read(n)` may return short, so this collects until the cap or EOF."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.content.iter_chunked(64 * 1024):
        chunks.append(chunk)
        size += len(chunk)
        if size > limit:
            break
    return b"".join(chunks)


class LocalVerifyUnavailable(Exception):
    """The local path did not produce a usable answer; `reason` is the metric label."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# The signed bytes: the wire document minus `signature`, sorted keys, no whitespace. The executor
# rebuilds the same string from the request body it received — with this very function.
canonical_intent_message = local_verify_signing_blob


def build_intent(
    *,
    executor_uuid: str,
    matmul: MatmulStep | None,
    verifyx: VerifyXStep | None,
    parallel_gpu: bool,
    deadline_s: int,
    now: float | None = None,
) -> dict[str, Any]:
    """The intent document as signed and sent. The step challenges are the datura models the
    executor parses (`MatmulStep`, `VerifyXStep`), so a field the executor's `extra="forbid"`
    schema does not know cannot be built here; `docker`, `ports` and `inspector` are asked for so
    the facts the phase-2 checks read (#1345) come with the same call."""
    now = time.time() if now is None else now
    return {
        "schema": SCHEMA,
        "nonce": secrets.token_hex(16),
        "issued_at": int(now),
        "expires_at": int(now) + INTENT_TTL_SECONDS,
        "executor_uuid": executor_uuid,
        "deadline_s": deadline_s,
        "parallel_gpu": parallel_gpu,
        "steps": {
            "matmul": matmul.model_dump(exclude_none=True) if matmul is not None else None,
            "verifyx": verifyx.model_dump() if verifyx is not None else None,
            "docker": True,
            "ports": True,
            "inspector": True,
        },
    }


def sign_intent(intent: dict[str, Any], keypair) -> dict[str, Any]:
    signed = dict(intent)
    signed["signature"] = "0x" + keypair.sign(canonical_intent_message(intent)).hex()
    return signed


@dataclass
class LocalVerifyOutcome:
    """What the consuming checks read (`ctx.state.local_verify`). A field is set only when the
    local step ran AND passed the validator's judgement; None means "run it over SSH"."""

    matmul: ValidationResult | None = None
    verifyx: VerifyXResponse | None = None
    round_trip_ms: int = 0
    executor_elapsed_ms: int = 0
    executor_version: str = ""
    fallbacks: dict[str, str] = field(default_factory=dict)  # step -> reason


@dataclass
class StepEvidence:
    status: str
    ms: int = 0
    exit_status: int | None = None
    stdout: str | None = None
    stderr_tail: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_wire(cls, raw: Any) -> StepEvidence:
        if not isinstance(raw, dict) or not isinstance(raw.get("status"), str):
            return cls(status="malformed", error="step is not an object with a status")
        if raw["status"] not in STEP_STATUSES:
            return cls(status="malformed", error="unknown step status")
        exit_status = raw.get("exit_status")
        return cls(
            status=raw["status"],
            ms=_wire_int(raw.get("ms")),
            exit_status=exit_status if isinstance(exit_status, int) else None,
            stdout=raw.get("stdout") if isinstance(raw.get("stdout"), str) else None,
            stderr_tail=raw.get("stderr_tail") if isinstance(raw.get("stderr_tail"), str) else None,
            data=raw.get("data") if isinstance(raw.get("data"), dict) else {},
            error=raw.get("error") if isinstance(raw.get("error"), str) else None,
        )


@dataclass
class LocalVerifyAnswer:
    nonce: str
    executor_uuid: str
    executor_version: str
    elapsed_ms: int
    deadline_hit: bool
    steps: dict[str, StepEvidence]
    round_trip_ms: int

    def step(self, name: str) -> StepEvidence:
        return self.steps.get(name) or StepEvidence(status="skipped")


def parse_answer(raw: Any, *, intent: dict[str, Any], round_trip_ms: int) -> LocalVerifyAnswer:
    """The answer must be to THIS intent: same schema, nonce and executor, steps as objects."""
    if not isinstance(raw, dict):
        raise LocalVerifyUnavailable("malformed", "answer is not an object")
    if raw.get("schema") != SCHEMA:
        raise LocalVerifyUnavailable("schema_mismatch", f"got {raw.get('schema')!r}"[:100])
    if raw.get("nonce") != intent["nonce"]:
        raise LocalVerifyUnavailable("nonce_mismatch", "answer does not echo the intent nonce")
    if raw.get("executor_uuid") != intent["executor_uuid"]:
        raise LocalVerifyUnavailable("executor_mismatch", f"got {raw.get('executor_uuid')!r}"[:100])
    steps = raw.get("steps")
    if not isinstance(steps, dict):
        raise LocalVerifyUnavailable("malformed", "steps is not an object")
    return LocalVerifyAnswer(
        nonce=raw["nonce"],
        executor_uuid=raw["executor_uuid"],
        executor_version=str(raw.get("executor_version") or "")[:EXECUTOR_VERSION_MAX_CHARS],
        elapsed_ms=_wire_int(raw.get("elapsed_ms")),
        deadline_hit=bool(raw.get("deadline_hit")),
        steps={name: StepEvidence.from_wire(steps[name]) for name in STEP_NAMES if name in steps},
        round_trip_ms=round_trip_ms,
    )


class LocalVerifyClient:
    """HTTP to the executor's own API port (`executor_info.port`, where the miner and the backend
    already talk to it) — no SSH session, no miner hop."""

    def __init__(self, keypair, *, timeout_s: int, connect_timeout_s: int, session_factory=None):
        self.keypair = keypair
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        self._session_factory = session_factory or aiohttp.ClientSession

    @staticmethod
    def base_url(executor_info) -> str:
        return f"http://{executor_info.address}:{executor_info.port}"

    async def capabilities(self, executor_info) -> set[str]:
        """What the executor's `/version` advertises; empty on any error (an old image has none)."""
        timeout = aiohttp.ClientTimeout(
            total=self.connect_timeout_s * 2, connect=self.connect_timeout_s
        )
        try:
            async with self._session_factory(timeout=timeout) as session:
                async with session.get(
                    f"{self.base_url(executor_info)}/version", allow_redirects=False
                ) as response:
                    if response.status != 200:
                        return set()
                    body = json.loads(await _read_bounded(response, MAX_ANSWER_BYTES))
        except Exception:
            return set()
        caps = body.get("capabilities") if isinstance(body, dict) else None
        if not isinstance(caps, list):
            return set()
        # A closed-size set: the event that lists them is a log sink, not a place for a novel.
        return {
            c
            for c in caps[:MAX_CAPABILITIES]
            if isinstance(c, str) and len(c) <= MAX_CAPABILITY_CHARS
        }

    async def verify(self, executor_info, intent: dict[str, Any]) -> LocalVerifyAnswer:
        signed = sign_intent(intent, self.keypair)
        timeout = aiohttp.ClientTimeout(total=self.timeout_s, connect=self.connect_timeout_s)
        started = time.perf_counter()
        try:
            async with self._session_factory(timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url(executor_info)}/verify", json=signed, allow_redirects=False
                ) as response:
                    # A redirect would re-send the signed intent to a host of the executor's
                    # choosing: it is an http_error below. The body is bounded before it is parsed.
                    body = await _read_bounded(response, MAX_ANSWER_BYTES)
                    status = response.status
        except TimeoutError:
            raise LocalVerifyUnavailable("timeout", f"no answer within {self.timeout_s}s")
        except Exception as exc:  # aiohttp client errors, DNS, refused connections
            raise LocalVerifyUnavailable("transport", f"{type(exc).__name__}: {exc}")
        round_trip_ms = int((time.perf_counter() - started) * 1000)
        text = body[:MAX_ANSWER_BYTES].decode("utf-8", errors="replace")
        if status == 404:
            raise LocalVerifyUnavailable(
                "not_supported", "executor answered 404 (route absent or flag off)"
            )
        if status == 409:
            # The executor runs one suite at a time and refuses a nonce it has seen.
            raise LocalVerifyUnavailable("busy_or_replay", text[:200])
        if status == 401:
            raise LocalVerifyUnavailable("refused", text[:200])
        if status != 200:
            raise LocalVerifyUnavailable("http_error", f"status {status}: {text[:200]}")
        if len(body) > MAX_ANSWER_BYTES:
            raise LocalVerifyUnavailable(
                "malformed", f"answer longer than {MAX_ANSWER_BYTES} bytes"
            )
        try:
            raw = json.loads(text)
        except ValueError:
            raise LocalVerifyUnavailable("malformed", "answer is not JSON")
        return parse_answer(raw, intent=intent, round_trip_ms=round_trip_ms)
