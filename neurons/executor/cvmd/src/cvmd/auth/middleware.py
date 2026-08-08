"""The auth middleware.

Order is load-bearing:

    size cap -> read body -> verify signature -> replay check -> scope check

The size cap runs before the body is buffered *and* before the `/health` exemption. Without it
an unauthenticated caller can make cvmd hold an arbitrary amount of memory before a single check
runs — on a host whose whole job is running GPU workloads under memory pressure.

The replay check runs before dispatch and records durably before it returns, so a crash cannot
leave an already-executed request replayable. See replay.py.

Shaped like `neurons/executor/src/middlewares/miner.py`, but the bittensor v11 calls differ:
`bittensor.Keypair` was removed (it lives at `bittensor.sp_core.Keypair`) and `verify()` is
strictly bytes-typed, so the hex signature is decoded here rather than passed through.
"""

import json
import logging
from dataclasses import dataclass

from bittensor.sp_core import Keypair
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from cvmd.auth.blob import signing_blob
from cvmd.auth.clients import AuthorizedClients, Scope
from cvmd.auth.replay import ReplayRejected, ReplayStore
from cvmd.auth.scope import EITHER_KEY, HEALTH_PATH, BodyRejected, required_scope

logger = logging.getLogger(__name__)

HOTKEY_HEADER = "X-Cvmd-Hotkey"
TIMESTAMP_HEADER = "X-Cvmd-Timestamp"
NONCE_HEADER = "X-Cvmd-Nonce"
SIGNATURE_HEADER = "X-Cvmd-Signature"

# A nonce below this is not plausibly random. The signed blob covers it, and the replay store
# treats it as the uniqueness key, so a client sending a short or constant nonce is degrading a
# control it cannot see the effect of — reject it rather than accept a weak one.
MIN_NONCE_HEX_CHARS = 32

UNAUTHORIZED = {"detail": "unauthorized"}
FORBIDDEN = {"detail": "scope violation"}

# A body that arrived but is not JSON. Distinct from None (no body at all) so the scope rules can
# 422 the routes that need a body without inventing a shape for the ones that do not.
UNPARSEABLE = object()


class _Unauthorized(Exception):
    """Any auth failure. The reason goes to the journal; the client gets one 401."""


class _TooLarge(Exception):
    pass


@dataclass(frozen=True)
class _Authenticated:
    hotkey: str
    scope: Scope
    timestamp_ns: int
    nonce: str


async def _read_body_capped(request: Request, limit: int) -> bytes:
    """Read the body, aborting as soon as it exceeds `limit` rather than after buffering it."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limit:
                raise _TooLarge(f"content-length {declared} exceeds {limit}")
        except ValueError as exc:
            raise _TooLarge(f"malformed content-length {declared!r}") from exc

    # Content-Length is a claim, not a fact — a chunked body carries none at all. The stream
    # walk below is the actual enforcement.
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise _TooLarge(f"body exceeded {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _require_header(request: Request, name: str) -> str:
    value = request.headers.get(name)
    if not value:
        raise _Unauthorized(f"missing {name}")
    return value


def _verify_signature(hotkey: str, blob: bytes, signature: str) -> None:
    try:
        raw_signature = bytes.fromhex(signature.removeprefix("0x"))
    except ValueError as exc:
        raise _Unauthorized(f"signature is not hex: {exc}") from exc

    # v11's verify() rejects a str message or a hex-string signature with TypeError. Both
    # arguments are bytes by construction here; the try guards against a future regression
    # surfacing as a 500 instead of a 401.
    try:
        verified = Keypair(ss58_address=hotkey).verify(blob, raw_signature)
    except (ValueError, TypeError) as exc:
        raise _Unauthorized(f"signature verification failed: {exc}") from exc

    if not verified:
        raise _Unauthorized("signature does not match")


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        clients: AuthorizedClients,
        replay: ReplayStore,
        max_body_bytes: int,
    ) -> None:
        super().__init__(app)
        self._clients = clients
        self._replay = replay
        self._max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        try:
            body = await _read_body_capped(request, self._max_body_bytes)
        except _TooLarge as exc:
            logger.warning("413 %s %s: %s", request.method, path, exc)
            return JSONResponse(status_code=413, content={"detail": "request body too large"})

        # /health is exempt from auth so systemd and monitoring can probe liveness. It is not
        # exempt from the cap above.
        if path == HEALTH_PATH:
            return await call_next(request)

        try:
            client = self._authenticate(request, body)
        except _Unauthorized as exc:
            logger.warning("401 %s %s: %s", request.method, path, exc)
            return JSONResponse(status_code=401, content=UNAUTHORIZED)

        # Before dispatch, and durable before it returns. Both halves matter — see replay.py.
        try:
            await self._replay.check_and_record(client.hotkey, client.nonce, client.timestamp_ns)
        except ReplayRejected as exc:
            logger.warning("401 %s %s: %s", request.method, path, exc)
            return JSONResponse(status_code=401, content=UNAUTHORIZED)

        parsed_body = _parse_body(body)
        try:
            needed = required_scope(request.method, path, parsed_body)
        except BodyRejected as exc:
            logger.info("422 %s %s: %s", request.method, path, exc)
            return JSONResponse(status_code=422, content={"detail": str(exc)})

        if needed is not EITHER_KEY and client.scope is not needed:
            logger.warning(
                "403 %s %s: %s holds %s, route needs %s",
                request.method,
                path,
                client.hotkey,
                client.scope,
                needed,
            )
            return JSONResponse(status_code=403, content=FORBIDDEN)

        # Handlers read these. They must not re-read the raw bytes — one parse, one truth.
        request.state.hotkey = client.hotkey
        request.state.scope = client.scope
        request.state.parsed_body = parsed_body

        return await call_next(request)

    def _authenticate(self, request: Request, body: bytes) -> _Authenticated:
        hotkey = _require_header(request, HOTKEY_HEADER)
        timestamp = _require_header(request, TIMESTAMP_HEADER)
        nonce = _require_header(request, NONCE_HEADER)
        signature = _require_header(request, SIGNATURE_HEADER)

        scope = self._clients.scope_for(hotkey)
        if scope is None:
            raise _Unauthorized(f"{hotkey} is not an authorized client")

        try:
            timestamp_ns = int(timestamp)
        except ValueError as exc:
            raise _Unauthorized(f"{TIMESTAMP_HEADER} is not an integer: {timestamp!r}") from exc

        if len(nonce) < MIN_NONCE_HEX_CHARS:
            raise _Unauthorized(f"{NONCE_HEADER} is shorter than {MIN_NONCE_HEX_CHARS} chars")

        # request_target is path + query exactly as received: the query is attacker-controlled
        # input on DAH-2578's /v1/catalog, so it has to be inside the signature.
        target = request.url.path
        if request.url.query:
            target = f"{target}?{request.url.query}"

        blob = signing_blob(
            method=request.method,
            request_target=target,
            body=body,
            timestamp=timestamp,
            nonce=nonce,
        )
        _verify_signature(hotkey, blob, signature)
        return _Authenticated(hotkey=hotkey, scope=scope, timestamp_ns=timestamp_ns, nonce=nonce)


def _parse_body(body: bytes):
    """Parse the body exactly once. Empty body -> None; present-but-not-JSON -> UNPARSEABLE."""
    if not body:
        return None
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return UNPARSEABLE
