"""The renter CVM's attest-agent: the one thing the platform can talk to inside a rental.

FR-E6 states its whole job — "one small Lium agent whose only job is to answer trust-check
requests … it gives the platform no access to the customer's workload". So this service has
exactly two endpoints, both read-only, and no way to reach the workload: no docker socket,
no shared volumes, no exec. What it cannot do is as much of the design as what it can.

It ships **through the measured customer compose**, not baked into the OS image. Two
consequences, both wanted: the agent can be updated without minting a new image hash and
re-approving it fleet-wide, and the customer can read exactly what runs beside their
workload — it is a service in the compose they submitted, not something invisible.

    POST /v1/attest {"nonce": "<64 hex>"}   a fresh TDX quote + GPU evidence over that nonce
    GET  /health                            liveness, and the identity a verifier pins

The nonce ledger makes the agent **idempotent, not amnesiac**. Asked twice with the same
nonce it returns the identical answer; it never produces two different quotes for one
challenge. Refusing a repeat outright would break a verifier's retry after a network
timeout, and rejecting a *reused* nonce is the verifier's half of the protocol anyway —
only it knows which nonces it issued and when.
"""

import logging
import threading
from collections import OrderedDict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from attest_agent import __version__
from attest_agent.config import Config
from attest_agent.evidence import (
    EvidenceError,
    collect_gpu_evidence,
    gpu_uuids,
    quote_digest,
    tdx_quote,
)
from attest_agent.identity import IdentityError, gpu_uuid_digest, parse_nonce, report_data

logger = logging.getLogger(__name__)

NONCE_HEX_CHARS = 64

# What a failing request is told. Authored here rather than taken from the exception that
# caused it: this agent answers anyone who can reach its port, and an exception's text is a
# view of the guest's internals — SDK import paths, the dstack socket's own error, pydantic's
# reading of the body (CodeQL `py/stack-trace-exposure`). Each one is logged in full first,
# so nothing is lost to the operator; only the caller's copy is narrowed.
#
# Narrowing costs the caller nothing here. There is exactly one way to make a bad request —
# the model has one field with one constraint — and the other two are host conditions no
# caller can act on beyond the status code they already carry.
BAD_REQUEST_DETAIL = 'the body must be {"nonce": "<64 hex characters>"}'
NO_IDENTITY_DETAIL = "this agent cannot state its identity, so it cannot be attested"
NO_QUOTE_DETAIL = "this agent could not obtain a TDX quote from the dstack guest agent"


class AttestRequest(BaseModel):
    nonce: str = Field(min_length=NONCE_HEX_CHARS, max_length=NONCE_HEX_CHARS)


class NonceLedger:
    """Remembers what was answered for each nonce, bounded.

    Bounded because it is fed by whoever can reach the agent, and an unbounded map keyed on
    caller-supplied input is memory the caller controls. Oldest-out: a nonce old enough to
    have been evicted is one the verifier has long since stopped waiting on, and a repeat of
    it produces a fresh quote — which is correct, just not free.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._answers: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, nonce: str) -> dict | None:
        with self._lock:
            answer = self._answers.get(nonce)
            if answer is not None:
                self._answers.move_to_end(nonce)
            return answer

    def put(self, nonce: str, answer: dict) -> None:
        with self._lock:
            self._answers[nonce] = answer
            self._answers.move_to_end(nonce)
            while len(self._answers) > self._capacity:
                self._answers.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._answers)


def create_app(config: Config) -> FastAPI:
    identity = config.tls_identity()
    app = FastAPI(title="lium-attest-agent", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.config = config
    app.state.identity = identity
    app.state.ledger = NonceLedger(config.nonce_cache_size)

    @app.get("/health")
    async def health() -> dict:
        """Liveness, plus the identity a verifier pins before it ever issues a challenge.

        The GPU list is read live rather than cached: a CVM whose passthrough changed under
        it must not keep reporting the set it started with, because that set is half of what
        every quote attests to.
        """
        uuids, detail = gpu_uuids()
        return {
            "version": __version__,
            "tls_public_key": identity.public_key_hex,
            "gpu_uuids": uuids,
            "gpu_uuid_digest": gpu_uuid_digest(uuids).hex(),
            "gpu_detail": detail,
        }

    @app.post("/v1/attest")
    async def attest(body: dict) -> JSONResponse:
        try:
            request = AttestRequest.model_validate(body)
            nonce = parse_nonce(request.nonce)
        except (ValidationError, IdentityError) as exc:
            logger.info("422: %s", exc)
            return JSONResponse(status_code=422, content={"detail": BAD_REQUEST_DETAIL})

        cached = app.state.ledger.get(request.nonce)
        if cached is not None:
            # The same answer, not a new quote. See the ledger's docstring.
            return JSONResponse(status_code=200, content=cached)

        uuids, gpu_detail = gpu_uuids()
        try:
            data = report_data(identity.public_key_der, uuids, nonce)
        except IdentityError as exc:
            logger.error("cannot bind report_data to this CVM's identity: %s", exc)
            return JSONResponse(status_code=500, content={"detail": NO_IDENTITY_DETAIL})

        try:
            quote = tdx_quote(data, timeout=config.quote_timeout_seconds)
        except EvidenceError as exc:
            # 503, not 500: the guest agent being unreachable is a condition of this host
            # right now, and a verifier that retries later may well succeed.
            logger.error("no quote: %s", exc)
            return JSONResponse(status_code=503, content={"detail": NO_QUOTE_DETAIL})

        gpu = collect_gpu_evidence(uuids, request.nonce)
        answer = {
            "version": __version__,
            "report_data": data.hex(),
            "tls_public_key": identity.public_key_hex,
            "gpu_uuids": gpu.uuids,
            "gpu_uuid_digest": gpu_uuid_digest(uuids).hex(),
            "gpu_detail": gpu_detail,
            "quote": quote,
            "gpu_evidence": gpu.evidence,
            "gpu_evidence_detail": gpu.detail,
        }
        app.state.ledger.put(request.nonce, answer)
        logger.info(
            "answered a trust check: quote %s over %d GPU(s)", quote_digest(quote), len(uuids)
        )
        return JSONResponse(status_code=200, content=answer)

    logger.info(
        "attest-agent %s ready on %s:%d, TLS key %s…",
        __version__,
        config.host,
        config.port,
        identity.public_key_hex[:16],
    )
    return app
