"""Routes.

Handlers read the body from `request.state.parsed_body` — the object the middleware already
parsed and the scope check already evaluated. They never re-read the raw bytes: a second parse
only has to disagree with the first once for the scope check and the handler to act on different
values.

`POST {kind: "validation"}` and `DELETE` do real work as of DAH-2576. `POST {kind: "renter"}`
still answers 501: the renter body is DAH-2580's to define, and validating it against a
validation-shaped model now would be inventing its contract.

Both mutating handlers run the manager on a worker thread. A launch waits minutes for a guest
to boot, and holding the event loop for that would make `/v1/state` unanswerable for exactly
the period during which someone wants to read it.
"""

import asyncio
import logging
import threading

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from cvmd import __version__
from cvmd.cvm.manager import KIND_RENTER, KIND_VALIDATION, CvmManager, LaunchFailure, Triple
from cvmd.state.store import StateStore

logger = logging.getLogger(__name__)

router = APIRouter()

RENTER_NOT_IMPLEMENTED = "renter CVM provisioning lands in DAH-2580"

HEX64 = r"^[0-9a-f]{64}$"


class CreateCvmRequest(BaseModel):
    """The create body: which stack to run, never how big to make it.

    The pinned triple is required, not defaulted. A host that chose the stack itself would give
    the validator no way to tell "the CVM I asked for" from "whatever this host felt like
    running" — and detecting exactly that is why the triple is attested at all.

    Sizing (vCPUs, memory, disk, GPUs) is absent by design: it is provider configuration read
    from `/etc/cvmd/config.toml`, settled while reviewing DAH-2575.
    """

    kind: str
    qemu: str = Field(min_length=1)
    os_image_hash: str = Field(pattern=HEX64)
    compose_hash: str = Field(pattern=HEX64)


def _store(request: Request) -> StateStore:
    return request.app.state.store


def _manager(request: Request) -> CvmManager:
    return request.app.state.cvm


def _lock(request: Request) -> threading.Lock:
    return request.app.state.cvm_lock


def _failure(exc: LaunchFailure) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"detail": exc.reason})


@router.get("/health")
async def health() -> dict:
    return {"version": __version__}


@router.get("/v1/state")
async def get_state(request: Request) -> dict:
    return {**_store(request).document.to_json(), "cvm": _manager(request).describe()}


@router.post("/v1/cvm")
async def create_cvm(request: Request) -> JSONResponse:
    body = request.state.parsed_body

    # Checked before the model so the renter path keeps answering 501 rather than 422 for a
    # body whose required fields DAH-2580 has not defined yet.
    if isinstance(body, dict) and body.get("kind") == KIND_RENTER:
        return JSONResponse(status_code=501, content={"detail": RENTER_NOT_IMPLEMENTED})

    try:
        parsed = CreateCvmRequest.model_validate(body)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": exc.errors(include_url=False)})

    manager = _manager(request)
    lock = _lock(request)
    # Non-blocking on purpose: a second launch arriving mid-launch is refused, not queued.
    # Queueing would leave the caller waiting on a node that is already committed, and the
    # answer it needs — "not here, not now" — is available immediately.
    if not lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409,
            content={"detail": "another CVM operation is already in progress on this node"},
        )
    try:
        triple = Triple(
            qemu=parsed.qemu,
            os_image_hash=parsed.os_image_hash,
            compose_hash=parsed.compose_hash,
        )
        report = await asyncio.to_thread(manager.create, kind=KIND_VALIDATION, triple=triple)
    except LaunchFailure as exc:
        logger.warning("launch refused (%d): %s", exc.status, exc.reason)
        return _failure(exc)
    finally:
        lock.release()
    return JSONResponse(status_code=201, content=report)


@router.delete("/v1/cvm")
async def destroy_cvm(request: Request) -> JSONResponse:
    manager = _manager(request)
    lock = _lock(request)
    if not lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409,
            content={"detail": "another CVM operation is already in progress on this node"},
        )
    try:
        report = await asyncio.to_thread(manager.destroy)
    except LaunchFailure as exc:
        logger.error("teardown failed (%d): %s", exc.status, exc.reason)
        return _failure(exc)
    finally:
        lock.release()
    return JSONResponse(status_code=200, content=report)
