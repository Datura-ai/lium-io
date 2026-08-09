"""Routes.

Handlers read the body from `request.state.parsed_body` — the object the middleware already
parsed and the scope check already evaluated. They never re-read the raw bytes: a second parse
only has to disagree with the first once for the scope check and the handler to act on different
values.

`POST {kind: "validation"}` and `DELETE` do real work as of DAH-2576. `POST {kind: "renter"}`
still answers 501: the renter body is DAH-2580's to define, and validating it against a
validation-shaped model now would be inventing its contract.

Both mutating handlers run the manager on a worker thread, and both are slow on purpose. A
launch waits for the guest to boot; a `DELETE` waits for the node's hardware to actually come
back, which on a large-memory guest is longer still. Holding the event loop for either would
make `/v1/state` unanswerable for exactly the period during which someone wants to read it.

`DELETE` is the FRD's teardown endpoint. Its 200 means the four conditions in `cvm/release.py`
held together; a teardown that stopped the CVM but left the node holding hardware answers 504
naming the conditions that did not.

The two `/v1/catalog` routes are DAH-2578's. Both are open to either authorized key, because
neither decides what this host may run — the platform's signature does, and it is checked on
every read of the cached manifest.
"""

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from cvmd import __version__
from cvmd.catalog import HEX64, CatalogStore, refresh_once
from cvmd.cvm.manager import KIND_RENTER, KIND_VALIDATION, CvmManager, LaunchFailure, Triple
from cvmd.state.store import StateStore

logger = logging.getLogger(__name__)

router = APIRouter()

RENTER_NOT_IMPLEMENTED = "renter CVM provisioning lands in DAH-2580"


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


def _catalog(request: Request) -> CatalogStore:
    return request.app.state.catalog


def _failure(exc: LaunchFailure) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"detail": exc.reason})


async def _exclusively(request: Request, operation, *, success: int) -> JSONResponse:
    """Run one CVM operation on a worker thread, at most one at a time.

    The lock is non-blocking on purpose: a second operation arriving mid-launch is refused, not
    queued. Queueing would leave the caller waiting on a node that is already committed, and the
    answer it needs — "not here, not now" — is available immediately.
    """
    lock = request.app.state.cvm_lock
    if not lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409,
            content={"detail": "another CVM operation is already in progress on this node"},
        )
    try:
        report = await asyncio.to_thread(operation)
    except LaunchFailure as exc:
        logger.warning("CVM operation refused (%d): %s", exc.status, exc.reason)
        return _failure(exc)
    finally:
        lock.release()
    return JSONResponse(status_code=success, content=report)


@router.get("/health")
async def health() -> dict:
    return {"version": __version__}


@router.get("/v1/state")
async def get_state(request: Request) -> dict:
    manager = _manager(request)
    return {
        **_store(request).document.to_json(),
        "cvm": manager.describe(),
        # The node's last crossing between modes, with the per-condition timings a caller needs
        # to tell "still switching" from "offline" (FR-I3). Present whether or not a CVM is.
        "last_switch": manager.last_switch(),
    }


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

    triple = Triple(
        qemu=parsed.qemu,
        os_image_hash=parsed.os_image_hash,
        compose_hash=parsed.compose_hash,
    )
    manager = _manager(request)
    return await _exclusively(
        request,
        lambda: manager.create(kind=KIND_VALIDATION, triple=triple),
        success=201,
    )


@router.delete("/v1/cvm")
async def destroy_cvm(request: Request) -> JSONResponse:
    return await _exclusively(request, _manager(request).destroy, success=200)


@router.get("/v1/catalog")
async def get_catalog(request: Request) -> dict:
    """What this host is approved to launch, and where that answer came from.

    A read, so either authorized key may make it. It answers 200 even when the catalog is
    unusable, carrying `usable: false` and the reason — a node that will not launch is exactly
    when someone needs to know *why*, and a 503 here would only tell them that something is
    wrong with a daemon that is plainly answering.
    """
    return _catalog(request).describe()


@router.post("/v1/catalog/refresh")
async def refresh_catalog(request: Request) -> JSONResponse:
    """Fetch the manifest now instead of waiting for the timer.

    This is how a revocation propagates in seconds rather than in `catalog_refresh_seconds`.
    Triggering it grants no authority over *what* the host will run: the fetched bytes go
    through exactly the same checks as the timer's, so the worst a caller can do is make the
    host re-read the platform's own catalog.

    On a worker thread for the same reason the CVM operations are — the fetch and the signature
    check both block, and `/v1/state` has to stay answerable.
    """
    store = _catalog(request)
    manifest = await asyncio.to_thread(refresh_once, store)
    return JSONResponse(
        status_code=200,
        content={"fetched": manifest is not None, "catalog": store.describe()},
    )
