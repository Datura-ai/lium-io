"""Routes.

Handlers read the body from `request.state.parsed_body` — the object the middleware already
parsed and the scope check already evaluated. They never re-read the raw bytes: a second parse
only has to disagree with the first once for the scope check and the handler to act on different
values.

The two mutating routes are registered with full auth and scope enforcement but return 501. That
is deliberate: the scope matrix is testable end-to-end today, and DAH-2576 only fills in handlers
rather than also wiring auth.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from cvmd import __version__
from cvmd.state.store import StateStore

router = APIRouter()

NOT_IMPLEMENTED = "not implemented in DAH-2575 — CVM operations land in DAH-2576"


class CreateCvmRequest(BaseModel):
    """The create body. `kind` is what the middleware resolved the scope from; the pinned triple
    is carried through unvalidated here because DAH-2576 owns what a valid triple is.
    """

    kind: str


def _store(request: Request) -> StateStore:
    return request.app.state.store


@router.get("/health")
async def health() -> dict:
    return {"version": __version__}


@router.get("/v1/state")
async def get_state(request: Request) -> dict:
    return _store(request).document.to_json()


@router.post("/v1/cvm")
async def create_cvm(request: Request) -> JSONResponse:
    try:
        CreateCvmRequest.model_validate(request.state.parsed_body)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": exc.errors(include_url=False)})
    return JSONResponse(status_code=501, content={"detail": NOT_IMPLEMENTED})


@router.delete("/v1/cvm")
async def destroy_cvm(request: Request) -> JSONResponse:
    return JSONResponse(status_code=501, content={"detail": NOT_IMPLEMENTED})
