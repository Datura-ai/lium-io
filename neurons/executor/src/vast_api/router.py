import hmac
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from vast_api.schemas import (
    DeleteRequest,
    DeleteResponse,
    HealthzResponse,
    ListRequest,
    ListResponse,
    PriceRequest,
    RunAccepted,
    RunDoc,
    SelfTestRequest,
    SetupRequest,
    StatusDoc,
    UnlistResponse,
)
from vast_api.service import VastManager

logger = logging.getLogger(__name__)

VERSION = "0.1.0"


def verify_token(request: Request, authorization: str | None = Header(None)) -> None:
    # mandatory bearer token on every route, /healthz included
    expected = request.app.state.vast_settings.VAST_API_TOKEN
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    provided = authorization.removeprefix("Bearer ")
    # compare bytes: compare_digest raises TypeError on non-ASCII str input
    if not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


router = APIRouter(dependencies=[Depends(verify_token)])


def _manager(request: Request) -> VastManager:
    return request.app.state.vast_manager


@router.get("/healthz", response_model=HealthzResponse)
def healthz():
    return HealthzResponse(version=VERSION)


@router.get("/vast/status", response_model=StatusDoc)
def vast_status(request: Request):
    return _manager(request).status()


@router.post("/vast/setup", response_model=RunAccepted, status_code=202)
def vast_setup(payload: SetupRequest, request: Request):
    return RunAccepted(run_id=_manager(request).setup(payload.machine_id, payload.force))


@router.post("/vast/self-test", response_model=RunAccepted, status_code=202)
def vast_self_test(payload: SelfTestRequest, request: Request):
    return RunAccepted(run_id=_manager(request).self_test())


@router.get("/vast/runs", response_model=list[RunDoc])
def vast_runs(request: Request):
    return _manager(request).runs.list_runs()


@router.get("/vast/runs/{run_id}", response_model=RunDoc)
def vast_run(run_id: str, request: Request):
    doc = _manager(request).runs.get(run_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return doc


@router.post("/vast/list", response_model=ListResponse)
def vast_list(payload: ListRequest, request: Request):
    return _manager(request).list_machine(payload.price_gpu_usd, payload.duration, payload.force)


@router.post("/vast/unlist", response_model=UnlistResponse)
def vast_unlist(request: Request):
    return _manager(request).unlist()


@router.post("/vast/price", response_model=ListResponse)
def vast_price(payload: PriceRequest, request: Request):
    return _manager(request).price(payload.price_gpu_usd)


@router.delete("/vast", response_model=DeleteResponse)
def vast_delete(request: Request, payload: DeleteRequest | None = None):
    payload = payload or DeleteRequest()
    return _manager(request).delete(payload.purge, payload.force)
