import logging

from fastapi import APIRouter, HTTPException, Request

from vast_api.schemas import (
    DeleteRequest,
    DeleteResponse,
    HealthzResponse,
    RunAccepted,
    RunDoc,
    SetupRequest,
    StatusDoc,
)
from vast_api.service import VastManager

logger = logging.getLogger(__name__)

VERSION = "0.1.0"

# Local operations only — market ops (list/unlist/price/self-test) belong to the
# backend, which holds the account key (plan-key-split). Auth rides the executor's
# MinerMiddleware: every non-GET request must carry a valid MinerAuthPayload
# signature in its body; GET routes are open like the rest of the executor app.
router = APIRouter()


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
    return RunAccepted(
        run_id=_manager(request).setup(payload.machine_key, payload.machine_id, payload.force)
    )


@router.get("/vast/runs", response_model=list[RunDoc])
def vast_runs(request: Request):
    return _manager(request).runs.list_runs()


@router.get("/vast/runs/{run_id}", response_model=RunDoc)
def vast_run(run_id: str, request: Request):
    doc = _manager(request).runs.get(run_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return doc


@router.delete("/vast", response_model=DeleteResponse)
def vast_delete(payload: DeleteRequest, request: Request):
    return _manager(request).delete(payload.purge, payload.force)
