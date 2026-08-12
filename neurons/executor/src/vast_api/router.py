import json
import logging
import time

import bittensor
from fastapi import APIRouter, Header, HTTPException, Request

from middlewares.miner import trusted_hotkeys
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

SIGNATURE_MAX_AGE_SECONDS = 300

# Local operations only — market ops (list/unlist/price/self-test) belong to the
# backend, which holds the account key (plan-key-split). Auth rides the executor's
# MinerMiddleware for mutating requests: every non-GET request must carry a valid
# MinerAuthPayload signature in its body. The middleware skips GETs, so the
# sensitive GET routes below (status/runs leak journal tails, host command lines
# and the machine_id) verify their own x-signature/x-timestamp headers; only
# /healthz stays open.
router = APIRouter()


def verify_signed_get(path: str, x_signature: str, x_timestamp: int) -> None:
    # header-based signature auth, mirroring stream_container_logs (routes/apis.py);
    # message binds the concrete request path so a signature opens exactly one route
    if abs(time.time() - x_timestamp) > SIGNATURE_MAX_AGE_SECONDS:
        raise HTTPException(status_code=401, detail="Stale timestamp")
    message = json.dumps({"path": path, "timestamp": x_timestamp}, sort_keys=True)
    signature = x_signature if x_signature.startswith("0x") else "0x" + x_signature
    for hotkey in trusted_hotkeys(path):
        try:
            if bittensor.Keypair(ss58_address=hotkey).verify(message, signature):
                return
        except Exception:
            logger.warning("signature verification errored for hotkey %s", hotkey, exc_info=True)
    raise HTTPException(status_code=401, detail="Invalid signature")


def _manager(request: Request) -> VastManager:
    return request.app.state.vast_manager


@router.get("/healthz", response_model=HealthzResponse)
def healthz():
    return HealthzResponse(version=VERSION)


@router.get("/vast/status", response_model=StatusDoc)
def vast_status(request: Request, x_signature: str = Header(...), x_timestamp: int = Header(...)):
    verify_signed_get(request.url.path, x_signature, x_timestamp)
    return _manager(request).status()


@router.post("/vast/setup", response_model=RunAccepted, status_code=202)
def vast_setup(payload: SetupRequest, request: Request):
    return RunAccepted(
        run_id=_manager(request).setup(payload.machine_key, payload.machine_id, payload.force)
    )


@router.get("/vast/runs", response_model=list[RunDoc])
def vast_runs(request: Request, x_signature: str = Header(...), x_timestamp: int = Header(...)):
    verify_signed_get(request.url.path, x_signature, x_timestamp)
    return _manager(request).runs.list_runs()


@router.get("/vast/runs/{run_id}", response_model=RunDoc)
def vast_run(
    run_id: str, request: Request, x_signature: str = Header(...), x_timestamp: int = Header(...)
):
    verify_signed_get(request.url.path, x_signature, x_timestamp)
    doc = _manager(request).runs.get(run_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return doc


@router.delete("/vast", response_model=DeleteResponse)
def vast_delete(payload: DeleteRequest, request: Request):
    return _manager(request).delete(payload.purge, payload.force)
