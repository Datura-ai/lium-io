from typing import Any

from payloads.miner import MinerAuthPayload
from pydantic import BaseModel

# Stage states: ok (already satisfied — skipped) / done (performed now) /
# running / pending / failed / skipped (aborted before reached).
# Run states: running / succeeded / failed.
#
# Mutating request bodies inherit MinerAuthPayload: MinerMiddleware verifies
# `signature` over `data_to_sign` on every non-GET request, so the auth fields
# are part of each mutating body's contract.


class StageDoc(BaseModel):
    name: str
    state: str = "pending"
    detail: str | None = None
    took_s: float | None = None


class RunDoc(BaseModel):
    run_id: str
    op: str
    args: dict[str, Any] = {}
    state: str = "running"
    started_at: str
    finished_at: str | None = None
    stages: list[StageDoc] = []
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None


class RunAccepted(BaseModel):
    run_id: str


class HealthzResponse(BaseModel):
    ok: bool = True
    version: str


class SetupRequest(MinerAuthPayload):
    # machine_key is backend-minted per call and rotates on every mint
    # (plan-key-split); the account key never reaches this box
    machine_key: str
    machine_id: int | None = None
    force: bool = False
    # contract-events delivery config (plan-contract-events): persisted under
    # STATE_DIR_HOST for the poller; re-setup overwrites it (idempotent)
    events_token: str | None = None
    events_url: str | None = None
    executor_id: str | None = None


class DeleteRequest(MinerAuthPayload):
    purge: bool = False
    force: bool = False


class DeleteResponse(BaseModel):
    container_removed: bool
    state_removed: bool


class StatusDoc(BaseModel):
    # Local sections only, degrading independently (a broken probe reports
    # {"error": ...} in its slot). Vast-side data (listed, reliability, offers)
    # is merged by the backend, which holds the account key.
    machine_id: int | None = None
    # set only when the machine_id read itself broke (nsenter timeout etc.);
    # a plain "not enrolled" keeps machine_id null with no error
    machine_id_error: str | None = None
    rental_containers: list[str] | dict[str, Any] | None = None
    box: dict[str, Any]
