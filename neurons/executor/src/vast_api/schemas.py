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
    machine_id: int | None = None
    force: bool = False


class SelfTestRequest(MinerAuthPayload):
    pass


class ListRequest(MinerAuthPayload):
    price_gpu_usd: float
    duration: str = "7 days"
    force: bool = False


class PriceRequest(MinerAuthPayload):
    price_gpu_usd: float


class UnlistRequest(MinerAuthPayload):
    force: bool = False


class ListResponse(BaseModel):
    listed: bool
    offers: list[dict[str, Any]]
    warnings: list[str] = []


class UnlistResponse(BaseModel):
    listed: bool
    offers_remaining: int


class DeleteRequest(MinerAuthPayload):
    purge: bool = False
    force: bool = False


class DeleteResponse(BaseModel):
    container_removed: bool
    state_removed: bool
    vast_record_deleted: bool


class StatusDoc(BaseModel):
    # Sections degrade independently: a broken probe reports {"error": ...}
    # in its section while the rest still answers.
    goal: dict[str, Any]
    vast: dict[str, Any]
    offers: list[dict[str, Any]] | dict[str, Any]
    box: dict[str, Any]
    lium: dict[str, Any]
    ttl: dict[str, Any]
