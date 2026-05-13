from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

ForceValidationStatus = Literal["queued", "running", "succeeded", "failed"]


class ForceValidationCreateRequest(BaseModel):
    executor_id: str
    miner_hotkey: str


class ForceValidationResult(BaseModel):
    success: bool
    message: str | None = None
    score: float | None = None
    job_score: float | None = None
    gpu_model: str | None = None
    gpu_count: int | None = None


class ForceValidationError(BaseModel):
    message: str


class ForceValidationRequestRecord(BaseModel):
    request_id: str
    executor_id: str
    miner_hotkey: str
    status: ForceValidationStatus
    stage: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    result: ForceValidationResult | None = None
    error: ForceValidationError | None = None
