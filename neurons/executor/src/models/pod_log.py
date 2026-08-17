import uuid
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class PodLog(BaseModel):
    """One container lifecycle event, stored as a JSON line in the pod-log file."""

    uuid: UUID = Field(default_factory=uuid.uuid4)
    container_name: str | None = None
    container_id: str | None = None
    event: str | None = None
    exit_code: int | None = None
    reason: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
