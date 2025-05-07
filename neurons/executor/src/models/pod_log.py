import uuid
from uuid import UUID
from datetime import datetime
from sqlmodel import Field, SQLModel


class PodLog(SQLModel, table=True):
    """Task model."""

    uuid: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    container_name: str | None = None
    container_id: str | None = None
    event: str | None = None
    exit_code: int | None = None
    reason: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

