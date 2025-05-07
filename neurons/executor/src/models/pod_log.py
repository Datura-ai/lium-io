import uuid
from uuid import UUID
from datetime import datetime
from sqlmodel import Field, SQLModel


class PodLog(SQLModel, table=True):
    """Task model."""

    uuid: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    container_name: str
    container_id: str
    event: str
    exit_code: int | None = None
    reason: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

