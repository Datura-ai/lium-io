from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Index
from sqlmodel import Field, SQLModel


class PortMapping(SQLModel, table=True):
    """Minimal port verification results for production usage."""

    uuid: UUID = Field(default_factory=uuid4, primary_key=True)
    miner_hotkey: str = Field(index=True)
    executor_id: UUID = Field(index=True)
    internal_port: int
    external_port: int
    is_successful: bool = True
    verification_time: datetime = Field(default_factory=datetime.utcnow, index=True)

    __table_args__ = (
        Index('idx_executor_success_time', 'executor_id', 'is_successful', 'verification_time'),
    )