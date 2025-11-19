from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Index
from sqlmodel import Field, SQLModel


class PortMapping(SQLModel, table=True):
    """
    Port verification results for tracking executor port availability.

    Tracks verification status of port mappings on executor machines, including:
    - Port accessibility status (is_successful)
    - Current reservation status for pods (rented_for_pod_id, docker_port)
    - Last verification timestamp for cleanup of stale entries
    """

    uuid: UUID = Field(default_factory=uuid4, primary_key=True)
    miner_hotkey: str = Field(index=True)
    executor_id: UUID = Field(index=True)
    internal_port: int
    external_port: int
    is_successful: bool = True
    rented_for_pod_id: UUID | None = Field(default=None, index=True)
    docker_port: int | None = None  # Docker container port (nullable for backward compatibility)
    verification_time: datetime = Field(default_factory=datetime.utcnow, index=True)

    __table_args__ = (
        Index('idx_executor_success_time', 'executor_id', 'is_successful', 'verification_time'),
    )