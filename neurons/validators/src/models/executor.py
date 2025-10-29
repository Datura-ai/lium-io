from typing import Optional
from uuid import UUID, uuid4
from sqlmodel import Field, SQLModel


class Executor(SQLModel, table=True):
    """Miner model."""

    uuid: UUID | None = Field(default_factory=uuid4, primary_key=True)
    miner_address: str
    miner_port: int
    miner_hotkey: str
    executor_id: UUID
    executor_ip_address: str
    executor_ssh_username: str
    executor_ssh_port: int
    rented: Optional[bool] = None

    # Attestation fields
    attestation_digest: Optional[str] = None
    tee_type: Optional[str] = None
    attestation_verified: Optional[bool] = None
    attestation_verified_at: Optional[str] = None  # ISO format timestamp
