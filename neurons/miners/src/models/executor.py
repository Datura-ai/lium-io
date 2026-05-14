import uuid
from uuid import UUID

from sqlmodel import Field, SQLModel, UniqueConstraint


class Executor(SQLModel, table=True):
    """Task model."""

    __table_args__ = (UniqueConstraint("address", "port", name="unique_contraint_address_port"),)

    uuid: UUID | None = Field(default_factory=uuid.uuid4, primary_key=True)
    address: str
    port: int
    validator: str
    price_per_hour: float | None = None
    price_per_gpu: float | None = None
    # Mirrors the tier the provider chose in the Portal; "secure" is the
    # marketplace default so legacy rows that never received a sync stay valid.
    tier: str = Field(default="secure", nullable=False, max_length=16)

    def __str__(self):
        return f"{self.address}:{self.port}"
