from datetime import datetime, timezone

from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel


class UnmarkedRecoveryStop(SQLModel, table=True):
    """A pod the validator stopped and parked as <name>__stopped while its Redis stop mark could not be written.

    The validator's own record of that stop, kept outside Redis so a validator restart before Redis is back still
    attributes the stop to the validator (the container name alone is forgeable on the provider's host)."""

    __tablename__ = "unmarked_recovery_stop"

    pod_id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))
