from typing import Optional
from uuid import UUID, uuid4
from sqlmodel import Field, SQLModel


class AttestationWhitelist(SQLModel, table=True):
    """Whitelist for trusted attestation digests by TEE type."""

    __tablename__ = "attestation_whitelist"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    tee_type: str  # "dstack/tdx", "dstack/sev", etc.
    attestation_digest: str  # The merged hash digest
    description: Optional[str] = None  # Human-readable description
    added_at: Optional[str] = None  # ISO format timestamp
    is_active: bool = True  # Can be disabled without deleting
