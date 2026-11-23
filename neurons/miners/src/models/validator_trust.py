from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime
from sqlmodel import Field, SQLModel


class ValidatorTrustMessage(SQLModel, table=True):
    """A message the subnet owner signed naming the validator this miner trusts: one row per revision taken, so the
    validators trusted before stay known (their executor rows are served to the current one)."""

    __tablename__ = "validator_trust_message"

    revision: int = Field(sa_column=Column(BigInteger, primary_key=True, autoincrement=False))
    message: str
    signature: str
    # the owner coldkey that signed it
    signer: str
    # the network of the build that wrote the row ("" release, "test" staging); None: a row written before it was kept
    network: str | None = None
    # when a chain read had named `signer` as the subnet owner before the row was written; None: never confirmed
    owner_confirmed_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
