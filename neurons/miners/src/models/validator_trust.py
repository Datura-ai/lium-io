from sqlalchemy import BigInteger, Column
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
