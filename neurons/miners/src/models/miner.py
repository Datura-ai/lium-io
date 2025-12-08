from typing import TYPE_CHECKING
from uuid import UUID

from sqlmodel import Field, SQLModel, Relationship

if TYPE_CHECKING:
    from models.miner_executor import MinerExecutor


class Miner(SQLModel, table=True):
    """Miner model for central mode (production DB)"""
    __tablename__ = "miner"

    id: UUID = Field(primary_key=True)
    miner_hotkey: str = Field()
    miner_coldkey: str = Field()
    opt_in_status: bool = Field(default=False)

    miner_executors: list["MinerExecutor"] = Relationship(back_populates="miner")