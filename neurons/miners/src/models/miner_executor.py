from typing import TYPE_CHECKING
from uuid import UUID
from datetime import datetime

from sqlmodel import Field, SQLModel, Relationship

if TYPE_CHECKING:
    from models.miner import Miner


class MinerExecutor(SQLModel, table=True):
    """MinerExecutor model for central mode (production DB)"""
    __tablename__ = "miner_executor"

    id: UUID = Field(primary_key=True)
    created_at: datetime = Field()
    updated_at: datetime | None = Field(default=None)

    miner_id: UUID = Field(foreign_key="miner.id")
    validator_hotkey: str
    gpu_type: str | None = None
    executor_ip_address: str
    executor_ip_port: str
    price_per_hour: float | None = None
    price_per_gpu: float | None = None
    gpu_count: int | None = None
    collateral_amount: float | None = None
    sync_status: str | None = Field(default=None, nullable=True)
    sync_error: str | None = Field(default=None)

    miner: "Miner" = Relationship(back_populates="miner_executors")
