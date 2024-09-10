from typing import Optional
import enum
import uuid
from uuid import UUID
from datetime import datetime

from sqlmodel import Column, Enum, Field, SQLModel


class TaskStatus(str, enum.Enum):
    Initiated = "Initiated"
    SSHConnected = "SSHConnected"
    Failed = "Failed"
    Finished = "Finished"


class Task(SQLModel, table=True):
    """Task model."""

    uuid: UUID | None = Field(default_factory=uuid.uuid4, primary_key=True)
    task_status: TaskStatus = Field(sa_column=Column(Enum(TaskStatus)))
    miner_hotkey: str
    executor_id: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    proceed_time: Optional[int] = Field(default=None)
    score: Optional[float] = None
