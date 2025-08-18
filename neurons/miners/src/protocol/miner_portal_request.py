import enum
from uuid import UUID
from typing import Self
from pydantic import BaseModel, model_validator

from datura.requests.base import BaseRequest


class RequestType(enum.Enum):
    AddExecutorRequest = "AddExecutorRequest"
    ExecutorAdded = "ExecutorAdded"
    AddExecutorFailed = "AddExecutorFailed"


class BaseMinerPortalRequest(BaseRequest):
    message_type: RequestType


class AddExecutorPayload(BaseModel):
    validator_hotkey: str
    gpu_type: str
    ip_address: str
    port: int
    price_per_hour: float
    collateral_amount: float | None
    gpu_count: int | None

    @model_validator(mode="after")
    def check_gpu_count_collateral_amount(self) -> Self:
        if self.gpu_count is None and self.collateral_amount is None:
            raise ValueError("gpu_count or collateral_amount is required")
        return self


class AddExecutorRequest(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.AddExecutorRequest
    validator_hotkey: str
    payload: AddExecutorPayload


class ExecutorAdded(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.ExecutorAdded
    executor_id: UUID
    ip_address: str
    port: int


class AddExecutorFailed(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.AddExecutorFailed
    ip_address: str
    port: int
    error: str
