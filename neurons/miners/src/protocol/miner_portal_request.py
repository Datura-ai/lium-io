import enum
from uuid import UUID
from typing import Self
from pydantic import BaseModel, model_validator

from datura.requests.base import BaseRequest

from models.executor import Executor


class RequestType(enum.Enum):
    AddExecutorRequest = "AddExecutorRequest"
    ExecutorAdded = "ExecutorAdded"
    AddExecutorFailed = "AddExecutorFailed"
    SyncExecutorMinerPortalRequest = "SyncExecutorMinerPortalRequest"
    SyncExecutorMinerPortalSuccess = "SyncExecutorMinerPortalSuccess"
    SyncExecutorMinerPortalFailed = "SyncExecutorMinerPortalFailed"
    SyncExecutorCentralMinerRequest = "SyncExecutorCentralMinerRequest"
    SyncExecutorCentralMinerSuccess = "SyncExecutorCentralMinerSuccess"
    SyncExecutorCentralMinerFailed = "SyncExecutorCentralMinerFailed"


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


class SyncExecutorPayload(BaseModel):
    uuid: UUID
    validator: str
    address: str
    port: int


class AddExecutorRequest(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.AddExecutorRequest
    executor_id: UUID
    validator_hotkey: str
    payload: AddExecutorPayload


class ExecutorAdded(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.ExecutorAdded
    executor_id: UUID


class AddExecutorFailed(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.AddExecutorFailed
    executor_id: UUID
    error: str


class SyncExecutorMinerPortalRequest(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.SyncExecutorMinerPortalRequest
    payload: list[SyncExecutorPayload]


class SyncExecutorMinerPortalSuccess(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.SyncExecutorMinerPortalSuccess


class SyncExecutorMinerPortalFailed(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.SyncExecutorMinerPortalFailed
    error: str


class SyncExecutorCentralMinerRequest(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.SyncExecutorCentralMinerRequest


class SyncExecutorCentralMinerSuccess(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.SyncExecutorCentralMinerSuccess
    miner_hotkey: str
    payload: list[Executor]


class SyncExecutorCentralMinerFailed(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.SyncExecutorCentralMinerFailed
    error: str