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
    UpdateExecutorRequest = "UpdateExecutorRequest"
    ExecutorUpdated = "ExecutorUpdated"
    ExecutorUpdateFailed = "ExecutorUpdateFailed"
    DeleteExecutorRequest = "DeleteExecutorRequest"
    ExecutorDeleted = "ExecutorDeleted"
    ExecutorDeleteFailed = "ExecutorDeleteFailed"


class BaseMinerPortalRequest(BaseRequest):
    message_type: RequestType


class AddExecutorPayload(BaseModel):
    gpu_type: str
    ip_address: str
    port: int
    price_per_hour: float | None = None
    gpu_count: int

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if self.gpu_count < 1:
            raise ValueError("Incorrect gpu_count")
        if self.price_per_hour and self.price_per_hour < 0:
            raise ValueError("Incorrect price_per_hour")

        return self


class SwitchValidatorPayload(BaseModel):
    validator_hotkey: str


class SyncExecutorPayload(BaseModel):
    uuid: UUID
    validator: str
    address: str
    port: int
    price_per_hour: float | None = None


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
    payload: list[Executor]


class SyncExecutorCentralMinerFailed(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.SyncExecutorCentralMinerFailed
    error: str


class UpdateExecutorRequest(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.UpdateExecutorRequest
    executor: SyncExecutorPayload


class ExecutorUpdated(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.ExecutorUpdated
    executor_id: UUID


class ExecutorUpdateFailed(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.ExecutorUpdateFailed
    executor_id: UUID
    error: str


class DeleteExecutorRequest(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.DeleteExecutorRequest
    executor: SyncExecutorPayload


class ExecutorDeleted(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.ExecutorDeleted
    executor_id: UUID


class ExecutorDeleteFailed(BaseMinerPortalRequest):
    message_type: RequestType = RequestType.ExecutorDeleteFailed
    executor_id: UUID
    error: str
