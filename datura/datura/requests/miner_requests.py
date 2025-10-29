import enum

import pydantic
from datura.requests.base import BaseRequest


class RequestType(enum.Enum):
    GenericError = "GenericError"
    AcceptJobRequest = "AcceptJobRequest"
    DeclineJobRequest = "DeclineJobRequest"
    AcceptSSHKeyRequest = "AcceptSSHKeyRequest"
    FailedRequest = "FailedRequest"
    UnAuthorizedRequest = "UnAuthorizedRequest"
    SSHKeyRemoved = "SSHKeyRemoved"
    PodLogsResponse = "PodLogsResponse"


class Executor(pydantic.BaseModel):
    uuid: str
    address: str
    port: int


class BaseMinerRequest(BaseRequest):
    message_type: RequestType


class GenericError(BaseMinerRequest):
    message_type: RequestType = RequestType.GenericError
    details: str | None = None


class AcceptJobRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.AcceptJobRequest
    executors: list[Executor]


class DeclineJobRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.DeclineJobRequest


class ExecutorSSHInfo(pydantic.BaseModel):
    uuid: str
    address: str
    port: int
    ssh_username: str
    ssh_port: int
    python_path: str
    root_dir: str
    port_range: str | None = None
    port_mappings: str | None = None
    price: float | None = None
    ssh_host_key: str | None = None
    tdx_quote: str | None = None


class AcceptSSHKeyRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.AcceptSSHKeyRequest
    executors: list[ExecutorSSHInfo]


class SSHKeyRemoved(BaseMinerRequest):
    message_type: RequestType = RequestType.SSHKeyRemoved


class FailedRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.FailedRequest
    details: str | None = None


class UnAuthorizedRequest(FailedRequest):
    message_type: RequestType = RequestType.UnAuthorizedRequest


class PodLog(pydantic.BaseModel):
    uuid: str
    container_name: str | None = None
    container_id: str | None = None
    event: str | None = None
    exit_code: int | None = None
    reason: str | None = None
    error: str | None = None
    created_at: str


class PodLogsResponse(BaseMinerRequest):
    message_type: RequestType = RequestType.PodLogsResponse
    logs: list[PodLog] = []
