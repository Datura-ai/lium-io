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
