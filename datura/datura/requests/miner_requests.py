import enum

from datura.requests.base import BaseRequest


class RequestType(enum.Enum):
    GenericError = "GenericError"
    AcceptJobRequest = "AcceptJobRequest"
    DeclineJobRequest = "DeclineJobRequest"
    AcceptSSHKeyRequest = "AcceptSSHKeyRequest"
    FailedRequest = "FailedRequest"
    UnAuthorizedRequest = "UnAuthorizedRequest"
    SSHKeyRemoved = "SSHKeyRemoved"


class BaseMinerRequest(BaseRequest):
    message_type: RequestType


class GenericError(BaseMinerRequest):
    message_type: RequestType = RequestType.GenericError
    details: str | None = None


class AcceptJobRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.AcceptJobRequest


class DeclineJobRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.DeclineJobRequest


class AcceptSSHKeyRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.AcceptSSHKeyRequest
    ssh_username: str
    ssh_port: int
    python_path: str
    root_dir: str


class SSHKeyRemoved(BaseMinerRequest):
    message_type: RequestType = RequestType.SSHKeyRemoved


class FailedRequest(BaseMinerRequest):
    message_type: RequestType = RequestType.FailedRequest
    details: str | None = None


class UnAuthorizedRequest(FailedRequest):
    message_type: RequestType = RequestType.UnAuthorizedRequest
