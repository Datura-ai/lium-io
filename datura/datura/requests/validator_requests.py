import enum
import json
from typing import Optional

import pydantic
from datura.requests.base import BaseRequest


class RequestType(enum.Enum):
    AuthenticateRequest = "AuthenticateRequest"
    SSHPubKeySubmitRequest = "SSHPubKeySubmitRequest"
    SSHPubKeyRemoveRequest = "SSHPubKeyRemoveRequest"
    GetPodLogsRequest = "GetPodLogsRequest"


class BaseValidatorRequest(BaseRequest):
    message_type: RequestType


class AuthenticationPayload(pydantic.BaseModel):
    validator_hotkey: str
    miner_hotkey: str
    timestamp: int

    def blob_for_signing(self):
        instance_dict = self.model_dump()
        return json.dumps(instance_dict, sort_keys=True)


class AuthenticateRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.AuthenticateRequest
    payload: AuthenticationPayload
    signature: str

    def blob_for_signing(self):
        return self.payload.blob_for_signing()


class SSHPubKeySubmitRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.SSHPubKeySubmitRequest
    public_key: bytes
    executor_id: Optional[str] = None
    is_rental_request: bool = False
    miner_hotkey: str


class SSHPubKeyRemoveRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.SSHPubKeyRemoveRequest
    public_key: bytes
    executor_id: Optional[str] = None
    miner_hotkey: str


class GetPodLogsRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.GetPodLogsRequest
    executor_id: str
    container_name: str
    miner_hotkey: str


# Simple REST API models for validator authentication
class SimpleValidatorRequest(pydantic.BaseModel):
    """Simplified request model for REST API with basic signature validation.

    Validator signs their own hotkey to prove ownership.
    No timestamp or miner_hotkey required for read-only operations.
    """
    signature: str
    validator_hotkey: str


class ExecutorInfo(pydantic.BaseModel):
    """Information about a single executor."""
    uuid: str
    address: str
    port: int


class ExecutorListResponse(pydantic.BaseModel):
    """Response model containing list of executors for validator."""
    validator_hotkey: str
    executors: list[ExecutorInfo]
