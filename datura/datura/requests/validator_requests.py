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


class SSHPubKeyRemoveRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.SSHPubKeyRemoveRequest
    public_key: bytes
    executor_id: Optional[str] = None


class GetPodLogsRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.GetPodLogsRequest
    executor_id: str
    container_name: str
