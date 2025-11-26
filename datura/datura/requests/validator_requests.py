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
        """Generate the canonical serialization used for signature generation and verification.

        CRITICAL: This method defines the canonical signing format used across both
        WebSocket and REST authentication flows. All signature generation and verification
        must use this method to ensure compatibility between validator and miner components.

        The serialization uses sorted keys (sort_keys=True) to ensure consistent ordering
        across different Python implementations and versions. This method must remain stable;
        any changes will break signature compatibility across the entire system.

        All callers MUST use this method instead of constructing their own JSON strings.
        See call sites in:
        - neurons/validators/src/clients/miner_client.py (WebSocket signing)
        - neurons/miners/src/dependencies/auth.py (REST verification)
        - neurons/validators/src/services/miner_service.py (REST signing)

        Returns:
            str: JSON string with sorted keys representing the payload
        """
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
