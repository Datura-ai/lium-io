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


def ssh_pubkey_signing_blob(public_key: str, nonce: str | None = None) -> str:
    """Canonical message the validator signs over an SSH public key.

    CRITICAL: This is the single source of truth for the validator_signature
    format shared by the validator (signer), the miner (relay), and the
    executor (verifier). Any change breaks signature compatibility fleet-wide.

    Without a nonce the blob is the bare public key string — the format
    deployed executors already verify. With an attestation nonce the nonce is
    folded into the signed message so it cannot be stripped or swapped in
    transit while keeping a valid signature (a stripped nonce downgrades the
    verification to the legacy blob, which no longer matches the signature).
    """
    if not nonce:
        return public_key
    return f"{public_key}\nnonce:{nonce}"


class SSHPubKeySubmitRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.SSHPubKeySubmitRequest
    public_key: bytes
    validator_signature: str
    executor_id: Optional[str] = None
    is_rental_request: bool = False
    miner_hotkey: str
    # Attestation challenge (hex-encoded 32 bytes), minted by the validator per
    # attestation event. Relayed by the miner to each executor, which folds it
    # into TDX report_data[32:64] and the GPU evidence nonce. Optional for
    # backward compatibility with executors that predate the nonce channel.
    nonce: str | None = None


class SSHPubKeyRemoveRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.SSHPubKeyRemoveRequest
    public_key: bytes
    validator_signature: str
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
