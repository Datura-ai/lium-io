import enum
import json
import secrets
import time
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


def ssh_pubkey_signing_blob(
    public_key: str, nonce: str | None = None, miner_hotkey: str | None = None
) -> str:
    """Canonical message the validator signs over an SSH public key.

    CRITICAL: This is the single source of truth for the validator_signature
    format shared by the validator (signer), the miner (relay), and the
    executor (verifier). Any change breaks signature compatibility fleet-wide.

    Without a nonce the blob is the bare public key string — the format
    deployed executors already verify. With an attestation nonce the nonce is
    folded into the signed message so it cannot be stripped or swapped in
    transit while keeping a valid signature (a stripped nonce downgrades the
    verification to the legacy blob, which no longer matches the signature).

    With a miner hotkey the blob also names the miner the validator is talking
    to, so the executor can check the key was meant for its own miner. That form
    is only used for the separate `validator_miner_signature`; the
    `validator_signature` blob never carries it, so older executors keep working.
    """
    blob = f"{public_key}\nnonce:{nonce}" if nonce else public_key
    return f"{blob}\nminer:{miner_hotkey}" if miner_hotkey else blob


# I-78: a validator signature over an SSH key used to be good forever, so one captured upload could be
# replayed to any executor at any time. Every upload now carries a nonce whose first bytes say when it
# was minted; the signature covers it, and executors refuse it once stale. Still 32 hex-encoded bytes,
# the shape the TDX report_data and GPU-evidence channels take.
UPLOAD_NONCE_MAGIC = b"LTS1"


def fresh_upload_nonce() -> str:
    """32 bytes, hex: the magic, the minting time (8 bytes, big-endian unix seconds), 20 random bytes."""
    issued_at = int(time.time())
    return (UPLOAD_NONCE_MAGIC + issued_at.to_bytes(8, "big") + secrets.token_bytes(20)).hex()


def upload_nonce_issued_at(nonce: str | None) -> int | None:
    """When a fresh_upload_nonce was minted, or None for any other nonce (or none)."""
    try:
        raw = bytes.fromhex(nonce or "")
    except ValueError:
        return None
    if len(raw) != 32 or not raw.startswith(UPLOAD_NONCE_MAGIC):
        return None
    return int.from_bytes(raw[4:12], "big")


# liumd phase 1 (DAH-2834): the validator's one-call verification intent, `POST /verify` on the
# executor. Schema id of the wire documents and the capability string `GET /version` advertises.
LOCAL_VERIFY_SCHEMA = "lium.local_verify/1"
LOCAL_VERIFY_CAPABILITY = "local_verify/1"


# The largest card count one host can claim; bounds the matmul fan-out an intent can ask for.
LOCAL_VERIFY_MAX_DEVICES = 64


class LocalVerifyWireModel(pydantic.BaseModel):
    """Every document on the `/verify` wire, both sides. `extra="forbid"`: a field this schema does
    not name is a 422, never silently dropped — the Rust liumd (`deny_unknown_fields`) refuses it
    too, and a schema both sides refuse identically is one that can be pinned (LIUMD_RUST_PLAN
    §2.3.6). The intent's step challenges live here so the validator builds what the executor
    parses from ONE definition; the executor's `payloads/verify.py` imports them."""

    model_config = pydantic.ConfigDict(extra="forbid", populate_by_name=True)


class DeviceChallenge(LocalVerifyWireModel):
    """One card's own challenge for the all-cards work-proof: a run pinned to `index`
    (`CUDA_VISIBLE_DEVICES`) with a cipher text sealed for that card alone (seeds may repeat, as
    on the SSH path). The validator derives the cipher text per device (domain-separated from the
    intent's nonce), so the output of one real run unseals for one card only — a host with fewer
    cards than it claims cannot answer for all of them with a single computation."""

    index: int = pydantic.Field(ge=0)
    seed: int
    cipher_text: str = pydantic.Field(min_length=1, max_length=4096)


class MatmulStep(LocalVerifyWireModel):
    """The capability matmul challenge, exactly the arguments `decrypt_challenge.py` takes."""

    dim_n: int
    dim_k: int
    seed: int
    cipher_text: str = pydantic.Field(min_length=1, max_length=4096)
    # The all-cards work-proof: one pinned run per card, each with its own challenge; None = one
    # unpinned run with the challenge above.
    devices: list[DeviceChallenge] | None = pydantic.Field(default=None, max_length=LOCAL_VERIFY_MAX_DEVICES)

    @pydantic.model_validator(mode="after")
    def _one_challenge_per_card(self) -> "MatmulStep":
        if not self.devices:
            return self
        indexes = [d.index for d in self.devices]
        if len(set(indexes)) != len(indexes):
            raise ValueError("devices: the same card index twice")
        ciphers = {d.cipher_text for d in self.devices} | {self.cipher_text}
        if len(ciphers) != len(self.devices) + 1:
            raise ValueError("devices: every card needs its own cipher_text")
        return self


class VerifyXStep(LocalVerifyWireModel):
    """The VerifyX challenge, exactly the arguments `verifyx_executor.py` takes."""

    seed: int
    cipher_text: str = pydantic.Field(min_length=1, max_length=65536)


def local_verify_signing_blob(intent: dict[str, object]) -> str:
    """Canonical message the validator signs over a `/verify` intent.

    CRITICAL: the single source of truth for the intent signature, shared by the validator
    (signer, `neurons/validators/src/services/local_verify_client.py`) and the executor (verifier,
    `neurons/executor/src/services/local_verify_service.py`). The executor rebuilds it from the
    request body it received, so both sides must serialise the same way: every key but `signature`,
    sorted keys, no whitespace, ASCII-escaped. Any change here breaks every /verify fleet-wide.
    """
    unsigned = {k: v for k, v in intent.items() if k != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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
    # Validator signature over ssh_pubkey_signing_blob(public_key, nonce, miner_hotkey): relayed by
    # the miner so the executor can check the key was issued for its own miner hotkey. Optional
    # while validators, miners and executors roll out.
    validator_miner_signature: str | None = None


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
