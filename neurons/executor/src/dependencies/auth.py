from dataclasses import dataclass

import time

from fastapi import Header, HTTPException
import bittensor
import json
from core.config import settings
from core.config import VALIDATOR_HOTKEY_SS58
from core.logger import get_logger
from datura.requests.validator_requests import AuthenticationPayload
from payloads.backend import SignaturePayload, HardwareUtilizationPayload, PingPayload, ContainerUtilizationPayload

logger = get_logger(__name__)
AUTH_MESSAGE_MAX_AGE = 10


@dataclass(frozen=True)
class ChutesMutationAuth:
    validator_hotkey: str
    miner_hotkey: str
    timestamp: int


async def verify_signature(payload: SignaturePayload, message: str) -> None:
    """
    Universal signature verification function for any message.

    Args:
        payload: SignaturePayload containing the signature
        message: The fixed string that was signed by the client

    Returns:
        None - just validates, raises HTTPException if invalid

    Raises:
        HTTPException: If signature verification fails
    """
    try:
        # Create keypair from the allowed hotkey SS58 address
        keypair = bittensor.Keypair(ss58_address=settings.ALLOWED_HOTKEY_SS58_ADDRESS)

        # Normalize signature format - Bittensor expects 0x prefix
        signature = payload.signature
        if not signature.startswith('0x'):
            signature = '0x' + signature

        # Verify the signature against the message
        is_valid = keypair.verify(message, signature)

        if not is_valid:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid signature from allowed hotkey {settings.ALLOWED_HOTKEY_SS58_ADDRESS}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Error verifying signature: {str(e)}"
        )


async def verify_allowed_hotkey_signature(payload: HardwareUtilizationPayload):
    FIXED_MESSAGE = "hardware_utilization_request"
    await verify_signature(payload, FIXED_MESSAGE)


async def verify_ping_signature(payload: PingPayload):
    FIXED_MESSAGE = "ping_request"
    await verify_signature(payload, FIXED_MESSAGE)


async def verify_container_signature(payload: ContainerUtilizationPayload):
    signing_data  = {
        "gpu_uuids": payload.gpu_uuids,
        "timestamp": payload.timestamp,
    }
    message = json.dumps(signing_data, sort_keys=True)
    await verify_signature(payload, message)


async def verify_container_logs_signature(container_name: str, timestamp: int, signature: str):
    """
    Verify signature for container logs endpoint using header-based auth.

    Args:
        container_name: Name of the container (part of signed message)
        timestamp: Unix timestamp (part of signed message)
        signature: The signature from header
    """
    signing_data = {
        "container_name": container_name,
        "timestamp": timestamp,
    }
    message = json.dumps(signing_data, sort_keys=True)

    payload = SignaturePayload(signature=signature)
    await verify_signature(payload, message)


async def verify_chutes_mutation_auth_from_headers(
    x_validator_hotkey: str | None = Header(default=None, alias="X-Validator-Hotkey"),
    x_miner_hotkey: str | None = Header(default=None, alias="X-Miner-Hotkey"),
    x_timestamp: str | None = Header(default=None, alias="X-Timestamp"),
    x_signature: str | None = Header(default=None, alias="X-Signature"),
) -> ChutesMutationAuth:
    missing_headers = []
    if x_validator_hotkey is None:
        missing_headers.append("X-Validator-Hotkey")
    if x_miner_hotkey is None:
        missing_headers.append("X-Miner-Hotkey")
    if x_timestamp is None:
        missing_headers.append("X-Timestamp")
    if x_signature is None:
        missing_headers.append("X-Signature")

    if missing_headers:
        raise HTTPException(
            status_code=401,
            detail=f"Missing auth headers: {', '.join(missing_headers)}",
        )

    if x_validator_hotkey != VALIDATOR_HOTKEY_SS58:
        raise HTTPException(
            status_code=403,
            detail="Validator is not authorized for Chutes relay mutations",
        )

    expected_miner_hotkey = settings.MINER_HOTKEY_SS58_ADDRESS
    if x_miner_hotkey != expected_miner_hotkey:
        raise HTTPException(
            status_code=403,
            detail="Miner hotkey does not match this executor",
        )

    try:
        timestamp_value = int(x_timestamp)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"X-Timestamp must be a Unix timestamp. Received: {x_timestamp}",
        ) from exc

    timestamp_seconds = timestamp_value / 1000 if timestamp_value > 1e12 else timestamp_value
    if timestamp_seconds < 946684800:
        raise HTTPException(
            status_code=400,
            detail=f"X-Timestamp must be a Unix timestamp in seconds. Received invalid value: {x_timestamp}",
        )

    now = int(time.time())
    time_diff = abs(now - timestamp_seconds)
    if time_diff > AUTH_MESSAGE_MAX_AGE:
        if timestamp_seconds < now - AUTH_MESSAGE_MAX_AGE:
            raise HTTPException(
                status_code=401,
                detail=f"Authentication message too old. Timestamp is outside the allowed {AUTH_MESSAGE_MAX_AGE}-second window and must be regenerated.",
            )
        raise HTTPException(
            status_code=401,
            detail=f"Authentication message timestamp is too far in the future. Timestamp must be within {AUTH_MESSAGE_MAX_AGE} seconds of current time.",
        )

    payload = AuthenticationPayload(
        validator_hotkey=x_validator_hotkey,
        miner_hotkey=x_miner_hotkey,
        timestamp=int(timestamp_seconds),
    )
    try:
        normalized_signature = x_signature if x_signature.startswith('0x') else f'0x{x_signature}'
        keypair = bittensor.Keypair(ss58_address=x_validator_hotkey)
        if not keypair.verify(payload.blob_for_signing(), normalized_signature):
            raise HTTPException(
                status_code=401,
                detail="Invalid signature",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error verifying Chutes mutation signature: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=401,
            detail=f"Signature verification error: {str(exc)}",
        )

    return ChutesMutationAuth(
        validator_hotkey=x_validator_hotkey,
        miner_hotkey=x_miner_hotkey,
        timestamp=int(timestamp_seconds),
    )
