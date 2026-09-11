"""Dependency for simple validator signature verification."""

import logging
import time
from typing import Annotated

import bittensor
from datura.requests.validator_requests import SimpleValidatorRequest
from fastapi import Depends, Header, HTTPException, status

from core import config as core_config
from services.validator_service import ValidatorService

logger = logging.getLogger(__name__)

AUTH_MESSAGE_MAX_AGE = 10  # Maximum age of authentication message in seconds


async def verify_validator_signature(
    validator_hotkey: str,
    signature: str,
) -> None:
    """Verify validator signature.

    Validator signs their own hotkey to prove ownership.

    Args:
        validator_hotkey: The hotkey that was signed
        signature: The signature to verify

    Raises:
        HTTPException: If signature verification fails
    """
    try:
        # Create keypair from the validator hotkey
        keypair = bittensor.Keypair(ss58_address=validator_hotkey)

        # Normalize signature format - Bittensor expects 0x prefix
        normalized_signature = signature if signature.startswith('0x') else f'0x{signature}'

        # Verify the signature (validator signs their own hotkey)
        is_valid = keypair.verify(validator_hotkey, normalized_signature)

        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Signature verification error: {str(e)}"
        )

async def verify_validator_auth_from_headers(
    x_validator_hotkey: Annotated[str, Header(alias="X-Validator-Hotkey")],
    x_miner_hotkey: Annotated[str, Header(alias="X-Miner-Hotkey")],
    x_timestamp: Annotated[int, Header(alias="X-Timestamp")],
    x_signature: Annotated[str, Header(alias="X-Signature")],
    validator_service: Annotated[ValidatorService, Depends(ValidatorService)],
) -> str:
    """Verify validator authentication from headers (REST API).
    
    This extracts authentication info from headers and verifies it.
    Headers expected: X-Validator-Hotkey, X-Miner-Hotkey, X-Timestamp, X-Signature
    
    The X-Timestamp header must be a Unix timestamp in seconds. Values larger than
    1e12 are treated as milliseconds and automatically converted to seconds. The timestamp
    must be within AUTH_MESSAGE_MAX_AGE seconds of the current time (symmetric window
    to handle small clock differences between servers). Each request must send a fresh
    timestamp to prevent replay attacks.

    X-Miner-Hotkey is part of the signed blob and names the miner the validator signed in
    to. Outside CENTRAL_MODE this miner serves one hotkey and answers 403 when the header
    names another one; the central miner serves many portal hotkeys and accepts any, with
    `authenticated_miner_hotkey` binding the request body to the same name.
    
    Args:
        x_validator_hotkey: Validator hotkey from header
        x_miner_hotkey: Miner hotkey from header (must be this miner's hotkey outside CENTRAL_MODE)
        x_timestamp: Unix timestamp from header (in seconds, or milliseconds if > 1e12)
        x_signature: Signature from header
        validator_service: Service to check validator registration
        
    Returns:
        str: validator_hotkey if authentication successful
        
    Raises:
        HTTPException: 400 if timestamp format invalid, 401 if signature invalid or timestamp
            outside allowed window, 403 if validator not registered or miner hotkey mismatch
    """
    validator_hotkey = x_validator_hotkey
    
    # Normalize timestamp to seconds
    # If timestamp is larger than 1e12, treat it as milliseconds and convert
    if x_timestamp > 1e12:
        timestamp_seconds = x_timestamp / 1000
    else:
        timestamp_seconds = x_timestamp
    
    # Validate timestamp is a reasonable Unix timestamp (after 2000-01-01)
    if timestamp_seconds < 946684800:  # 2000-01-01 00:00:00 UTC
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"X-Timestamp must be a Unix timestamp in seconds. Received invalid value: {x_timestamp}"
        )
    
    now = int(time.time())
    
    # Check timestamp is within allowed symmetric window
    # Allow timestamps within AUTH_MESSAGE_MAX_AGE seconds in either direction
    # to handle small clock differences between validator and miner servers
    time_diff = abs(now - timestamp_seconds)
    if time_diff > AUTH_MESSAGE_MAX_AGE:
        if timestamp_seconds < now - AUTH_MESSAGE_MAX_AGE:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication message too old. Timestamp is outside the allowed {AUTH_MESSAGE_MAX_AGE}-second window and must be regenerated."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication message timestamp is too far in the future. Timestamp must be within {AUTH_MESSAGE_MAX_AGE} seconds of current time."
            )
    
    # Check validator registration
    if not validator_service.is_valid_validator(validator_hotkey):
        logger.warning("Validator %s is not registered", validator_hotkey)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Validator is not registered",
        )

    # A standard miner serves one hotkey: a sign-in made for another miner is not for us.
    # settings is read through its module so a test can replace it for one request.
    if not core_config.settings.CENTRAL_MODE:
        own_hotkey = core_config.settings.get_bittensor_wallet().get_hotkey().ss58_address
        if x_miner_hotkey != own_hotkey:
            logger.warning(
                "Validator %s signed in for miner %s; this miner is %s",
                validator_hotkey,
                x_miner_hotkey,
                own_hotkey,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Authentication names another miner",
            )
    
    # Create authentication payload and verify signature
    # Use normalized timestamp in seconds
    from datura.requests.validator_requests import AuthenticationPayload
    payload = AuthenticationPayload(
        validator_hotkey=validator_hotkey,
        miner_hotkey=x_miner_hotkey,
        timestamp=int(timestamp_seconds),
    )

    # Verify signature
    # IMPORTANT: Use AuthenticationPayload.blob_for_signing() for canonical serialization.
    # This contract is defined in datura/datura/requests/validator_requests.py
    try:
        keypair = bittensor.Keypair(ss58_address=validator_hotkey)
        normalized_signature = x_signature if x_signature.startswith('0x') else f'0x{x_signature}'
        is_valid = keypair.verify(payload.blob_for_signing(), normalized_signature)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error verifying signature: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Signature verification error: {str(e)}"
        )
    
    return validator_hotkey


async def authenticated_miner_hotkey(
    x_miner_hotkey: Annotated[str, Header(alias="X-Miner-Hotkey")],
    _validator_hotkey: Annotated[str, Depends(verify_validator_auth_from_headers)],
) -> str:
    """The miner hotkey the validator's signed headers named (verified by the dependency above)."""
    return x_miner_hotkey


def require_request_names_authenticated_miner(request_miner_hotkey: str, authenticated: str) -> None:
    """A request body's miner_hotkey must be the one the signed headers named.

    The validator sends the same hotkey in both places (MinerService._generate_auth_headers and
    every request it builds); a body naming another miner would let signed-in headers act for a
    miner the signature never named.

    Raises:
        HTTPException: 403 when the body names another miner
    """
    if request_miner_hotkey != authenticated:
        logger.warning(
            "Request names miner %s but the signed headers named %s",
            request_miner_hotkey,
            authenticated,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request names a miner the authentication did not",
        )


async def verify_simple_validator_signature(
    request: SimpleValidatorRequest,
    validator_service: Annotated[ValidatorService, Depends(ValidatorService)],
) -> str:
    """Verify simple validator signature and return validator_hotkey if valid.

    This is a simplified authentication scheme for read-only REST API endpoints.
    Validator signs their own hotkey to prove ownership.

    Args:
        request: Request containing signature and validator_hotkey
        validator_service: Service to check validator registration

    Returns:
        str: validator_hotkey if authentication successful

    Raises:
        HTTPException: 401 if signature invalid, 403 if validator not registered
    """
    validator_hotkey = request.validator_hotkey

    # Check validator registration
    if not validator_service.is_valid_validator(validator_hotkey):
        logger.warning("Validator %s is not registered", validator_hotkey)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Validator is not registered",
        )

    # Verify signature
    await verify_validator_signature(validator_hotkey, request.signature)

    return validator_hotkey
