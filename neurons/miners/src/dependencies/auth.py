"""Dependency for simple validator signature verification."""

import logging
import time
from typing import Annotated

import bittensor
from datura.requests.validator_requests import SimpleValidatorRequest, AuthenticateRequest, AuthenticationPayload
from fastapi import Depends, Header, HTTPException, status

from services.validator_service import ValidatorService

logger = logging.getLogger(__name__)

AUTH_MESSAGE_MAX_AGE = 10


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
    
    Args:
        x_validator_hotkey: Validator hotkey from header
        x_miner_hotkey: Miner hotkey from header
        x_timestamp: Timestamp from header
        x_signature: Signature from header
        validator_service: Service to check validator registration
        
    Returns:
        str: validator_hotkey if authentication successful
        
    Raises:
        HTTPException: 401 if signature invalid, 403 if validator not registered
    """
    validator_hotkey = x_validator_hotkey
    
    # Check timestamp
    if x_timestamp < time.time() - AUTH_MESSAGE_MAX_AGE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication message too old"
        )
    
    # Check validator registration
    if not validator_service.is_valid_validator(validator_hotkey):
        logger.warning("Validator %s is not registered", validator_hotkey)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Validator is not registered",
        )
    
    # Create authentication payload and verify signature
    from datura.requests.validator_requests import AuthenticationPayload
    payload = AuthenticationPayload(
        validator_hotkey=validator_hotkey,
        miner_hotkey=x_miner_hotkey,
        timestamp=x_timestamp,
    )
    
    # Verify signature
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
