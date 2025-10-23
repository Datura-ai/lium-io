"""Dependency for simple validator signature verification."""

import logging
from typing import Annotated

import bittensor
from datura.requests.validator_requests import SimpleValidatorRequest
from fastapi import Depends, HTTPException, status

from services.validator_service import ValidatorService

logger = logging.getLogger(__name__)


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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Signature verification error: {str(e)}"
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
