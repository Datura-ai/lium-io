from fastapi import HTTPException
import bittensor
from core.config import settings
from core.logger import get_logger
from payloads.backend import SignaturePayload, HardwareUtilizationPayload, PingPayload, ContainerUtilizationPayload

logger = get_logger(__name__)


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
    FIXED_MESSAGE = "container_utilization_request"
    await verify_signature(payload, FIXED_MESSAGE)
