from fastapi import HTTPException
import bittensor
from core.config import settings
from core.logger import get_logger
from payloads.backend import HardwareUtilizationPayload

logger = get_logger(__name__)


async def verify_allowed_hotkey_signature(payload: HardwareUtilizationPayload):
    """
    Dependency function to verify a signature from the configured allowed Bittensor hotkey.
    Only accepts signatures from the hotkey specified in ALLOWED_HOTKEY_SS58_ADDRESS.
    
    Args:
        payload: HardwareUtilizationPayload containing only the signature
        
    Returns:
        None - just validates, raises HTTPException if invalid
        
    Raises:
        HTTPException: If signature verification fails
    """
    # Fixed string that must be signed by the client
    FIXED_MESSAGE = "hardware_utilization_request"
    
    try:
        # Create keypair from the allowed hotkey SS58 address
        keypair = bittensor.Keypair(ss58_address=settings.ALLOWED_HOTKEY_SS58_ADDRESS)
        
        # Normalize signature format - Bittensor expects 0x prefix
        signature = payload.signature
        if not signature.startswith('0x'):
            signature = '0x' + signature
        
        # Verify the signature against the fixed message
        is_valid = keypair.verify(FIXED_MESSAGE, signature)
        
        if not is_valid:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid signature from allowed hotkey {settings.ALLOWED_HOTKEY_SS58_ADDRESS}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error verifying signature: {str(e)}"
        )
