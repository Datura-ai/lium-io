from fastapi import HTTPException
import bittensor
import json
import time
from core.config import VALIDATOR_HOTKEYS_SS58, settings
from core.logger import get_logger
from payloads.backend import SignaturePayload, HardwareUtilizationPayload, PingPayload, ContainerUtilizationPayload

logger = get_logger(__name__)


def match_validator_hotkey(message: str | bytes, signature: str) -> str | None:
    """The id (`current` / `next`) of the configured validator hotkey that signed `message`, or None.

    Every validator-signature check in the executor goes through here, so the hotkey rotation
    (DAH-3394) is one list: VALIDATOR_HOTKEYS_SS58, tried in order. Malformed signatures raise
    as they did before; the caller decides the status code.
    """
    for key_id, ss58 in VALIDATOR_HOTKEYS_SS58.items():
        if bittensor.Keypair(ss58_address=ss58).verify(message, signature):
            # the id only: which key is in use is operational information, the key itself is not
            logger.debug("validator signature verified with the %s hotkey", key_id)
            return key_id
    return None


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
        # Normalize signature format - Bittensor expects 0x prefix
        signature = payload.signature
        if not signature.startswith('0x'):
            signature = '0x' + signature

        # Verify the signature against the message with each configured validator hotkey
        if match_validator_hotkey(message, signature) is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid signature: not signed by a configured validator hotkey"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Error verifying signature: {str(e)}"
        )


def require_fresh_timestamp(timestamp: int) -> None:
    """The signed timestamp is within CONTAINER_SIGNATURE_MAX_AGE_SECONDS of this host's clock.

    A valid signature proves who signed, not when the request was made: without this check a
    captured /containers request stays accepted for as long as the container exists (DAH-3200).
    The window is symmetric, so a host clock that runs ahead of the signer is refused the same way
    as one that runs behind, and the 401 names the skew so a provider can see it is their clock.
    """
    max_age = settings.CONTAINER_SIGNATURE_MAX_AGE_SECONDS
    # integer arithmetic: the signer's resolution is whole seconds, and a float subtraction would
    # overflow (500) on an absurdly large int the header parser still admits
    skew = int(time.time()) - timestamp
    if abs(skew) > max_age:
        detail = (
            f"Signed timestamp is {abs(skew)}s "
            f"{'behind' if skew > 0 else 'ahead of'} the executor clock; "
            f"the accepted window is {max_age}s. Check that this host's clock is NTP-synced."
        )
        # the 401 body goes back to the platform; this line is what the provider sees in `docker logs`
        logger.warning("Refusing a signed /containers request: %s", detail)
        raise HTTPException(status_code=401, detail=detail)


async def verify_allowed_hotkey_signature(payload: HardwareUtilizationPayload):
    FIXED_MESSAGE = "hardware_utilization_request"
    await verify_signature(payload, FIXED_MESSAGE)


async def verify_ping_signature(payload: PingPayload):
    FIXED_MESSAGE = "ping_request"
    await verify_signature(payload, FIXED_MESSAGE)


async def verify_container_signature(payload: ContainerUtilizationPayload):
    require_fresh_timestamp(payload.timestamp)
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
    require_fresh_timestamp(timestamp)
    signing_data = {
        "container_name": container_name,
        "timestamp": timestamp,
    }
    message = json.dumps(signing_data, sort_keys=True)

    payload = SignaturePayload(signature=signature)
    await verify_signature(payload, message)
