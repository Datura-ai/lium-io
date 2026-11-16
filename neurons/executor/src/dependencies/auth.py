import json
import time

from bittensor_wallet import Keypair
from core.logger import get_logger
from fastapi import HTTPException
from payloads.backend import (
    BoundSignaturePayload,
    ContainerUtilizationPayload,
    HardwareUtilizationPayload,
    PingPayload,
    SignaturePayload,
)
from services.local_verify_service import accepted_miner_hotkeys

from core.config import VALIDATOR_HOTKEYS_SS58, settings

logger = get_logger(__name__)


def match_validator_hotkey(message: str | bytes, signature: str) -> str | None:
    """The id (`current` / `next`) of the configured validator hotkey that signed `message`, or None.

    Every validator-signature check in the executor goes through here, so the hotkey rotation
    (DAH-3394) is one list: VALIDATOR_HOTKEYS_SS58, tried in order. Malformed signatures raise
    as they did before; the caller decides the status code.
    """
    for key_id, ss58 in VALIDATOR_HOTKEYS_SS58.items():
        if Keypair(ss58_address=ss58).verify(message, signature):
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
        logger.warning("Refusing a signed request: %s", detail)
        raise HTTPException(status_code=401, detail=detail)


# DAH-6430: the fixed-string signature is still accepted while the backend sends both forms.
# Set False, then remove the legacy path and the `signature` field, once the fleet runs this version.
ACCEPT_UNBOUND_SIGNATURES = True


def bound_signing_message(purpose: str, miner_hotkey: str, timestamp: int) -> str:
    """What the backend signs for /hardware_utilization and /ping (lium-platform signs the same string)."""
    return f"{purpose}:{miner_hotkey}:{timestamp}"


async def verify_bound_signature(payload: BoundSignaturePayload, purpose: str) -> None:
    """The bound form when sent; while ACCEPT_UNBOUND_SIGNATURES, a refused or absent bound form falls back
    to the legacy one, so a host clock off by more than the window does not take the node offline."""
    if payload.bound_signature is not None:
        try:
            await _verify_bound(payload, purpose)
            return
        except HTTPException as refused:
            if not ACCEPT_UNBOUND_SIGNATURES:
                raise
            # counted before the legacy path is turned off: each line is a node that would then refuse
            logger.warning("Bound %s signature refused (%s); accepting the legacy one", purpose, refused.detail)
    elif not ACCEPT_UNBOUND_SIGNATURES:
        raise HTTPException(status_code=401, detail="A bound signature is required")
    await verify_signature(payload, purpose)


async def _verify_bound(payload: BoundSignaturePayload, purpose: str) -> None:
    if payload.miner_hotkey is None or payload.timestamp is None:
        raise HTTPException(status_code=401, detail="A bound signature needs miner_hotkey and timestamp")
    accepted = accepted_miner_hotkeys(
        own=settings.MINER_HOTKEY_SS58_ADDRESS,
        pool=settings.LIUM_POOL_HOTKEY,
        shared_portal_hotkey=settings.DEFAULT_MINER_HOTKEY,
    )
    if payload.miner_hotkey not in accepted:
        raise HTTPException(status_code=401, detail="Signature is for another miner's nodes")
    require_fresh_timestamp(payload.timestamp)
    message = bound_signing_message(purpose, payload.miner_hotkey, payload.timestamp)
    await verify_signature(SignaturePayload(signature=payload.bound_signature), message)


async def verify_allowed_hotkey_signature(payload: HardwareUtilizationPayload):
    await verify_bound_signature(payload, "hardware_utilization_request")


async def verify_ping_signature(payload: PingPayload):
    await verify_bound_signature(payload, "ping_request")


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
