from pydantic import BaseModel


class SignaturePayload(BaseModel):
    """Base payload class for requests that require signature verification"""
    signature: str


class BoundSignaturePayload(SignaturePayload):
    """`signature` is the legacy fixed-string form. DAH-6430: `bound_signature` signs
    "<purpose>:<miner_hotkey>:<timestamp>" (bound_signing_message), so a request is valid only for
    one miner's nodes and a short window. An executor that predates it ignores the extra fields."""
    miner_hotkey: str | None = None
    timestamp: int | None = None
    bound_signature: str | None = None


class HardwareUtilizationPayload(BoundSignaturePayload):
    """Payload for hardware utilization endpoint with signature verification"""
    pass


class PingPayload(BoundSignaturePayload):
    """Payload for ping endpoint with signature verification"""
    pass


class ContainerUtilizationPayload(SignaturePayload):
    """Payload for container hardware utilization endpoint with signature verification"""
    gpu_uuids: list[str]
    timestamp: int