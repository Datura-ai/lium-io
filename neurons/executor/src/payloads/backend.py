from pydantic import BaseModel


class SignaturePayload(BaseModel):
    """Base payload class for requests that require signature verification"""
    signature: str


class HardwareUtilizationPayload(SignaturePayload):
    """Payload for hardware utilization endpoint with signature verification"""
    pass


class PingPayload(SignaturePayload):
    """Payload for ping endpoint with signature verification"""
    pass


class ContainerUtilizationPayload(SignaturePayload):
    """Payload for container hardware utilization endpoint with signature verification"""
    gpu_uuids: list[str]