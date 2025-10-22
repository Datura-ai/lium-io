from pydantic import BaseModel


class SignaturePayload(BaseModel):
    """Base payload class for requests that require signature verification"""
    signature: str


class HardwareUtilizationPayload(SignaturePayload):
    """Payload for hardware utilization endpoint with signature verification"""
    pass  # Hex signature of the fixed string "hardware_utilization_request"


class PingPayload(SignaturePayload):
    """Payload for ping endpoint with signature verification"""
    pass  # Hex signature of the fixed string "ping_request"