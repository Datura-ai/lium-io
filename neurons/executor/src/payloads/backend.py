from pydantic import BaseModel


class HardwareUtilizationPayload(BaseModel):
    signature: str  # Hex signature of the fixed string "hardware_utilization_request"