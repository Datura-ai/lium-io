import enum
import json
import time

import bittensor
import pydantic
from datura.requests.base import BaseRequest


class RequestType(enum.Enum):
    AuthenticateRequest = "AuthenticateRequest"
    MachineSpecRequest = "MachineSpecRequest"
    ExecutorSpecRequest = "ExecutorSpecRequest"
    RentedMachineRequest = "RentedMachineRequest"
    LogValidatorRequest = "LogValidatorRequest"

class BaseValidatorRequest(BaseRequest):
    message_type: RequestType


class AuthenticationPayload(pydantic.BaseModel):
    validator_hotkey: str
    timestamp: int

    def blob_for_signing(self):
        instance_dict = self.model_dump()
        return json.dumps(instance_dict, sort_keys=True)


class AuthenticateRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.AuthenticateRequest
    payload: AuthenticationPayload
    signature: str

    def blob_for_signing(self):
        return self.payload.blob_for_signing()

    @classmethod
    def from_keypair(cls, keypair: bittensor.Keypair):
        payload = AuthenticationPayload(
            validator_hotkey=keypair.ss58_address,
            timestamp=int(time.time()),
        )
        return cls(payload=payload, signature=f"0x{keypair.sign(payload.blob_for_signing()).hex()}")


class ExecutorSpecRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.ExecutorSpecRequest
    miner_hotkey: str
    validator_hotkey: str
    executor_uuid: str
    executor_ip: str
    executor_port: int
    specs: dict | None
    score: float | None
    log_text: str | None
    log_status: str | None
    job_batch_id: str

class LogValidatorRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.LogValidatorRequest
    current_block: int
    validator_hotkey: str
    weights: list[float]
    scores: dict[str, float]
    timestamp: str
    log_text: str | None
    log_status: str | None

class RentedMachineRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.RentedMachineRequest
