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
    LogStreamRequest = "LogStreamRequest"
    ResetVerifiedJobRequest = "ResetVerifiedJobRequest"
    DuplicateExecutorsRequest = "DuplicateExecutorsRequest"
    NormalizedScoreRequest = "NormalizedScoreRequest"
    RevenuePerGpuTypeRequest = "RevenuePerGpuTypeRequest"
    ScorePortionPerGpuTypeRequest = "ScorePortionPerGpuTypeRequest"


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
    miner_coldkey: str
    validator_hotkey: str
    executor_uuid: str
    executor_ip: str
    executor_port: int
    executor_price: float | None = None
    specs: dict | None
    score: float | None
    synthetic_job_score: float | None
    log_text: str | None
    log_status: str | None
    job_batch_id: str
    collateral_deposited: bool
    ssh_pub_keys: list[str] | None = None


class RentedMachineRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.RentedMachineRequest


class LogStreamRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.LogStreamRequest
    miner_hotkey: str
    validator_hotkey: str
    executor_uuid: str
    logs: list[dict]


class ResetVerifiedJobReason(int, enum.Enum):
    DEFAULT = 0
    POD_NOT_RUNNING = 1         # container for pod is not running


class ResetVerifiedJobRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.ResetVerifiedJobRequest
    validator_hotkey: str
    miner_hotkey: str
    executor_uuid: str
    reason: ResetVerifiedJobReason = ResetVerifiedJobReason.DEFAULT


class DuplicateExecutorsRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.DuplicateExecutorsRequest


class NormalizedScoreRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.NormalizedScoreRequest
    normalized_scores: list[dict]


class RevenuePerGpuTypeRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.RevenuePerGpuTypeRequest


class ScorePortionPerGpuTypeRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.ScorePortionPerGpuTypeRequest
    portions: dict[str, float]
