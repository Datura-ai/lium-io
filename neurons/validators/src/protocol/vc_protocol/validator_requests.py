import enum
import json
import time
import uuid
from datetime import datetime
from typing import Any

import bittensor
import pydantic
from datura.requests.base import BaseRequest
from incentive.miner_incentive_log import IncentiveReason
from payload_models.payloads import DeliveryStamps


class RequestType(enum.Enum):
    AuthenticateRequest = "AuthenticateRequest"
    MachineSpecRequest = "MachineSpecRequest"
    ExecutorSpecRequest = "ExecutorSpecRequest"
    RentedMachineRequest = "RentedMachineRequest"
    LogStreamRequest = "LogStreamRequest"
    InspectorEventRequest = "InspectorEventRequest"
    ResetVerifiedJobRequest = "ResetVerifiedJobRequest"
    DuplicateExecutorsRequest = "DuplicateExecutorsRequest"
    NormalizedScoreRequest = "NormalizedScoreRequest"
    RevenuePerGpuTypeRequest = "RevenuePerGpuTypeRequest"
    ScorePortionPerGpuTypeRequest = "ScorePortionPerGpuTypeRequest"
    GpuEstimatesRequest = "GpuEstimatesRequest"
    EstimateResponse = "EstimateResponse"


class BaseValidatorRequest(BaseRequest, DeliveryStamps):
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


class ValidationEvent(pydantic.BaseModel):
    event: str
    reason_code: str
    severity: str
    category: str = "runtime"
    impact: str
    remediation: str | None = None
    what_we_saw: dict[str, Any] = pydantic.Field(default_factory=dict)
    warnings: list[str] = pydantic.Field(default_factory=list)
    help_uri: str | None = None
    check_id: str | None = None
    pipeline_id: str | None = None
    trace_id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
    when: datetime
    context: dict[str, Any] = pydantic.Field(default_factory=dict)

    model_config = pydantic.ConfigDict(extra="allow")


class ExecutorSpecRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.ExecutorSpecRequest
    miner_hotkey: str
    miner_coldkey: str
    validator_hotkey: str
    executor_uuid: str
    executor_ip: str
    executor_port: int
    executor_ssh_port: int | None = None
    price_per_gpu: float | None = None
    specs: dict | None
    score: float | None
    synthetic_job_score: float | None
    log_text: str | None
    log_status: str | None
    validation_event: ValidationEvent | None = None
    job_batch_id: str
    netuid: int | None = None
    scored_at: datetime | None = None
    incentive: float | None = None
    # DAH-2467: how `incentive` splits across the two pools; both None on an old publisher.
    incentive_rented: float | None = None
    incentive_idle: float | None = None
    incentive_source: str | None = None
    node_state_at_cycle: str | None = None
    incentive_formula_version: str | None = None
    incentive_formula_inputs: dict[str, Any] | None = None
    collateral_deposited: bool
    ssh_pub_keys: list[str] | None = None
    # CVM attestation provenance (minimal-G5: the connector must not drop these).
    # Optional so the backend, which ignores unknown fields until it adopts them,
    # stays compatible in both directions.
    tee_type: str | None = None
    attestation_digest: str | None = None
    tdx_attestation_passed: bool | None = None
    gpu_attestation_passed: bool | None = None
    executor_image: dict[str, Any] | None = None
    # None = publisher did not report the field; [] = no catalogued reason was recorded.
    incentive_reasons: list[IncentiveReason] | None = None
    # DAH-2792: specs the validator scored for this miner in this job_batch_id, counting the ones
    # whose redis publish failed; the backend reads expected minus received as lost on the websocket.
    batch_total: int | None = None


class RentedMachineRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.RentedMachineRequest


class LogStreamRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.LogStreamRequest
    miner_hotkey: str
    validator_hotkey: str
    executor_uuid: str
    pod_id: str
    logs: list[dict]


class InspectorEventRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.InspectorEventRequest
    miner_hotkey: str
    validator_hotkey: str
    executor_id: str
    job_batch_id: str
    pod_ids: list[str]
    outcome: str
    reason_code: str
    report: dict | None = None
    error: dict | None = None
    context: dict = pydantic.Field(default_factory=dict)
    pipeline_id: str | None = None
    trace_id: str | None = None
    when: str


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


class GpuEstimatesRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.GpuEstimatesRequest
    estimates: dict


class EstimateResponse(BaseValidatorRequest):
    message_type: RequestType = RequestType.EstimateResponse
    request_id: str = ""
    estimate: dict
