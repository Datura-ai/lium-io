"""Messages the validator sends the backend over its WebSocket (`/validator/...`), one model per
`message_type`.

Two families share the socket and the backend reads both with one parser: the cycle results
(`ExecutorSpecRequest`, scores, estimates) and the answers to the backend's container requests
(`ContainerCreated`, `FailedRequest`, …). Both are here under one enum, `ValidatorMessageType`, with
the wire values the validator emits today (`neurons/validators/src/protocol/vc_protocol/validator_requests.py`
+ `payload_models/payloads.py`); the backend's copy (`src/protocol/validator_requests.py`) is a subset of
it — the difference is what this package exists to make visible.

Field types are what a receiver must accept: a field one side sends and the other ignores is here and
optional; a field an older validator omits is optional. A consumer that wants a stricter type
subclasses and registers (see base.py).
"""

from __future__ import annotations

import enum
import json
from datetime import datetime
from typing import Any

import pydantic

from .base import DeliveryStamps, Message, Registry, WorkloadKind


class ValidatorMessageType(enum.Enum):
    # cycle results and scores
    AuthenticateRequest = "AuthenticateRequest"
    MachineSpecRequest = "MachineSpecRequest"  # declared by both peers, sent by neither today
    ExecutorSpecRequest = "ExecutorSpecRequest"
    RentedMachineRequest = "RentedMachineRequest"
    LogStreamRequest = "LogStreamRequest"
    InspectorEventRequest = "InspectorEventRequest"
    DuplicateExecutorsRequest = "DuplicateExecutorsRequest"
    ResetVerifiedJobRequest = "ResetVerifiedJobRequest"
    NormalizedScoreRequest = "NormalizedScoreRequest"
    ScorePortionPerGpuTypeRequest = "ScorePortionPerGpuTypeRequest"
    RevenuePerGpuTypeRequest = "RevenuePerGpuTypeRequest"
    GpuEstimatesRequest = "GpuEstimatesRequest"
    EstimateResponse = "EstimateResponse"
    # answers to the backend's container requests (payload_models.ContainerResponseType)
    ContainerCreated = "ContainerCreated"
    ContainerStarted = "ContainerStarted"
    ContainerStopped = "ContainerStopped"
    ContainerDeleted = "ContainerDeleted"
    SshPubKeyAdded = "SshPubKeyAdded"
    SshPubKeyRemoved = "SshPubKeyRemoved"
    FailedRequest = "FailedRequest"
    PodLogsResponseToServer = "PodLogsResponseToServer"
    FailedGetPodLogs = "FailedGetPodLogs"
    DebugSshKeyAdded = "DebugSshKeyAdded"
    FailedAddDebugSshKey = "FailedAddDebugSshKey"
    JupyterServerInstalled = "JupyterServerInstalled"
    JupyterInstallationFailed = "JupyterInstallationFailed"


VALIDATOR_MESSAGES: Registry[ValidatorMessage] = Registry(ValidatorMessageType)


class VolumeEncryptionStatus(enum.StrEnum):
    ENABLED = "ENABLED"
    UNSUPPORTED_IMAGE = "UNSUPPORTED_IMAGE"
    DISABLED = "DISABLED"
    FAILED = "FAILED"


class ResetVerifiedJobReason(int, enum.Enum):
    DEFAULT = 0
    POD_NOT_RUNNING = 1  # container for pod is not running


class FailedContainerErrorCodes(enum.Enum):
    UnknownError = "UnknownError"
    NoSshKeys = "NoSshKeys"
    ContainerNotRunning = "ContainerNotRunning"
    DeletionInProgress = "DeletionInProgress"
    NoPortMappings = "NoPortMappings"
    InvalidExecutorId = "InvalidExecutorId"
    ExceptionError = "ExceptionError"
    FailedMsgFromMiner = "FailedMsgFromMiner"
    RentingInProgress = "RentingInProgress"
    NoJupyterPortMapping = "NoJupyterPortMapping"
    AttestationError = "AttestationError"
    # DAH-2703: the container the validator created was gone from the host before creation finished
    ContainerVanished = "ContainerVanished"


class FailedContainerErrorTypes(enum.Enum):
    ContainerCreationFailed = "ContainerCreationFailed"
    ContainerDeletionFailed = "ContainerDeletionFailed"
    ContainerStopFailed = "ContainerStopFailed"
    ContainerStartFailed = "ContainerStartFailed"
    AddSSkeyFailed = "AddSSkeyFailed"
    UnknownRequest = "UnknownRequest"


class ContainerWarningCode(enum.Enum):
    ExternalVolumeFailed = "ExternalVolumeFailed"


class ValidatorMessage(Message, DeliveryStamps):
    message_type: ValidatorMessageType


class AuthenticationPayload(pydantic.BaseModel):
    validator_hotkey: str
    timestamp: int

    def blob_for_signing(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True)


@VALIDATOR_MESSAGES.register
class AuthenticateRequest(ValidatorMessage):
    message_type: ValidatorMessageType = ValidatorMessageType.AuthenticateRequest
    payload: AuthenticationPayload
    signature: str

    def blob_for_signing(self) -> str:
        return self.payload.blob_for_signing()


@VALIDATOR_MESSAGES.register
class MachineSpecRequest(ValidatorMessage):
    message_type: ValidatorMessageType = ValidatorMessageType.MachineSpecRequest


class IncentiveReason(pydantic.BaseModel):
    """One structured zero-incentive reason (DAH-2340). `reason` is an append-only machine-readable code
    the backend keys off; `message_for_miner` is free text; `context` grows by key, never renames."""

    reason: str
    message_for_miner: str
    context: dict[str, Any] = pydantic.Field(default_factory=dict)


# `ExecutorSpecRequest.executor_uuid` of a miner-level failure (the validator's FAILED_MINER_EXECUTOR_UUID,
# core/validator.py): a placeholder, not a provider node. The backend keys its emission eligibility off it.
EXCLUDED_PROVIDER_EMISSION_EXECUTOR_ID = "11111111-1111-1111-1111-111111111111"


@VALIDATOR_MESSAGES.register
class ExecutorSpecRequest(ValidatorMessage):
    """One executor's result for one cycle. `specs` is the scraped machine description; the backend types
    it as its `MachineSpecs`, the validator sends whatever the scraper produced, so the wire type is a
    JSON object."""

    message_type: ValidatorMessageType = ValidatorMessageType.ExecutorSpecRequest
    miner_hotkey: str
    miner_coldkey: str
    validator_hotkey: str
    executor_uuid: str
    executor_ip: str
    executor_port: int
    executor_ssh_port: int | None = None
    price_per_gpu: float | None = None
    specs: dict[str, Any] | None = None
    score: float | None = None
    synthetic_job_score: float | None = None
    log_text: str | None = None
    log_status: str | None = None
    job_batch_id: str
    netuid: int | None = None
    scored_at: datetime | None = None
    incentive: float | None = None
    incentive_source: str | None = None
    # DAH-2467: how `incentive` splits across the two pools; both None on an old publisher
    incentive_rented: float | None = None
    incentive_idle: float | None = None
    node_state_at_cycle: str | None = None
    incentive_formula_version: str | None = None
    incentive_formula_inputs: dict[str, Any] | None = None
    # None = the publisher did not report the field; [] = no catalogued reason was recorded
    incentive_reasons: list[IncentiveReason] | None = None
    collateral_deposited: bool | None = None
    ssh_pub_keys: list[str] | None = None
    executor_image: dict[str, Any] | None = None
    # CVM attestation provenance (validator-side; the backend ignores them until it adopts them)
    tee_type: str | None = None
    attestation_digest: str | None = None
    tdx_attestation_passed: bool | None = None
    gpu_attestation_passed: bool | None = None
    # DAH-2792: specs the validator scored for this miner in this job_batch_id; expected minus received
    # is what was lost on the socket
    batch_total: int | None = None


@VALIDATOR_MESSAGES.register
class RentedMachineRequest(ValidatorMessage):
    message_type: ValidatorMessageType = ValidatorMessageType.RentedMachineRequest


@VALIDATOR_MESSAGES.register
class LogStreamRequest(ValidatorMessage):
    message_type: ValidatorMessageType = ValidatorMessageType.LogStreamRequest
    miner_hotkey: str
    validator_hotkey: str
    executor_uuid: str
    pod_id: str
    logs: list[dict[str, Any]]


@VALIDATOR_MESSAGES.register
class InspectorEventRequest(ValidatorMessage):
    message_type: ValidatorMessageType = ValidatorMessageType.InspectorEventRequest
    miner_hotkey: str
    validator_hotkey: str
    executor_id: str
    job_batch_id: str
    pod_ids: list[str]
    outcome: str
    reason_code: str
    report: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    context: dict[str, Any] = pydantic.Field(default_factory=dict)
    pipeline_id: str | None = None
    trace_id: str | None = None
    when: str


@VALIDATOR_MESSAGES.register
class DuplicateExecutorsRequest(ValidatorMessage):
    message_type: ValidatorMessageType = ValidatorMessageType.DuplicateExecutorsRequest


@VALIDATOR_MESSAGES.register
class ResetVerifiedJobRequest(ValidatorMessage):
    message_type: ValidatorMessageType = ValidatorMessageType.ResetVerifiedJobRequest
    validator_hotkey: str
    miner_hotkey: str
    executor_uuid: str
    reason: ResetVerifiedJobReason = ResetVerifiedJobReason.DEFAULT


@VALIDATOR_MESSAGES.register
class NormalizedScoreRequest(ValidatorMessage):
    message_type: ValidatorMessageType = ValidatorMessageType.NormalizedScoreRequest
    normalized_scores: list[dict[str, Any]]


@VALIDATOR_MESSAGES.register
class ScorePortionPerGpuTypeRequest(ValidatorMessage):
    message_type: ValidatorMessageType = ValidatorMessageType.ScorePortionPerGpuTypeRequest
    portions: dict[str, float]


@VALIDATOR_MESSAGES.register
class RevenuePerGpuTypeRequest(ValidatorMessage):
    message_type: ValidatorMessageType = ValidatorMessageType.RevenuePerGpuTypeRequest


@VALIDATOR_MESSAGES.register
class GpuEstimatesRequest(ValidatorMessage):
    message_type: ValidatorMessageType = ValidatorMessageType.GpuEstimatesRequest
    estimates: dict[str, Any]  # {gpu_model: {"rented": {...}, "unrented": {...}}}


@VALIDATOR_MESSAGES.register
class EstimateResponse(ValidatorMessage):
    """The validator's answer to the backend's `GetEstimateRequest` (backend_to_validator)."""

    message_type: ValidatorMessageType = ValidatorMessageType.EstimateResponse
    request_id: str = ""
    estimate: dict[str, Any]
    # declared by the backend's copy; the validator's model has no such field, so it never sends one
    snapshot: dict[str, Any] | None = None


# --- answers to the backend's container requests --------------------------------------------------


class ContainerResponse(ValidatorMessage):
    """Every answer names the miner and the executor the request was for."""

    miner_hotkey: str
    executor_id: str


class PodContainerResponse(ContainerResponse):
    pod_id: str
    workload_kind: WorkloadKind = WorkloadKind.CUSTOMER_RENTAL


@VALIDATOR_MESSAGES.register
class ContainerCreated(PodContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.ContainerCreated
    container_name: str
    volume_name: str
    port_maps: list[tuple[int, int]]
    # DAH-1524: deploy-time profiler steps, each {name, [timestamp], [duration], [skipped]}; loose objects
    # so a new step name on the validator cannot break parsing on the backend
    profilers: list[dict[str, Any]] = []
    backup_log_id: str | None = None
    restore_path: str | None = None
    restore_log_id: str | None = None
    jupyter_url: str | None = None
    warnings: list[ContainerWarningCode] | None = None
    storage_limit_gb: int | None = None
    volume_limit_gb: int | None = None
    local_volume_path: str | None = None
    volume_encryption_status: VolumeEncryptionStatus | None = None


@VALIDATOR_MESSAGES.register
class ContainerStarted(PodContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.ContainerStarted
    container_name: str


@VALIDATOR_MESSAGES.register
class ContainerStopped(PodContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.ContainerStopped
    container_name: str


@VALIDATOR_MESSAGES.register
class ContainerDeleted(PodContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.ContainerDeleted


@VALIDATOR_MESSAGES.register
class SshPubKeyAdded(PodContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.SshPubKeyAdded
    user_public_keys: list[str] = []


@VALIDATOR_MESSAGES.register
class SshPubKeyRemoved(PodContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.SshPubKeyRemoved
    user_public_keys: list[str] = []


@VALIDATOR_MESSAGES.register
class FailedContainerRequest(PodContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.FailedRequest
    error_type: FailedContainerErrorTypes = FailedContainerErrorTypes.ContainerCreationFailed
    error_code: FailedContainerErrorCodes | None = None
    # renter-safe headline: customer-facing events may show it, so never executor host details
    msg: str
    # DAH-2475: full structured diagnosis for ops — filler_run.failure_reason and logs, never customer events
    detail: str | None = None
    failure_step: str | None = None
    volume_encryption_status: VolumeEncryptionStatus | None = None


class PodLog(pydantic.BaseModel):
    uuid: str
    container_name: str | None = None
    container_id: str | None = None
    event: str | None = None
    exit_code: int | None = None
    reason: str | None = None
    error: str | None = None
    created_at: str


@VALIDATOR_MESSAGES.register
class PodLogsResponseToServer(PodContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.PodLogsResponseToServer
    container_name: str
    logs: list[PodLog] = []


@VALIDATOR_MESSAGES.register
class FailedGetPodLogs(PodContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.FailedGetPodLogs
    container_name: str
    msg: str


@VALIDATOR_MESSAGES.register
class DebugSshKeyAdded(ContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.DebugSshKeyAdded
    address: str
    port: int
    ssh_username: str
    ssh_port: int


@VALIDATOR_MESSAGES.register
class FailedAddDebugSshKey(ContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.FailedAddDebugSshKey
    msg: str


@VALIDATOR_MESSAGES.register
class JupyterServerInstalled(PodContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.JupyterServerInstalled
    jupyter_url: str


@VALIDATOR_MESSAGES.register
class JupyterInstallationFailed(PodContainerResponse):
    message_type: ValidatorMessageType = ValidatorMessageType.JupyterInstallationFailed
    msg: str
