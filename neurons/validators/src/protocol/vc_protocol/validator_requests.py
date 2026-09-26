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
    PodStatesReport = "PodStatesReport"


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


AVAILABILITY_CATEGORY = "availability"


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

    @property
    def is_availability_error(self) -> bool:
        return self.category == AVAILABILITY_CATEGORY


class ContainerState(str, enum.Enum):
    """What the validator saw of one rented pod's container this cycle (DAH-3338)."""

    RUNNING = "running"
    # On the host but not running: docker inspect answered a status other than running.
    EXITED = "exited"
    # No container of that name on the host.
    ABSENT = "absent"
    # The SSH transport died before the container could be inspected; never read as absent.
    UNKNOWN = "unknown"
    # StaleContainerCleanupCheck removed it: an orphan the backend no longer lists as rented.
    REAPED = "reaped"


class PodContainerState(pydantic.BaseModel):
    pod_id: str
    container_state: ContainerState
    observed_at: datetime


# The backend bounds ExecutorSpecRequest.pod_states at 256 entries (lium-platform#312,
# `Field(max_length=256)`); a longer list fails its validation and the WHOLE spec is dropped, node
# listing included. One PodStatesReport chunk holds the same number. With
# settings.POD_STATES_REPORT_ENABLED every state of the cycle goes out in report chunks after the
# spec, and the spec keeps this bounded head as a copy for a backend that does not read the report
# yet (a copy that is reaped ids only when 256 or more are queued). With the flag off the spec is
# the only carrier: StaleContainerCleanupCheck hands its queued `reaped` ids at least
# REAPED_POD_STATES_FLOOR slots (they sit first in the list, so this cut never reaches them) and
# the last rented pods' observed states are cut every cycle until the queue drains.
POD_STATES_MAX_ITEMS = 256


def bound_pod_states(states: list[PodContainerState]) -> list[PodContainerState]:
    return states[:POD_STATES_MAX_ITEMS]


def chunk_pod_states(states: list[PodContainerState]) -> list[list[PodContainerState]]:
    """The cycle's states cut into PodStatesReport chunks of at most POD_STATES_MAX_ITEMS, in order.

    An empty list gives no chunk: a cycle that observed nothing sends no report.
    """
    return [states[start : start + POD_STATES_MAX_ITEMS] for start in range(0, len(states), POD_STATES_MAX_ITEMS)]


class PodStatesReport(BaseValidatorRequest):
    """DAH-3338: one chunk of the container states one cycle saw on one node.

    Sent after the cycle's ExecutorSpecRequest, one message per chunk, so a node whose states do not
    fit the spec's bound (256 rented pods plus queued reaped ids) still reports every one of them in
    the same cycle. The backend (lium-platform#312) writes the states onto the rental rows and
    nothing else: no cycle row, no validation report, so a second message per cycle changes no
    accounting. The write is idempotent, so a chunk delivered twice leaves the rows as they were.
    ``job_batch_id`` is the cycle; ``chunk_index`` counts from 0 up to ``chunk_total - 1``.
    """

    message_type: RequestType = RequestType.PodStatesReport
    validator_hotkey: str
    miner_hotkey: str
    executor_uuid: str
    job_batch_id: str
    chunk_index: int = pydantic.Field(ge=0)
    chunk_total: int = pydantic.Field(ge=1)
    pod_states: list[PodContainerState] = pydantic.Field(min_length=1, max_length=POD_STATES_MAX_ITEMS)


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
    # DAH-2748: every reachability check this cycle failed, each with its own reason code and
    # what we saw. The backend keeps the node off the market while the list is not empty and
    # clears it on an empty one. None means the cycle never got to check.
    availability_errors: list[dict[str, Any]] | None = None
    # DAH-3338: the container state of every rented pod this cycle observed, plus the orphans the
    # stale cleanup reaped. None when the cycle never reached the rented-state check. The backend
    # writes it onto rental_history; an older backend ignores the key.
    pod_states: list[PodContainerState] | None = None
    # The answer to the backend's recheck request, run out of cycle: the backend lifts its hold on a
    # passing one and credits no uptime for it, since the cycle's own report does that.
    recheck: bool = False


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


class _AbsentWhenNone(pydantic.BaseModel):
    """A field the host did not answer stays absent on the wire (as the redis payload had it), instead of a
    `null` the backend's JSONB row would keep; `None` set on purpose (`container_finished_at: None`) is dropped
    too — the row reads the same either way."""

    model_config = pydantic.ConfigDict(extra="allow")

    @pydantic.model_serializer(mode="wrap")
    def _drop_none(self, handler: pydantic.SerializerFunctionWrapHandler) -> dict[str, Any]:
        return {key: value for key, value in handler(self).items() if value is not None}


class ResetVerifiedJobContainerEvidence(_AbsentWhenNone):
    """What the rented-machine check saw of the pod's container (`docker inspect`, typed and bounded on the
    validator: rented_machine.py `_penalty_evidence_from_diagnostics`). Every field is optional."""

    container_status: str | None = None
    container_exit_code: int | None = None
    container_oom_killed: bool | None = None
    container_error: str | None = None
    container_started_at: str | None = None
    container_finished_at: str | None = None
    container_missing: bool | None = None
    diagnostics_capture_error: str | None = None


class ResetVerifiedJobEvidence(_AbsentWhenNone):
    """DAH-3386: what the check that cleared the verified job saw. The backend copies it into the penalty row
    (lium-platform DAH-3385 `details.evidence.validator`), so the keys are named here and on that side alike;
    `extra="allow"` keeps a key a newer check adds on the wire instead of dropping the reset."""

    pod_id: str | None = None
    container_name: str | None = None
    container: ResetVerifiedJobContainerEvidence | None = None


class ResetVerifiedJobRequest(BaseValidatorRequest):
    message_type: RequestType = RequestType.ResetVerifiedJobRequest
    validator_hotkey: str
    miner_hotkey: str
    executor_uuid: str
    reason: ResetVerifiedJobReason = ResetVerifiedJobReason.DEFAULT
    # DAH-3386: the check that cleared the verified job (its event reason_code and check_id) and what it saw —
    # for POD_NOT_RUNNING the container's status, exit code, OOM flag and finish time. The backend writes them
    # into the penalty it raises (lium-platform DAH-3385 details.evidence.validator). Optional both ways.
    reason_code: str | None = None
    check_id: str | None = None
    evidence: ResetVerifiedJobEvidence | None = None


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
