import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from datura.requests.miner_requests import ExecutorSSHInfo


class JobResult(BaseModel):
    spec: dict | None = None
    executor_info: ExecutorSSHInfo
    score: float
    job_score: float
    collateral_deposited: bool = False
    job_batch_id: str
    log_status: str
    log_text: str
    gpu_model: str | None = None
    gpu_count: int = 0
    sysbox_runtime: bool = False
    ssh_pub_keys: list[str] | None = None
    is_rented: bool = False

    # Incentive relevant fields 
    mining_score: float | None = None                   # Score for mining pool for scoring logic
    sysbox_multiplier: float | None = None              # Multiplier for sysbox runtime for scoring logic
    uptime_multiplier: float | None = None              # Multiplier for uptime
    gpu_portion: float | None = None                    # Portion of the GPU model for scoring logic
    total_gpu_count: int | None = None                  # Total number of GPUs of the same model
    incentive: float | None = None                      # Incentive score for the executor in this cycle

    # V2 incentive relevant fields 
    effective_rate: float | None = None              # Effective rate for the executor in this cycle for scoring logic
    hourly_rate: float | None = None                  # Hourly rate for the executor in this cycle for scoring logic
    max_cap: int | None = None                        # Max cap for GPU counts in this cycle for scoring logic
    total_unrented_by_gpu_type: int | None = None           # GPU count for the executor in this cycle for scoring logic
    cap_dilution_applied: bool | None = None           # Whether the cap dilution is applied for the executor in this cycle for scoring logic
    eligible_for_rental_share: bool = False
    rental_share: float | None = None                  # Rental share for the executor in this cycle for scoring logic
    burn_share: float | None = None                    # Burn share for the executor in this cycle for scoring logic
    total_rental_cost: float | None = None              # Total rental cost for the executor in this cycle for scoring logic
    
    incentive_logs: list[str] = []


class ValidationEvent(BaseModel):
    event: str
    reason_code: str
    severity: str
    category: str = "runtime"
    impact: str
    remediation: str | None = None
    what_we_saw: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    help_uri: str | None = None
    check_id: str | None = None
    pipeline_id: str | None = None
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    when: datetime
    context: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
        extra = "allow"


def build_msg(
    *,
    event: str,
    reason: str,
    severity: str,
    impact: str,
    remediation: str = "",
    category: str = "runtime",
    what: dict | None = None,
    warnings: list[str] | None = None,
    help_uri: str | None = None,
    check_id: str = "",
    pipeline_id: str | None = None,
    ctx: dict | None = None,
) -> ValidationEvent:
    """Return a consistent structured message with timestamp and trace ID."""
    return ValidationEvent(
        event=event,
        reason_code=reason,
        severity=severity,
        category=category,
        impact=impact,
        remediation=remediation,
        what_we_saw=what or {},
        warnings=warnings or [],
        help_uri=help_uri,
        check_id=check_id,
        pipeline_id=pipeline_id,
        trace_id=str(uuid.uuid4()),
        when=datetime.now(UTC),
        context=ctx or {},
    )
