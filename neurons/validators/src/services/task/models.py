import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

from datura.requests.miner_requests import ExecutorSSHInfo

from incentive.miner_incentive_log import IncentiveReason

if TYPE_CHECKING:
    from incentive.miner_incentive_log import MinerLogLine


class MixedFormulaInputs(BaseModel):
    """DAH-2467: what a partially rented split node's incentive was computed from.

    The two blocks are whole `JobResult.incentive_formula_inputs` payloads — the mining
    one for the rented GPUs, the unrented one for the free GPUs — so the stored numbers
    reproduce the paid incentive, which one flattened block cannot do.
    """

    rented_gpu_count: int
    free_gpu_count: int
    mining: dict[str, Any]
    unrented: dict[str, Any]


class JobResult(BaseModel):
    spec: dict | None = None
    executor_info: ExecutorSSHInfo
    score: float
    job_score: float
    collateral_deposited: bool = False
    job_batch_id: str
    log_status: str
    log_text: str
    # One timestamp shared by every executor finalized in this validator cycle.
    # It is assigned after final weights are calculated and is the accounting/day key.
    scored_at: datetime | None = None
    gpu_model: str | None = None
    gpu_count: int = 0
    sysbox_runtime: bool = False
    nvidia_driver_version: str = ""  # reported gpu.driver string, e.g. "580.95.05"
    supports_gpu_splitting: bool = False
    gpu_splitting_min_count: int | None = None
    ssh_pub_keys: list[str] | None = None
    is_rented: bool = False
    # GPUs held by rented pods (DAH-2467). None = unknown (old backend or not rented) — the
    # executor is then scored whole-box; set and below gpu_count on a split node, the incentive
    # engine scores the free GPUs in the unrented pool.
    rented_gpu_count: int | None = None
    # DAH-2467: the free remainder of a partially rented split node. It is always rated at the
    # minimum-split tier and rate, never as a bundle of its own size.
    is_split_remainder: bool = False
    is_spot: bool = False
    is_new_rentals_paused: bool = False
    is_provider_banned: bool = False
    provider_discord_connected: bool = True
    rental_created_at: datetime | None = None
    default_job_owner: str | None = None  # "miner" | "lium" | None; miner default job is excluded from unrented incentive

    # tdx attestation relevant fields
    attestation_digest: str | None = None
    tee_type: str | None = None
    tdx_attestation_passed: bool = False
    # G1 — NVIDIA CC GPU attestation outcome (None = not performed)
    gpu_attestation_passed: bool | None = None
    executor_image_report: dict[str, Any] | None = None

    inspector_outcome: str = "SKIPPED"

    # DAH-2748: every reachability check this cycle failed. The backend hides such a node from
    # the market until a cycle reports none. None means this cycle never got to check — a
    # failure before the connect leaves the stored codes alone rather than re-listing a node
    # nobody tested.
    availability_error_codes: list[str] | None = None

    # Incentive relevant fields
    mining_score: float | None = None                   # Score for mining pool for scoring logic
    sysbox_multiplier: float | None = None              # Multiplier for sysbox runtime for scoring logic
    driver_multiplier: float | None = None              # Multiplier for the minimum NVIDIA driver requirement
    uptime_multiplier: float | None = None              # Multiplier for uptime
    gpu_portion: float | None = None                    # Portion of the GPU model for scoring logic
    total_gpu_count: int | None = None                  # Total number of GPUs of the same model
    incentive: float | None = None                      # Incentive score for the executor in this cycle
    # DAH-2467: how `incentive` splits across the two pools. Always sums to `incentive`.
    incentive_rented: float | None = None
    incentive_idle: float | None = None
    mining_share: float | None = None
    total_mining_score: float | None = None

    # V2 incentive relevant fields
    effective_rate: float | None = None              # Effective rate for the executor in this cycle for scoring logic
    hourly_rate: float | None = None                  # Hourly rate for the executor in this cycle for scoring logic
    max_cap: int | None = None                        # Max cap for GPU counts in this cycle for scoring logic
    count_bucket: int | None = None                    # gpu_count_bucket the executor is accounted against; 0 = aggregate-cap path
    bucket_reassigned_from: int | None = None          # DAH-2528: gpu_count bucket the executor left because it was over cap
    bucket_reassigned_from_multiplier: float | None = None  # source bucket's cap multiplier at reassignment time
    total_unrented_by_gpu_type: float | None = None          # Weighted GPU count for the executor in this cycle for scoring logic
    cap_dilution_applied: bool | None = None           # Whether the cap dilution is applied for the executor in this cycle for scoring logic
    eligible_for_rental_share: bool = False
    unrented_cap_multiplier: float | None = None          # Cap dilution multiplier: min(count, cap) / count
    rental_share: float | None = None                  # Rental share for the executor in this cycle for scoring logic
    burn_share: float | None = None                    # Burn share for the executor in this cycle for scoring logic
    total_rental_cost: float | None = None              # Total rental cost for the executor in this cycle for scoring logic
    rental_share_raw: float | None = None
    total_burn_emission: float | None = None
    validator_tao_price_usd: float | None = None
    validator_alpha_rate_tao_per_block: float | None = None
    estimated_epoch_emission_usd: float | None = None
    tempo_blocks: int | None = None
    seconds_per_block: int | None = None
    fixed_ratio: float | None = None


    # Delivery buffers with dedicated export paths (full_log_text, direct publish);
    # excluded so no model_dump ever serializes them raw.
    incentive_logs: list[str] = Field(default_factory=list, exclude=True)
    # DAH-2340 wire reasons for the backend; [] when no catalogued reason was recorded.
    zero_incentive_reasons: list[IncentiveReason] = Field(default_factory=list, exclude=True)

    _mixed_formula_inputs: "MixedFormulaInputs | None" = PrivateAttr(default=None)

    def set_incentive_split(self, rented: float, idle: float) -> None:
        """Record the two-pool breakdown and keep `incentive` equal to their sum."""
        self.incentive_rented = rented
        self.incentive_idle = idle
        self.incentive = rented + idle

    def record_mixed_formula_inputs(self, inputs: "MixedFormulaInputs") -> None:
        """Take both portions' formula snapshots — call it while each portion still holds
        the GPU count its own formula ran on, before the merge restores the whole box."""
        self._mixed_formula_inputs = inputs

    def record_incentive_log(self, line: "MinerLogLine") -> None:
        """Append the line to the miner-facing log; zero-incentive lines also become data for the backend."""
        self.incentive_logs.append(line.to_log_line())
        if line.reason is not None:
            self.zero_incentive_reasons.append(line.to_incentive_reason())

    @property
    def incentive_source(self) -> str:
        """Classify the executor's finalized contribution for this cycle."""
        if self.incentive is None:
            return "unknown"
        if self.incentive <= 0:
            return "zero_incentive"
        if self._is_mixed:
            return "mixed"
        return "rented_emission" if self.is_rented else "idle_incentive"

    @property
    def _is_mixed(self) -> bool:
        """True when the merge produced this row from a rented and a free portion (DAH-2467).

        Keyed on the snapshot, not on the two amounts: a free portion that scored 0 still
        went through the mixed formula, and every label must agree on one answer.
        """
        return self._mixed_formula_inputs is not None

    @property
    def node_state_at_cycle(self) -> str:
        if self._is_mixed:
            return "mixed"
        return "rented" if self.is_rented else "idle"

    @property
    def incentive_formula_version(self) -> str:
        if self._is_mixed:
            return "mixed_v1"
        return "rental_price_v2" if self.eligible_for_rental_share else "mining_v1"

    @property
    def incentive_formula_inputs(self) -> dict[str, Any]:
        """Snapshot the exact scalar inputs needed to explain this cycle's incentive."""
        if self._mixed_formula_inputs is not None:
            # The property's contract is a plain JSON-ready dict, and the payload is
            # published straight into a JSON message.
            return self._mixed_formula_inputs.model_dump()
        if self.eligible_for_rental_share:
            return {
                "rental_share": self.rental_share,
                "rental_share_raw": self.rental_share_raw,
                "total_burn_emission": self.total_burn_emission,
                "burn_share": self.burn_share,
                "gpu_model": self.gpu_model,
                "gpu_count": self.gpu_count,
                "hourly_rate": self.hourly_rate,
                "unrented_cap_multiplier": self.unrented_cap_multiplier,
                "sysbox_multiplier": self.sysbox_multiplier,
                "driver_multiplier": self.driver_multiplier,
                "effective_rate": self.effective_rate,
                "total_rental_cost": self.total_rental_cost,
                "count_bucket": self.count_bucket,
                "bucket_reassigned_from": self.bucket_reassigned_from,
                "bucket_reassigned_from_multiplier": self.bucket_reassigned_from_multiplier,
                "max_cap": self.max_cap,
                "total_unrented_by_gpu_type": self.total_unrented_by_gpu_type,
                "cap_dilution_applied": self.cap_dilution_applied,
                "validator_tao_price_usd": self.validator_tao_price_usd,
                "validator_alpha_rate_tao_per_block": self.validator_alpha_rate_tao_per_block,
                "estimated_epoch_emission_usd": self.estimated_epoch_emission_usd,
                "tempo_blocks": self.tempo_blocks,
                "seconds_per_block": self.seconds_per_block,
                "fixed_ratio": self.fixed_ratio,
            }
        return {
            "score": self.score,
            "mining_share": self.mining_share,
            "mining_score": self.mining_score,
            "total_mining_score": self.total_mining_score,
            "rental_share": self.rental_share,
            "rental_share_raw": self.rental_share_raw,
            "total_burn_emission": self.total_burn_emission,
            "burn_share": self.burn_share,
            "gpu_portion": self.gpu_portion,
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "total_gpu_count": self.total_gpu_count,
            "sysbox_multiplier": self.sysbox_multiplier,
            "uptime_multiplier": self.uptime_multiplier,
            "driver_multiplier": self.driver_multiplier,
            "validator_tao_price_usd": self.validator_tao_price_usd,
            "validator_alpha_rate_tao_per_block": self.validator_alpha_rate_tao_per_block,
            "estimated_epoch_emission_usd": self.estimated_epoch_emission_usd,
            "tempo_blocks": self.tempo_blocks,
            "seconds_per_block": self.seconds_per_block,
            "fixed_ratio": self.fixed_ratio,
        }

    @property
    def full_log_text(self) -> str:
        """Return the full log text including the incentive logs."""
        if not self.log_text or not self.incentive_logs:
            return self.log_text
        text: str = self.log_text + "\n\n Incentive Scores Calculation Logs: " + "\n\n".join(self.incentive_logs)
        if self._is_mixed:
            text += (
                f"\n\nPartially rented split node total: rented {self.incentive_rented} "
                f"+ idle {self.incentive_idle} = {self.incentive}"
            )
        return text

    @property
    def is_successful(self):
        return (self.score > 0 or self.job_score > 0) and self.gpu_model and self.gpu_count > 0


# DAH-2748: the category of an error that means "someone could not reach something".
AVAILABILITY_CATEGORY = "availability"


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

    @property
    def is_availability_error(self) -> bool:
        """DAH-2748: this error means someone could not reach something, so the node is hidden."""
        return self.category == AVAILABILITY_CATEGORY


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
