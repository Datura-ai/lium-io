"""Everything a miner sees in their "Incentive Scores Calculation Logs" — one catalog.

WHERE A NODE EARNS (two "pools" of subnet emission; the rest is burned):
  - Mining pool   — for RENTED executors. Paid by a mining score (GPU model/count
                    times sysbox / driver / uptime multipliers).
  - Unrented pool — for IDLE executors of an eligible GPU model. Paid as if the GPU
                    were rented, based on its rental market value (rental-share).

HOW A NODE PICKS A POOL:
  rented?                                              -> mining pool (always earns)
  idle AND model in program AND price ok AND disk >= 1.5x VRAM
                               AND capacity?            -> unrented pool (earns)
  otherwise                                            -> 0 incentive (reason below)

WHAT THIS CATALOG HOLDS — every `MinerLogLine` the miner-facing log block
(JobResult.incentive_logs, delivered via MACHINE_SPEC_CHANNEL) can contain:

1. ZERO-INCENTIVE REASONS — each records the fact "this executor gets NO payout
   because <reason>" (`MinerLogLine.no_payout_because_*` constructors):
   Group A — earns nothing in EITHER pool (built by `_reason_excluded_from_both_pools`):
     spot tier, Discord not connected, paused for new rentals, running own default job
   Group B — idle but does not qualify for the unrented pool:
     GPU model not in the unrented program (earns only when rented),
     price above the market soft limit (lower the price to earn),
     total disk below 1.5x total GPU VRAM (add disk to earn),
     no unrented capacity for that GPU-count tier this cycle,
     NVIDIA driver below the minimum, sysbox runtime not enabled

2. CALCULATION REPORTS — the per-cycle score/incentive lines every scored node gets:
     mining_score_calculated, mining_incentive_calculated,
     rental_incentive_calculated, mining_score_missing (internal-error case)

The scoring code (rental_price.py / default.py) detects each condition where its
data naturally lives (some per-executor upfront, some only after cohort aggregation),
builds the matching line and records it via `result.record_incentive_log(line)` —
which appends the text to incentive_logs AND, for zero-incentive lines, ships the
structured reason to the backend (DAH-2340). Never append to incentive_logs directly.
Open THIS file to see everything a miner can be told and exactly how each message reads.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from core.utils import _m, _StructuredMessage, get_extra_info

if TYPE_CHECKING:
    from incentive.rental_price import InsufficientDisk
    from services.task_service import JobResult


class ZeroIncentiveReason(StrEnum):
    """Machine-readable codes for every reason a validated executor earns 0.

    APPEND-ONLY contract: Loki dashboards and the backend (DAH-2340 structured
    payload) key off these exact strings — renaming a value is a breaking change.
    """

    # Group A: excluded from BOTH pools
    SPOT_TIER = "spot_tier"
    PROVIDER_DISCORD_NOT_CONNECTED = "provider_discord_not_connected"
    NEW_RENTALS_PAUSED = "new_rentals_paused"
    MINER_DEFAULT_JOB = "miner_default_job"
    # Group B: idle but not qualified for the unrented pool
    GPU_MODEL_NOT_ELIGIBLE_FOR_UNRENTED_INCENTIVE = "gpu_model_not_eligible_for_unrented_incentive"
    PRICE_ABOVE_MARKET_P90_SOFT_LIMIT = "price_above_market_p90_soft_limit"
    NO_UNRENTED_CAPACITY_FOR_GPU_COUNT = "no_unrented_capacity_for_gpu_count"
    NVIDIA_DRIVER_BELOW_MINIMUM = "nvidia_driver_below_minimum"
    INSUFFICIENT_DISK_FOR_VRAM = "insufficient_disk_for_vram"
    SYSBOX_NOT_ENABLED = "sysbox_not_enabled"


class IncentiveReason(BaseModel):
    """One structured zero-incentive reason as it travels on the wire (DAH-2340).

    Single definition for the whole validator: built here by the catalog
    (`MinerLogLine.to_incentive_reason`) and reused by `ExecutorSpecRequest`.
    The contract stays additive by growing `context` keys, never by renaming.
    """

    reason: str               # stable, APPEND-ONLY machine-readable code the backend keys off
    message_for_miner: str    # free text, may change any time
    # per-reason details shown next to the message, e.g. soft_limit_threshold,
    # gpu_model, gpu_count, executor_id, incentive
    context: dict[str, Any] = Field(default_factory=dict)


class MinerLogLine(BaseModel):
    """One line of the miner-facing incentive log.

    Built ONLY via the named constructors below (the catalog). The constructor bakes
    every field in; recording takes no arguments:

        line: MinerLogLine = MinerLogLine.no_payout_because_spot_tier(result)
        result.record_incentive_log(line)
    """

    message: str                                                   # plain-English, shown to the miner
    fields: dict[str, Any] = Field(default_factory=dict)           # structured payload next to the message
    reason: ZeroIncentiveReason | None = None                      # machine-readable zero-incentive code; None for reports
    internal_message: str | None = None                            # separate internal-log wording (both-pools exclusions only)
    internal_fields: dict[str, Any] = Field(default_factory=dict)  # full extra dict for that internal log

    def to_log_line(self) -> str:
        """Render as one string; the caller appends it to result.incentive_logs."""
        return self.as_internal_log().to_full_string()

    def to_incentive_reason(self) -> IncentiveReason:
        """Typed wire reason for MACHINE_SPEC_CHANNEL (DAH-2340): zero-incentive lines only, never internal_* fields."""
        # fields carries "reason" for the Loki extra; in the wire model the code already sits top-level.
        context: dict[str, Any] = {key: value for key, value in self.fields.items() if key != "reason"}
        return IncentiveReason(reason=self.reason.value, message_for_miner=self.message, context=context)

    def as_internal_log(self) -> _StructuredMessage:
        """The same line as an `_m` object, for mirroring into the internal logger."""
        return _m(self.message, extra=get_extra_info(self.fields))

    def to_internal_log(self) -> _StructuredMessage:
        """The separate internal observability log (both-pools exclusions only).

        Uses `internal_message`/`internal_fields` baked in by `_no_payout`; the caller
        just does `logger.info(exclusion.to_internal_log())`.
        """
        return _m(self.internal_message, extra=self.internal_fields)

    @staticmethod
    def _no_payout(
        result: JobResult,
        reason: ZeroIncentiveReason,
        message: str,
        extra_fields: dict[str, Any] | None = None,
        internal_message: str | None = None,
        internal_extra_fields: dict[str, Any] | None = None,
    ) -> MinerLogLine:
        """Shared shape of every zero-incentive reason: executor identity + reason + incentive 0."""
        fields: dict[str, Any] = {
            "executor_id": str(result.executor_info.uuid),
            "gpu_model": result.gpu_model,
            "gpu_count": result.gpu_count,
            "reason": reason,
            "incentive": 0.0,
            **(extra_fields or {}),
        }
        internal_fields: dict[str, Any] = {}
        if internal_message is not None:
            internal_fields = {
                "executor_id": str(result.executor_info.uuid),
                "gpu_model": result.gpu_model,
                "gpu_count": result.gpu_count,
                "reason": reason,
                "score": 0,
                "pool": "none",
                **(internal_extra_fields or {}),
            }
        return MinerLogLine(
            message=message,
            fields=fields,
            reason=reason,
            internal_message=internal_message,
            internal_fields=internal_fields,
        )

    # ── Group A: excluded from BOTH pools (mining + unrented) — earns nothing ─

    @staticmethod
    def no_payout_because_spot_tier(result: JobResult) -> MinerLogLine:
        return MinerLogLine._no_payout(
            result,
            reason=ZeroIncentiveReason.SPOT_TIER,
            message=(
                "No subnet incentive: this executor is on the spot tier, and spot-tier "
                "executors do not earn subnet incentive."
            ),
            internal_message="Executor excluded from both pools - spot tier",
        )

    @staticmethod
    def no_payout_because_discord_not_connected(result: JobResult) -> MinerLogLine:
        return MinerLogLine._no_payout(
            result,
            reason=ZeroIncentiveReason.PROVIDER_DISCORD_NOT_CONNECTED,
            message=(
                "No subnet incentive: provider Discord is not connected. Connect your "
                "provider Discord for this executor to start earning incentive."
            ),
            internal_message="Executor excluded from both pools - provider Discord not connected",
            internal_extra_fields={"provider_discord_connected": result.provider_discord_connected},
        )

    @staticmethod
    def no_payout_because_paused_for_new_rentals(result: JobResult) -> MinerLogLine:
        return MinerLogLine._no_payout(
            result,
            reason=ZeroIncentiveReason.NEW_RENTALS_PAUSED,
            message=(
                "No subnet incentive: this executor is paused for new rentals and earns "
                "nothing while paused. Resume new rentals to start earning again."
            ),
            internal_message="Executor excluded from both pools - paused for new rentals",
        )

    @staticmethod
    def no_payout_because_running_own_default_job(result: JobResult) -> MinerLogLine:
        return MinerLogLine._no_payout(
            result,
            reason=ZeroIncentiveReason.MINER_DEFAULT_JOB,
            message=(
                "No subnet incentive: this executor is running your own default job instead "
                "of a Lium job, and executors on your own job do not earn subnet incentive."
            ),
            internal_message="Executor excluded from both pools - running miner's own default job",
        )

    # ── Group B: idle but not qualified for the unrented pool ─────────────────

    @staticmethod
    def no_payout_because_gpu_model_not_in_unrented_program(result: JobResult) -> MinerLogLine:
        return MinerLogLine._no_payout(
            result,
            reason=ZeroIncentiveReason.GPU_MODEL_NOT_ELIGIBLE_FOR_UNRENTED_INCENTIVE,
            message=(
                f"No subnet incentive while idle: GPU model {result.gpu_model} is not part of "
                "the unrented (idle-node) incentive program, so this executor earns nothing "
                "unless rented. Rent it out to earn."
            ),
        )

    @staticmethod
    def no_payout_because_price_above_market_soft_limit(
        result: JobResult, market_p90: float, rate: float
    ) -> MinerLogLine:
        price_per_gpu: float | None = result.executor_info.price_per_gpu
        soft_limit: float = round(market_p90 * rate, 4)
        return MinerLogLine._no_payout(
            result,
            reason=ZeroIncentiveReason.PRICE_ABOVE_MARKET_P90_SOFT_LIMIT,
            message=(
                f"No unrented incentive: your price ${price_per_gpu}/GPU/h is above the market "
                f"soft price limit ${soft_limit} (90th-percentile market rate ${market_p90} x "
                f"{rate}). Lower the price to ${soft_limit} or below to earn the unrented incentive."
            ),
            extra_fields={
                "price_per_gpu": price_per_gpu,
                "machine_price_p90": market_p90,
                "soft_limit_rate": rate,
                "soft_limit_threshold": soft_limit,
            },
        )

    @staticmethod
    def no_payout_because_insufficient_disk_for_vram(
        result: JobResult, measured: InsufficientDisk
    ) -> MinerLogLine:
        # takes the measurement whole: the two totals are interchangeable floats, and swapping
        # them would quietly tell the miner to add disk he already has
        required_disk_gb: float = round(measured.vram_gb * measured.rate, 1)
        return MinerLogLine._no_payout(
            result,
            reason=ZeroIncentiveReason.INSUFFICIENT_DISK_FOR_VRAM,
            message=(
                f"No unrented incentive: this executor has {measured.disk_gb} GB of total disk but "
                f"{measured.vram_gb} GB of GPU VRAM. An idle executor must have at least "
                f"{measured.rate}x its GPU VRAM in disk to earn the unrented incentive. Give it at "
                f"least {required_disk_gb} GB of total disk, or rent it out to earn."
            ),
            extra_fields={
                "total_vram_gb": measured.vram_gb,
                "total_disk_gb": measured.disk_gb,
                "min_disk_to_vram_rate": measured.rate,
                "required_disk_gb": required_disk_gb,
            },
        )

    @staticmethod
    def no_payout_because_no_unrented_capacity_for_gpu_count(
        result: JobResult, count_bucket: int | None
    ) -> MinerLogLine:
        return MinerLogLine._no_payout(
            result,
            reason=ZeroIncentiveReason.NO_UNRENTED_CAPACITY_FOR_GPU_COUNT,
            message=(
                f"No unrented incentive: there is currently no unrented-incentive capacity for "
                f"{result.gpu_count}x {result.gpu_model} (its GPU-count tier has no cap this "
                f"cycle). Rent this executor out to earn."
            ),
            extra_fields={
                "count_bucket": count_bucket,
                "max_cap": result.max_cap,
                "unrented_cap_multiplier": result.unrented_cap_multiplier,
                "total_rental_cost": result.total_rental_cost,
            },
        )

    @staticmethod
    def no_payout_because_nvidia_driver_below_minimum(result: JobResult) -> MinerLogLine:
        return MinerLogLine._no_payout(
            result,
            reason=ZeroIncentiveReason.NVIDIA_DRIVER_BELOW_MINIMUM,
            message=(
                f"No unrented incentive: NVIDIA driver {result.nvidia_driver_version} is below "
                "the minimum version required by the network. Upgrade the NVIDIA driver on this "
                "executor to earn the unrented incentive."
            ),
            extra_fields={
                "nvidia_driver_version": result.nvidia_driver_version,
                "driver_multiplier": result.driver_multiplier,
            },
        )

    @staticmethod
    def no_payout_because_sysbox_not_enabled(result: JobResult) -> MinerLogLine:
        return MinerLogLine._no_payout(
            result,
            reason=ZeroIncentiveReason.SYSBOX_NOT_ENABLED,
            message=(
                "No unrented incentive: this executor does not run the sysbox runtime, which is "
                "required for unrented incentive. Enable sysbox on this executor to earn."
            ),
            extra_fields={"sysbox_runtime": result.sysbox_runtime},
        )

    # ── Calculation reports: the per-cycle lines every scored node gets ───────

    @staticmethod
    def mining_score_calculated(result: JobResult, is_rented_after_cutoff: bool) -> MinerLogLine:
        return MinerLogLine(
            message=(
                "Mining score is calculated successfully. Formula: score * gpu_portion * gpu_count "
                "/ total_gpu_count * sysbox_multiplier * uptime_multiplier * driver_multiplier"
            ),
            fields={
                "executor_id": str(result.executor_info.uuid),
                "gpu_model": result.gpu_model,
                "gpu_count": result.gpu_count,
                "sysbox_multiplier": result.sysbox_multiplier,
                "driver_multiplier": result.driver_multiplier,
                "nvidia_driver_version": result.nvidia_driver_version,
                "uptime_multiplier": result.uptime_multiplier,
                "mining_score": result.mining_score,
                "gpu_portion": result.gpu_portion,
                "total_gpu_count": result.total_gpu_count,
                "rental_created": result.rental_created_at,
                "is_rented_after_cutoff": is_rented_after_cutoff,
            },
        )

    @staticmethod
    def mining_incentive_calculated(
        hotkey: str, result: JobResult, total_mining_score: float, mining_share: float
    ) -> MinerLogLine:
        return MinerLogLine(
            message=(
                "Incentive score is calculated successfully. Formula: mining_share * mining_score "
                "/ total_mining_score"
            ),
            fields={
                "hotkey": hotkey,
                "executor_id": str(result.executor_info.uuid),
                "mining_score": result.mining_score,
                "total_mining_score": total_mining_score,
                "mining_share": mining_share,
                "gpu_model": result.gpu_model,
                "gpu_count": result.gpu_count,
                "incentive": result.incentive,
            },
        )

    @staticmethod
    def rental_incentive_calculated(hotkey: str, result: JobResult, count_bucket: int) -> MinerLogLine:
        return MinerLogLine(
            message=(
                "Rental price incentive for executor is calculated successfully. Formula: "
                "rental_share * gpu_count * effective_rate / total_rental_cost"
            ),
            fields={
                "hotkey": hotkey,
                "executor_id": str(result.executor_info.uuid),
                "gpu_model": result.gpu_model,
                "gpu_count": result.gpu_count,
                "hourly_rate": result.hourly_rate,
                "sysbox_runtime": result.sysbox_runtime,
                "sysbox_multiplier": result.sysbox_multiplier,
                "provider_discord_connected": result.provider_discord_connected,
                "nvidia_driver_version": result.nvidia_driver_version,
                "driver_multiplier": result.driver_multiplier,
                "unrented_cap_multiplier": result.unrented_cap_multiplier,
                "effective_rate": result.effective_rate,
                "total_unrented_by_gpu_type": result.total_unrented_by_gpu_type,
                "count_bucket": count_bucket,
                "max_cap": result.max_cap,
                "cap_dilution_applied": result.cap_dilution_applied,
                "rental_share": result.rental_share,
                "burn_share": result.burn_share,
                "incentive": result.incentive,
                "total_rental_cost": result.total_rental_cost,
            },
        )

    @staticmethod
    def mining_score_missing(hotkey: str, result: JobResult) -> MinerLogLine:
        # Internal-error case: scoring finished without a mining score. Should not happen.
        return MinerLogLine(
            message="Mining score is not set for job result. This should not happen.",
            fields={
                "hotkey": hotkey,
                "executor_id": str(result.executor_info.uuid),
                "score": result.score,
                "job_score": result.job_score,
                "gpu_model": result.gpu_model,
                "gpu_count": result.gpu_count,
            },
        )
