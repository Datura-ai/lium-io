"""Everything a miner sees in their "Incentive Scores Calculation Logs" — one catalog.

WHERE A NODE EARNS (two "pools" of subnet emission; the rest is burned):
  - Mining pool   — for RENTED executors. Paid by a mining score (GPU model/count
                    times sysbox / driver / uptime multipliers).
  - Unrented pool — for IDLE executors of an eligible GPU model. Paid as if the GPU
                    were rented, based on its rental market value (rental-share).

HOW A NODE PICKS A POOL:
  rented?                                              -> mining pool (always earns)
  idle AND model in program AND price ok AND capacity? -> unrented pool (earns)
  otherwise                                            -> 0 incentive (reason below)

WHAT THIS CATALOG HOLDS (the miner-facing log block, JobResult.incentive_logs,
delivered via MACHINE_SPEC_CHANNEL):

1. ZERO-INCENTIVE REASONS — each records the fact "this executor gets NO payout
   because <reason>" in the miner's log (`no_payout_because_*` builders):
   Group A — earns nothing in EITHER pool (built by `_reason_excluded_from_both_pools`):
     spot tier, Discord not connected, paused for new rentals, running own default job
   Group B — idle but does not qualify for the unrented pool:
     GPU model not in the unrented program (earns only when rented),
     price above the market soft limit (lower the price to earn),
     no unrented capacity for that GPU-count tier this cycle

2. CALCULATION REPORTS — the per-cycle score/incentive lines every scored node gets:
     mining_score_calculated, mining_incentive_calculated,
     rental_incentive_calculated, mining_score_missing (internal-error case)

The scoring code (rental_price.py / default.py) detects each condition where its
data naturally lives (some per-executor upfront, some only after cohort aggregation)
and calls the matching builder below, then `.write_to_miner_log(result)`. Open THIS
file to see everything a miner can be told and exactly how each message reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from core.utils import _m, get_extra_info

if TYPE_CHECKING:
    from services.task_service import JobResult


class ZeroIncentiveReason(BaseModel):
    """One reason a validated executor earns 0 subnet incentive."""

    reason: str                                                    # machine-readable code, e.g. "spot_tier"
    message_for_miner: str                                         # plain-English, shown to the miner
    miner_log_fields: dict[str, Any] = Field(default_factory=dict)
    internal_log_message: str | None = None                        # set only for both-pools exclusions
    internal_log_fields: dict[str, Any] = Field(default_factory=dict)

    def write_to_miner_log(self, result: JobResult) -> None:
        """Append this reason to the executor's customer-facing incentive log.

        Lands in JobResult.incentive_logs -> the "Incentive Scores Calculation Logs"
        block delivered to the miner via MACHINE_SPEC_CHANNEL (DAH-2327).
        """
        info = {
            "executor_id": str(result.executor_info.uuid),
            "gpu_model": result.gpu_model,
            "gpu_count": result.gpu_count,
            "reason": self.reason,
            "incentive": 0.0,
            **self.miner_log_fields,
        }
        result.incentive_logs.append(
            _m(self.message_for_miner, extra=get_extra_info(info)).to_full_string()
        )


# ── Group A: excluded from BOTH pools (mining + unrented) — earns nothing ─────

def no_payout_because_spot_tier() -> ZeroIncentiveReason:
    return ZeroIncentiveReason(
        reason="spot_tier",
        message_for_miner=(
            "No subnet incentive: this executor is on the spot tier, and spot-tier "
            "executors do not earn subnet incentive."
        ),
        internal_log_message="Executor excluded from both pools - spot tier",
    )


def no_payout_because_discord_not_connected(is_connected: bool) -> ZeroIncentiveReason:
    return ZeroIncentiveReason(
        reason="provider_discord_not_connected",
        message_for_miner=(
            "No subnet incentive: provider Discord is not connected. Connect your "
            "provider Discord for this executor to start earning incentive."
        ),
        internal_log_message="Executor excluded from both pools - provider Discord not connected",
        internal_log_fields={"provider_discord_connected": is_connected},
    )


def no_payout_because_paused_for_new_rentals() -> ZeroIncentiveReason:
    return ZeroIncentiveReason(
        reason="new_rentals_paused",
        message_for_miner=(
            "No subnet incentive: this executor is paused for new rentals and earns "
            "nothing while paused. Resume new rentals to start earning again."
        ),
        internal_log_message="Executor excluded from both pools - paused for new rentals",
    )


def no_payout_because_running_own_default_job() -> ZeroIncentiveReason:
    return ZeroIncentiveReason(
        reason="miner_default_job",
        message_for_miner=(
            "No subnet incentive: this executor is running your own default job instead "
            "of a Lium job, and executors on your own job do not earn subnet incentive."
        ),
        internal_log_message="Executor excluded from both pools - running miner's own default job",
    )


# ── Group B: idle but not qualified for the unrented pool ─────────────────────

def no_payout_because_gpu_model_not_in_unrented_program(gpu_model: str) -> ZeroIncentiveReason:
    return ZeroIncentiveReason(
        reason="gpu_model_not_eligible_for_unrented_incentive",
        message_for_miner=(
            f"No subnet incentive while idle: GPU model {gpu_model} is not part of the "
            "unrented (idle-node) incentive program, so this executor earns nothing "
            "unless rented. Rent it out to earn."
        ),
    )


def no_payout_because_price_above_market_soft_limit(price_per_gpu: float, market_p90: float, rate: float) -> ZeroIncentiveReason:
    soft_limit = round(market_p90 * rate, 4)
    return ZeroIncentiveReason(
        reason="price_above_market_p90_soft_limit",
        message_for_miner=(
            f"No unrented incentive: your price ${price_per_gpu}/GPU/h is above the market "
            f"soft price limit ${soft_limit} (90th-percentile market rate ${market_p90} x "
            f"{rate}). Lower the price to ${soft_limit} or below to earn the unrented incentive."
        ),
        miner_log_fields={
            "price_per_gpu": price_per_gpu,
            "machine_price_p90": market_p90,
            "soft_limit_rate": rate,
            "soft_limit_threshold": soft_limit,
        },
    )


def no_payout_because_no_unrented_capacity_for_gpu_count(
    gpu_count: int,
    gpu_model: str,
    count_bucket: int | None,
    max_cap: int | None,
    cap_multiplier: float | None,
    total_rental_cost: float,
) -> ZeroIncentiveReason:
    return ZeroIncentiveReason(
        reason="no_unrented_capacity_for_gpu_count",
        message_for_miner=(
            f"No unrented incentive: there is currently no unrented-incentive capacity for "
            f"{gpu_count}x {gpu_model} (its GPU-count tier has no cap this cycle). Rent this "
            f"executor out to earn."
        ),
        miner_log_fields={
            "count_bucket": count_bucket,
            "max_cap": max_cap,
            "unrented_cap_multiplier": cap_multiplier,
            "total_rental_cost": total_rental_cost,
        },
    )


# ── Calculation reports: the per-cycle lines every scored node gets ───────────

class MinerLogLine(BaseModel):
    """One fully-built line for the miner-facing incentive log."""

    message: str
    fields: dict[str, Any] = Field(default_factory=dict)

    def as_internal_log(self):
        """The same line as an `_m` object, for mirroring into the internal logger."""
        return _m(self.message, extra=get_extra_info(self.fields))

    def write_to_miner_log(self, result: JobResult) -> None:
        result.incentive_logs.append(self.as_internal_log().to_full_string())


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


def rental_incentive_calculated(hotkey: str, result: JobResult, bucket: int) -> MinerLogLine:
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
            "count_bucket": bucket,
            "max_cap": result.max_cap,
            "cap_dilution_applied": result.cap_dilution_applied,
            "rental_share": result.rental_share,
            "burn_share": result.burn_share,
            "incentive": result.incentive,
            "total_rental_cost": result.total_rental_cost,
        },
    )


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
