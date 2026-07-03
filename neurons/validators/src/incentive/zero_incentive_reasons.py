"""Every reason a validated executor earns 0 subnet incentive — the single catalog.

The scoring code (rental_price.py) detects each condition where its data naturally
lives (some per-executor upfront, some only after cohort aggregation) and calls the
matching builder below. Open THIS file to see the full list of what a miner can be
told and exactly how each message reads — no need to trace the scoring flow.

Each builder returns a `ZeroIncentiveReason` carrying:
  - the machine-readable `reason` code (also used by internal Loki dashboards),
  - `message_for_miner`: the plain-English line shown in the miner's incentive log,
  - `miner_log_fields`: structured fields attached to that miner-facing log,
  - `internal_log_message` / `internal_log_fields`: the separate internal observability
    log emitted for the "excluded from both pools" reasons (None for the rest).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ZeroIncentiveReason(BaseModel):
    """One reason a validated executor earns 0 subnet incentive."""

    reason: str                                                    # machine-readable code, e.g. "spot_tier"
    message_for_miner: str                                         # plain-English, shown to the miner
    miner_log_fields: dict[str, Any] = Field(default_factory=dict)
    internal_log_message: str | None = None                        # set only for both-pools exclusions
    internal_log_fields: dict[str, Any] = Field(default_factory=dict)


# ── Excluded from BOTH pools (mining + unrented) — earns nothing at all ───────

def spot_tier() -> ZeroIncentiveReason:
    return ZeroIncentiveReason(
        reason="spot_tier",
        message_for_miner=(
            "No subnet incentive: this executor is on the spot tier, and spot-tier "
            "executors do not earn subnet incentive."
        ),
        internal_log_message="Executor excluded from both pools - spot tier",
    )


def provider_discord_not_connected(is_connected: bool) -> ZeroIncentiveReason:
    return ZeroIncentiveReason(
        reason="provider_discord_not_connected",
        message_for_miner=(
            "No subnet incentive: provider Discord is not connected. Connect your "
            "provider Discord for this executor to start earning incentive."
        ),
        internal_log_message="Executor excluded from both pools - provider Discord not connected",
        internal_log_fields={"provider_discord_connected": is_connected},
    )


def new_rentals_paused() -> ZeroIncentiveReason:
    return ZeroIncentiveReason(
        reason="new_rentals_paused",
        message_for_miner=(
            "No subnet incentive: this executor is paused for new rentals and earns "
            "nothing while paused. Resume new rentals to start earning again."
        ),
        internal_log_message="Executor excluded from both pools - paused for new rentals",
    )


def miner_default_job() -> ZeroIncentiveReason:
    return ZeroIncentiveReason(
        reason="miner_default_job",
        message_for_miner=(
            "No subnet incentive: this executor is running your own default job instead "
            "of a Lium job, and executors on your own job do not earn subnet incentive."
        ),
        internal_log_message="Executor excluded from both pools - running miner's own default job",
    )


# ── Unrented-pool-only reasons — earns only when rented / repriced ────────────

def gpu_model_not_in_unrented_program(gpu_model: str) -> ZeroIncentiveReason:
    return ZeroIncentiveReason(
        reason="gpu_model_not_eligible_for_unrented_incentive",
        message_for_miner=(
            f"No subnet incentive while idle: GPU model {gpu_model} is not part of the "
            "unrented (idle-node) incentive program, so this executor earns nothing "
            "unless rented. Rent it out to earn."
        ),
    )


def price_over_soft_limit(price_per_gpu: float, market_p90: float, rate: float) -> ZeroIncentiveReason:
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


def no_unrented_capacity(
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
