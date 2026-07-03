"""Rental price incentive algorithm implementation.

This module implements the three-phase rental price incentive algorithm that
rewards unrented high-end GPUs based on their rental market value.

The system uses per-`(base_model, gpu_count_bucket)` caps to dilute incentives
when supply exceeds demand for specific GPU configurations. Each base model's
cap is a `dict[gpu_count_bucket, cap]`; an empty dict opts the family out of
rental subsidy. See `incentive/config.py:MAX_UNRENTED_GPUS_BY_TYPE`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import bittensor
from pydantic import BaseModel, Field

from core.config import get_total_burn_emission, settings, shared_client
from core.utils import _m, get_extra_info, get_logger
from incentive.config import BASE_GPU_MAP
from incentive.eligibility import is_missing_discord_after_cutoff

if TYPE_CHECKING:
    from incentive.config import IncentiveConfig
    from services.redis_service import RedisService
from incentive.utils import get_hourly_rate
from incentive.default import DefaultIncentive, get_min_driver_multiplier
from incentive.price_provider import PriceProvider
from services.const import TEMPO, SECONDS_PER_BLOCK, FIXED_RATIO, DEFAULT_JOB_OWNER_MINER
from services.task_service import JobResult

logger = get_logger(__name__)

# DAH-2250 — unrented incentive soft price limit (section 3 of the market-pricing
# proposal). An unrented executor whose price_per_gpu exceeds the market p90 times
# this multiplier forfeits the unrented rental incentive but stays active.
SOFT_LIMIT_PRICE_RATE = 1.1


# ── Snapshot models ──────────────────────────────────────────────────────────

class GpuBucketRentalState(BaseModel):
    """Per-`(base_model, gpu_count_bucket)` rental state."""

    unrented_count: int
    max_cap: int
    cap_multiplier: float
    weighted_rate_sum: float  # sum(gpu_count * hourly_rate * sysbox_multiplier) in this bucket


class RentalMiningState(BaseModel):
    total_gpu_count: int
    total_mining_score: float
    # Per full GPU model totals used by DefaultIncentive.calculate_executor_score.
    # This must be per `JobResult.gpu_model` (not base model), so the default mining
    # score formula can normalize consistently for both real and estimated jobs.
    total_gpu_model_count_map: dict[str, int] = Field(default_factory=dict)


class RentalShareState(BaseModel):
    total_rental_cost: float
    # Bucket-keyed state. Key format: f"{base_model}·{bucket}".
    by_bucket: dict[str, GpuBucketRentalState] = Field(default_factory=dict)


class RentalPriceSnapshot(BaseModel):
    epoch_subnet_emission: float
    rental_share: float
    burn_share: float
    mining: RentalMiningState
    rental: RentalShareState


class ExecutorEstimateParams(BaseModel):
    gpu_model: str
    gpu_count: int = 1
    is_rented: bool = False
    gpu_splitting: bool = False
    gpu_splitting_min_count: int | None = None
    sysbox_runtime: bool = True
    collateral_deposited: bool = True


# ── Estimate model ────────────────────────────────────────────────────────────

class RentalPriceEstimate(BaseModel):
    gpu_model: str
    base_model: str
    gpu_count: int
    is_rented: bool
    usd_per_epoch: float
    count_bucket: int | None = None                     # gpu_count_bucket the executor was placed into
    mining_score: float | None = None                   # Score for mining pool for scoring logic
    sysbox_multiplier: float | None = None              # Multiplier for sysbox runtime for scoring logic
    provider_discord_connected: bool = True             # Whether provider Discord is connected for scoring logic
    uptime_multiplier: float | None = None              # Multiplier for uptime
    gpu_portion: float | None = None                    # Portion of the GPU model for scoring logic
    total_gpu_count: int | None = None                  # Total number of GPUs of the same model
    incentive: float | None = None                      # Incentive score for the executor in this cycle

    # V2 incentive relevant fields
    effective_rate: float | None = None                 # Effective rate for the executor in this cycle for scoring logic
    hourly_rate: float | None = None                    # Hourly rate for the executor in this cycle for scoring logic
    max_cap: int | None = None                          # Max cap for GPU counts in this cycle for scoring logic
    total_unrented_by_gpu_type: float | None = None     # Weighted GPU count for the executor in this cycle for scoring logic
    cap_dilution_applied: bool | None = None            # Whether the cap dilution is applied for the executor in this cycle for scoring logic
    eligible_for_rental_share: bool = False
    unrented_cap_multiplier: float | None = None        # Cap dilution multiplier: min(count, cap) / count
    rental_share: float | None = None                   # Rental share for the executor in this cycle for scoring logic
    burn_share: float | None = None                     # Burn share for the executor in this cycle for scoring logic
    total_rental_cost: float | None = None              # Total rental cost for the executor in this cycle for scoring logic


class RentalPriceIncentive(DefaultIncentive):
    """Rental price incentive algorithm.

    Implements a three-phase algorithm:
    - Phase 1: Exclude unrented eligible GPUs from mining scores
    - Phase 2: Calculate dynamic emission splits based on rental costs
    - Phase 3: Distribute weights across burn/mining/rental pools

    Cap dilution is applied per `(base_model, gpu_count_bucket)`. Buckets are
    derived from each executor's `gpu_splitting_min_count` (when GPU splitting
    is enabled) or its `gpu_count`.
    """

    price_provider: PriceProvider = PriceProvider()

    def __init__(self, *args, snapshot: "RentalPriceSnapshot | None" = None, **kwargs):
        """Initialize rental price incentive algorithm.

        Args:
            config: Incentive configuration with rental_incentive_gpu_types,
                   max_unrented_gpus (dict[base_model, dict[bucket, cap]]),
                   and rental_prices_per_hour
            redis_service: Redis service for accessing shared state
            burn_service: Burn emission distribution service
            snapshot: Optional snapshot to seed accumulated state (for estimation)
        """
        super().__init__(*args, **kwargs)

        # Bucket-keyed state. Key = (base_model, bucket).
        self.unrented_count_by_bucket: dict[tuple[str, int], int] = {}
        self._weighted_rate_sum_by_bucket: dict[tuple[str, int], float] = {}
        self.cap_multiplier_by_bucket: dict[tuple[str, int], float] = {}
        self.total_rental_cost = 0.0
        self.rental_share = 0.0
        self.burn_share = 0.0
        self.epoch_subnet_emission: float = 0.0
        # Store the snapshot so estimation can derive per-model totals from it.
        self._seed_snapshot = snapshot

        # validate configs
        for base_model in self.config.rental_incentive_gpu_types:
            assert base_model in BASE_GPU_MAP.values(), f"Base model {base_model} not found in BASE_GPU_MAP"

        for gpu_type in self.config.rental_prices_per_hour.keys():
            assert gpu_type in BASE_GPU_MAP.keys(), f"GPU type {gpu_type} not found in BASE_GPU_MAP"

        if snapshot:
            self._seed_state_from_snapshot(snapshot)
            self.total_mining_score = snapshot.mining.total_mining_score
            self.epoch_subnet_emission = snapshot.epoch_subnet_emission

    def _seed_state_from_snapshot(self, snapshot: "RentalPriceSnapshot") -> None:
        """Restore bucket-keyed state from a snapshot."""
        for key_str, state in snapshot.rental.by_bucket.items():
            base_model, bucket_str = key_str.rsplit("·", 1)
            key = (base_model, int(bucket_str))
            self.unrented_count_by_bucket[key] = state.unrented_count
            self._weighted_rate_sum_by_bucket[key] = state.weighted_rate_sum

    def get_base_model_for_gpu(self, gpu_model: str) -> str:
        base_model = BASE_GPU_MAP[gpu_model]
        return base_model

    def _is_over_soft_price_limit(self, result: JobResult) -> bool:
        # miner price_per_gpu above the market p90 ceiling (p90 * SOFT_LIMIT_PRICE_RATE)
        price_per_gpu = result.executor_info.price_per_gpu
        if not price_per_gpu:
            return False
        # machine_prices_p90 is partial: GPUs without market data keep full incentive
        p90 = shared_client.config.machine_prices_p90.get(result.gpu_model)
        if not p90:
            return False
        return price_per_gpu > p90 * SOFT_LIMIT_PRICE_RATE

    def _log_soft_price_limit(self, result: JobResult) -> None:
        # structured log for every unrented executor over the p90 soft ceiling
        enforced = settings.ENABLE_UNRENTED_SOFT_PRICE_LIMIT
        p90 = shared_client.config.machine_prices_p90.get(result.gpu_model)
        logger.info(
            _m(
                "Unrented executor over market p90 soft price limit"
                + ("" if enforced else " (shadow only - flag off)"),
                extra={
                    "executor_id": str(result.executor_info.uuid),
                    "gpu_model": result.gpu_model,
                    "gpu_count": result.gpu_count,
                    "price_per_gpu": result.executor_info.price_per_gpu,
                    "machine_price_p90": p90,
                    "soft_limit_rate": SOFT_LIMIT_PRICE_RATE,
                    "soft_limit_threshold": p90 * SOFT_LIMIT_PRICE_RATE if p90 else None,
                    "enforced": enforced,
                    "reason": "price_above_market_p90_soft_limit",
                    "pool": "rental_excluded" if enforced else "rental_kept_shadow",
                },
            )
        )

    def _append_soft_price_limit_incentive_log(self, result: JobResult) -> None:
        # DAH-2327: surface the zero-incentive reason in the customer-facing incentive
        # log (JobResult.incentive_logs -> "Incentive Scores Calculation Logs"), the exact
        # block the miner reads, at the moment the soft price limit zeros the payout.
        p90 = shared_client.config.machine_prices_p90.get(result.gpu_model)
        soft_limit = round(p90 * SOFT_LIMIT_PRICE_RATE, 4)
        result.incentive_logs.append(
            _m(
                f"Unrented incentive set to 0: price ${result.executor_info.price_per_gpu}/GPU/h "
                f"exceeds the market soft price limit ${soft_limit} (p90 ${p90} x "
                f"{SOFT_LIMIT_PRICE_RATE}). Lower the price to ${soft_limit} or below to receive "
                f"the unrented incentive.",
                extra=get_extra_info({
                    "executor_id": str(result.executor_info.uuid),
                    "gpu_model": result.gpu_model,
                    "gpu_count": result.gpu_count,
                    "price_per_gpu": result.executor_info.price_per_gpu,
                    "machine_price_p90": p90,
                    "soft_limit_rate": SOFT_LIMIT_PRICE_RATE,
                    "soft_limit_threshold": soft_limit,
                    "reason": "price_above_market_p90_soft_limit",
                }),
            ).to_full_string()
        )

    @staticmethod
    def _resolve_bucket(result: JobResult, cap_spec: dict[int, int]) -> int:
        """Pick the gpu_count bucket the executor is rated against.

        Non-splitting executor always uses `gpu_count`. A splitting-capable
        executor prefers the `gpu_count` bucket when it is configured with a
        positive cap, otherwise falls back to the `min_count` tier.
        """
        if not (result.supports_gpu_splitting and result.gpu_splitting_min_count):
            return result.gpu_count
        if cap_spec.get(result.gpu_count, 0) > 0:
            return result.gpu_count
        return result.gpu_splitting_min_count

    @staticmethod
    def _bucket_key_str(base_model: str, bucket: int) -> str:
        return f"{base_model}·{bucket}"

    async def _pre_process_job_result(self, hotkey: str, result: JobResult):
        """Aggregate per-`(base_model, bucket)` metrics for the rental-share
        algorithm. Bucket resolution is symmetric with the rate-resolution path
        so split-capable executors land in the bucket of their
        `gpu_splitting_min_count`.
        """
        if not result.is_successful:
            return

        await super()._pre_process_job_result(hotkey, result)

        # Check if GPU is eligible
        base_model = self.get_base_model_for_gpu(result.gpu_model)
        if base_model not in self.config.rental_incentive_gpu_types:
            return

        #  calculate unrented gpu count that's eligible for rental price incentive
        if result.eligible_for_rental_share:
            # update result state
            result.hourly_rate = get_hourly_rate(
                result.gpu_model, result.gpu_count,
                self.config.gpu_count_custom_prices, self.config.rental_prices_per_hour,
            )
            # GPU splitting: always pick the best of the bundle rate vs min-count rate
            if result.supports_gpu_splitting and result.gpu_splitting_min_count:
                rate_for_min = get_hourly_rate(
                    result.gpu_model, result.gpu_splitting_min_count,
                    self.config.gpu_count_custom_prices, self.config.rental_prices_per_hour,
                )
                result.hourly_rate = max(result.hourly_rate, rate_for_min)

            # Sysbox penalty is applied later via effective_rate, not baked into hourly_rate
            result.sysbox_multiplier = 1.0 if result.sysbox_runtime else 1 - settings.PORTION_FOR_SYSBOX_UNRENTED

            # Minimum NVIDIA driver penalty: applied later via effective_rate
            result.driver_multiplier = get_min_driver_multiplier(result.nvidia_driver_version)

            cap_spec = self.config.max_unrented_gpus.get(base_model, {})
            bucket = self._resolve_bucket(result, cap_spec)
            max_cap = cap_spec.get(bucket, 0)
            result.count_bucket = bucket
            result.max_cap = max_cap

            # accumulate raw unrented GPU count and weighted rate sum per bucket
            if result.hourly_rate > 0 and max_cap > 0:
                key = (base_model, bucket)
                self.unrented_count_by_bucket[key] = (
                    self.unrented_count_by_bucket.get(key, 0) + result.gpu_count
                )
                self._weighted_rate_sum_by_bucket[key] = (
                    self._weighted_rate_sum_by_bucket.get(key, 0.0)
                    + result.gpu_count
                    * result.hourly_rate
                    * result.sysbox_multiplier
                    * result.driver_multiplier
                )

    async def _on_finish_pre_process(self) -> None:
        """Callback after pre-processing all job results.

        - Calculate rental share
        """
        # Step 1: cap multiplier per (base_model, bucket).
        for (base_model, bucket), unrented_count in self.unrented_count_by_bucket.items():
            max_cap = self.config.max_unrented_gpus.get(base_model, {}).get(bucket, 0)
            if unrented_count > 0 and max_cap > 0:
                self.cap_multiplier_by_bucket[(base_model, bucket)] = (
                    min(unrented_count, max_cap) / unrented_count
                )

        # Step 2: total_rental_cost from per-bucket weighted rate sums.
        for key, weighted_sum in self._weighted_rate_sum_by_bucket.items():
            cap_mult = self.cap_multiplier_by_bucket.get(key, 0.0)
            self.total_rental_cost += cap_mult * weighted_sum

        rental_share_raw = await self._calculate_rental_share(self.total_rental_cost)

        # Cap rental_share at the burn-emission share (sourced from shared config, DAH-2274)
        total_burn_emission = get_total_burn_emission()
        rental_share_capped = rental_share_raw > total_burn_emission
        self.rental_share = min(rental_share_raw, total_burn_emission)

        if rental_share_capped:
            logger.warning(
                _m(
                    "Rental share capped at max burn emission",
                    extra={
                        "rental_share_raw": rental_share_raw,
                        "rental_share_capped": self.rental_share,
                        "max_cap": total_burn_emission,
                        "hint": f"Rental share would have been {rental_share_raw:.4f} but capped at {total_burn_emission}",
                    },
                )
            )

        # Calculate emission splits
        self.burn_share = total_burn_emission - self.rental_share
        logger.info(
            _m(
                "Final emission splits calculated",
                extra={
                    "rental_share": self.rental_share,
                    "burn_share": self.burn_share,
                    "total_rental_cost": self.total_rental_cost,
                },
            )
        )

    async def _post_process_job_result(self, hotkey: str, result: JobResult):
        """Process a job result.

        Calculate incentive score for the executor.

        Args:
            result: Job execution result to process
        """
        if not result.eligible_for_rental_share:
            return await super()._post_process_job_result(hotkey, result) # use default incentive logic.

        # state updates
        base_model = self.get_base_model_for_gpu(result.gpu_model)
        if result.count_bucket is not None:
            bucket = result.count_bucket
        else:
            cap_spec = self.config.max_unrented_gpus.get(base_model, {})
            bucket = self._resolve_bucket(result, cap_spec)
        key = (base_model, bucket)
        result.total_unrented_by_gpu_type = self.unrented_count_by_bucket.get(key, 0)
        result.cap_dilution_applied = result.total_unrented_by_gpu_type > result.max_cap
        result.rental_share = self.rental_share
        result.burn_share = self.burn_share
        result.total_rental_cost = self.total_rental_cost
        result.unrented_cap_multiplier = self.cap_multiplier_by_bucket.get(key, 0.0)
        result.effective_rate = (
            result.hourly_rate
            * result.unrented_cap_multiplier
            * result.sysbox_multiplier
            * result.driver_multiplier
        )

        # calculate incentive score
        result.incentive = (
            result.rental_share * result.gpu_count * result.effective_rate / result.total_rental_cost
            if result.total_rental_cost > 0 else 0.0
        )

        # update incentive logs
        result.incentive_logs.append(
            _m(
                "Rental price incentive for executor is calculated successfully. Formula: rental_share * gpu_count * effective_rate / total_rental_cost",
                extra=get_extra_info({
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
                }),
            ).to_full_string()
        )

        # aggregate miner incentives
        self.miner_incentives[hotkey] = self.miner_incentives.get(hotkey, 0.0) + result.incentive

    async def calculate_executor_score(
        self,
        job_result: JobResult,
    ) -> JobResult:
        """Calculate score for a single executor/job result.

        Phase 1: Unrented eligible GPUs are excluded from mining emission
        by returning score = 0. All other GPUs use normal scoring logic.

        Eligibility is determined by whether the GPU type has any positive
        bucket cap in max_unrented_gpus.

        Args:
            total_gpu_model_count_map: Mapping of GPU models to total counts
            job_result: Job execution result to score

        Returns:
            Calculated score (0 for unrented eligible GPUs, normal score otherwise)
        """
        if job_result.is_spot:
            logger.info(
                _m(
                    "Executor excluded from both pools - spot tier",
                    extra={
                        "executor_id": str(job_result.executor_info.uuid),
                        "gpu_model": job_result.gpu_model,
                        "gpu_count": job_result.gpu_count,
                        "reason": "spot_tier",
                        "score": 0,
                        "pool": "none",
                    },
                )
            )
            job_result.mining_score = 0
            job_result.eligible_for_rental_share = False
            return job_result

        if is_missing_discord_after_cutoff(job_result):
            logger.info(
                _m(
                    "Executor excluded from both pools - provider Discord not connected",
                    extra={
                        "executor_id": str(job_result.executor_info.uuid),
                        "gpu_model": job_result.gpu_model,
                        "gpu_count": job_result.gpu_count,
                        "provider_discord_connected": job_result.provider_discord_connected,
                        "reason": "provider_discord_not_connected",
                        "score": 0,
                        "pool": "none",
                    },
                )
            )
            job_result.mining_score = 0
            job_result.eligible_for_rental_share = False
            return job_result

        if job_result.is_new_rentals_paused and not job_result.is_rented:
            logger.info(
                _m(
                    "Executor excluded from both pools - paused for new rentals",
                    extra={
                        "executor_id": str(job_result.executor_info.uuid),
                        "gpu_model": job_result.gpu_model,
                        "gpu_count": job_result.gpu_count,
                        "reason": "new_rentals_paused",
                        "score": 0,
                        "pool": "none",
                    },
                )
            )
            job_result.mining_score = 0
            job_result.eligible_for_rental_share = False
            return job_result

        if job_result.default_job_owner == DEFAULT_JOB_OWNER_MINER and not job_result.is_rented:
            logger.info(
                _m(
                    "Executor excluded from both pools - running miner's own default job",
                    extra={
                        "executor_id": str(job_result.executor_info.uuid),
                        "gpu_model": job_result.gpu_model,
                        "gpu_count": job_result.gpu_count,
                        "reason": "miner_default_job",
                        "score": 0,
                        "pool": "none",
                    },
                )
            )
            job_result.mining_score = 0
            job_result.eligible_for_rental_share = False
            return job_result

        # Check if GPU is unrented and eligible (has positive cap in max_unrented_gpus)
        base_model = self.get_base_model_for_gpu(job_result.gpu_model)
        eligible_for_rental_share = (
            not job_result.is_rented
            and (base_model in self.config.rental_incentive_gpu_types)
            and (job_result.score > 0 or job_result.job_score > 0)
        )

        # DAH-2250 soft price limit: an otherwise-eligible unrented executor priced
        # above the market p90 ceiling forfeits the unrented incentive (node stays
        # active). While the flag is off we only log the would-be exclusion (shadow).
        if eligible_for_rental_share and self._is_over_soft_price_limit(job_result):
            self._log_soft_price_limit(job_result)
            if settings.ENABLE_UNRENTED_SOFT_PRICE_LIMIT:
                eligible_for_rental_share = False
                self._append_soft_price_limit_incentive_log(job_result)

        job_result.eligible_for_rental_share = eligible_for_rental_share
        if job_result.eligible_for_rental_share:
            logger.info(
                _m(
                    "Executor excluded from mining pool - unrented eligible GPU",
                    extra={
                        "executor_id": str(job_result.executor_info.uuid),
                        "gpu_model": job_result.gpu_model,
                        "gpu_count": job_result.gpu_count,
                        "reason": "unrented_and_eligible",
                        "score": 0,
                        "pool": "rental_only",
                    },
                )
            )
            job_result.mining_score = 0
            return job_result # Exclude from mining pool

        if not job_result.is_rented:
            job_result.mining_score = 0
            return job_result

        # For rented or non-eligible GPUs, use parent's default scoring logic
        return await super().calculate_executor_score(job_result)

    async def estimate_executor(
        self,
        params: ExecutorEstimateParams,
    ) -> RentalPriceEstimate:
        """Estimate USD/epoch reward for a hypothetical executor from this instance's snapshot.

        This uses the same internal 3-stage pipeline as real scoring:
        `_pre_process_job_result` -> `_on_finish_pre_process` -> `_post_process_job_result`.
        """
        # Snapshot is required so we can:
        # 1) seed rental-phase state (unrented counts, weighted rate sums, etc.)
        # 2) get per-model GPU totals for DefaultIncentive's normalization.
        if self._seed_snapshot is None:
            raise ValueError("estimate_executor requires RentalPriceIncentive initialized with snapshot=")

        from datura.requests.miner_requests import ExecutorSSHInfo

        gpu_model = params.gpu_model
        gpu_count = params.gpu_count
        is_rented = params.is_rented

        base_model = BASE_GPU_MAP.get(gpu_model)
        cap_spec = self.config.max_unrented_gpus.get(base_model, {}) if base_model else {}
        eligible_for_unrented_estimate = (
            base_model is not None and any(v > 0 for v in cap_spec.values())
        )
        if base_model is None or (not is_rented and not eligible_for_unrented_estimate):
            return RentalPriceEstimate(
                gpu_model=gpu_model,
                base_model=base_model or gpu_model,
                gpu_count=gpu_count,
                is_rented=is_rented,
                usd_per_epoch=0.0,
                eligible_for_rental_share=False,
            )

        # Build per-model totals including the hypothetical executor.
        # DefaultIncentive.calculate_executor_score needs this per `JobResult.gpu_model`.
        base_total_map = self._seed_snapshot.mining.total_gpu_model_count_map or {}
        total_map_with_hypo = dict(base_total_map)
        total_map_with_hypo[gpu_model] = total_map_with_hypo.get(gpu_model, 0) + gpu_count
        self.total_gpu_model_count_map = total_map_with_hypo

        fake_result = JobResult(
            executor_info=ExecutorSSHInfo(
                uuid="estimate",
                address="0.0.0.0",
                port=0,
                ssh_username="",
                ssh_port=0,
                python_path="",
                root_dir="",
            ),
            score=1.0,
            job_score=1.0,
            job_batch_id="estimate",
            log_status="",
            log_text="",
            gpu_model=gpu_model,
            gpu_count=gpu_count,
            is_rented=is_rented,
            supports_gpu_splitting=params.gpu_splitting,
            gpu_splitting_min_count=params.gpu_splitting_min_count,
            collateral_deposited=params.collateral_deposited,
            sysbox_runtime=params.sysbox_runtime,
            # Price estimates assume a compliant driver (no requirement penalty).
            nvidia_driver_version=settings.MIN_NVIDIA_DRIVER_VERSION,
        )

        await self._pre_process_job_result("estimate", fake_result)
        await self._on_finish_pre_process()
        await self._post_process_job_result("estimate", fake_result)

        return RentalPriceEstimate(
            gpu_model=gpu_model,
            base_model=base_model,
            gpu_count=fake_result.gpu_count,
            is_rented=fake_result.is_rented,
            usd_per_epoch=(fake_result.incentive or 0.0) * self.epoch_subnet_emission * FIXED_RATIO,
            count_bucket=fake_result.count_bucket,
            mining_score=fake_result.mining_score,
            sysbox_multiplier=fake_result.sysbox_multiplier,
            provider_discord_connected=fake_result.provider_discord_connected,
            uptime_multiplier=fake_result.uptime_multiplier,
            gpu_portion=fake_result.gpu_portion,
            total_gpu_count=fake_result.total_gpu_count,
            incentive=fake_result.incentive,
            effective_rate=fake_result.effective_rate,
            hourly_rate=fake_result.hourly_rate,
            max_cap=fake_result.max_cap,
            total_unrented_by_gpu_type=fake_result.total_unrented_by_gpu_type,
            cap_dilution_applied=fake_result.cap_dilution_applied,
            eligible_for_rental_share=fake_result.eligible_for_rental_share or False,
            unrented_cap_multiplier=fake_result.unrented_cap_multiplier,
            rental_share=fake_result.rental_share,
            burn_share=fake_result.burn_share,
            total_rental_cost=fake_result.total_rental_cost,
        )

    async def _calculate_rental_share(self, total_rental_cost: float) -> float:
        """Calculate rental emission share (X).

        Formula:
        epoch_subnet_emission = TEMPO * tao_price * alpha_rate
        rental_share = (total_rental_cost * (TEMPO * SECONDS_PER_BLOCK) / 3600
                       / FIXED_RATIO / epoch_subnet_emission)

        Args:
            total_rental_cost: Total rental cost in USD per hour

        Returns:
            Rental emission share (0 to 0.87)
        """
        # If seeded from a snapshot, epoch_subnet_emission is already correct — skip price fetch.
        if self._seed_snapshot is not None:
            epoch_subnet_emission = self.epoch_subnet_emission
        else:
            tao_price = await self.price_provider.get_tao_price()
            alpha_rate = await self.price_provider.get_alpha_rate()

            if tao_price is None or alpha_rate is None:
                logger.warning(
                    _m(
                        "Failed to fetch TAO price or alpha rate - falling back to 0 rental share",
                        extra={
                            "tao_price": tao_price,
                            "alpha_rate": alpha_rate,
                            "total_rental_cost": total_rental_cost,
                            "rental_share": 0.0,
                            "reason": "missing_price_data",
                            "hint": "Check price provider connection",
                        },
                    )
                )
                return 0.0

            # Calculate epoch subnet emission and store for estimation use
            epoch_subnet_emission = TEMPO * tao_price * alpha_rate
            self.epoch_subnet_emission = epoch_subnet_emission

        # Calculate rental cost per epoch
        rental_cost_per_epoch = total_rental_cost * (TEMPO * SECONDS_PER_BLOCK) / 3600

        # Calculate rental share (before capping)
        rental_share_raw = rental_cost_per_epoch / FIXED_RATIO / epoch_subnet_emission if epoch_subnet_emission > 0 else 0.0

        logger.info(
            _m(
                "Phase 2: Calculated rental share formula breakdown",
                extra={
                    "total_rental_cost_per_hour": total_rental_cost,
                    "tempo": TEMPO,
                    "seconds_per_block": SECONDS_PER_BLOCK,
                    "fixed_ratio": FIXED_RATIO,
                    "epoch_subnet_emission": epoch_subnet_emission,
                    "rental_cost_per_epoch": rental_cost_per_epoch,
                    "rental_share_raw": rental_share_raw,
                    "formula": f"({total_rental_cost} * ({TEMPO} * {SECONDS_PER_BLOCK}) / 3600) / {FIXED_RATIO} / {epoch_subnet_emission}",
                },
            )
        )

        return rental_share_raw

    def get_snapshot(self) -> RentalPriceSnapshot:
        """Return a snapshot of the current epoch incentive state."""
        total_gpu_model_count_map: dict[str, int] = {}
        for results in self.job_results.values():
            for result in results:
                if not result.is_successful or not result.gpu_model:
                    continue
                # Keep snapshot totals aligned with live scoring denominators:
                # executors excluded from incentive pools should not inflate estimates.
                if result.is_spot or is_missing_discord_after_cutoff(result):
                    continue
                total_gpu_model_count_map[result.gpu_model] = (
                    total_gpu_model_count_map.get(result.gpu_model, 0) + result.gpu_count
                )

        total_gpu_count = sum(total_gpu_model_count_map.values())

        by_bucket: dict[str, GpuBucketRentalState] = {}
        for (base_model, bucket), unrented_count in self.unrented_count_by_bucket.items():
            max_cap = self.config.max_unrented_gpus.get(base_model, {}).get(bucket, 0)
            cap_multiplier = self.cap_multiplier_by_bucket.get((base_model, bucket), 0.0)
            weighted_rate_sum = self._weighted_rate_sum_by_bucket.get((base_model, bucket), 0.0)
            by_bucket[self._bucket_key_str(base_model, bucket)] = GpuBucketRentalState(
                unrented_count=unrented_count,
                max_cap=max_cap,
                cap_multiplier=cap_multiplier,
                weighted_rate_sum=weighted_rate_sum,
            )

        return RentalPriceSnapshot(
            epoch_subnet_emission=self.epoch_subnet_emission,
            rental_share=self.rental_share,
            burn_share=self.burn_share,
            mining=RentalMiningState(
                total_gpu_count=total_gpu_count,
                total_mining_score=self.total_mining_score,
                total_gpu_model_count_map=total_gpu_model_count_map,
            ),
            rental=RentalShareState(
                total_rental_cost=self.total_rental_cost,
                by_bucket=by_bucket,
            ),
        )


# ── Module-level standalone estimation functions ──────────────────────────────

async def estimate_executor(
    config: IncentiveConfig,
    redis_service: RedisService,
    snapshot: RentalPriceSnapshot,
    params: ExecutorEstimateParams,
) -> RentalPriceEstimate:
    """Estimate USD/epoch reward for a single hypothetical executor against a snapshot."""
    estimator = RentalPriceIncentive(
        config,
        redis_service,
        jobs_results={},
        total_gpu_model_count_map=snapshot.mining.total_gpu_model_count_map or {},
        snapshot=snapshot,
    )
    return await estimator.estimate_executor(params=params)


async def precompute_all_estimates(
    config: IncentiveConfig,
    snapshot: RentalPriceSnapshot,
    redis_service: RedisService,
) -> dict[str, dict]:
    """Precompute rented and unrented estimates for every GPU model in BASE_GPU_MAP.

    Returns a dict keyed by full GPU model name with "rented", "unrented" (gpu_count=1)
    and "unrented_8x" (gpu_count=8) estimates. The 8x variant exposes per-bucket capacity
    (max_cap, total_unrented_by_gpu_type, hourly_rate) for the 8-GPU rig bucket so that
    downstream consumers can render network-wide bucket fill state without an extra
    on-demand request to the validator.
    """
    gpu_models = list(BASE_GPU_MAP.keys())
    estimates: dict[str, dict] = {}
    for gpu_model in gpu_models:
        estimates[gpu_model] = {
            "unrented": await estimate_executor(
                config,
                redis_service,
                snapshot,
                ExecutorEstimateParams(gpu_model=gpu_model, is_rented=False),
            ),
            "rented": await estimate_executor(
                config,
                redis_service,
                snapshot,
                ExecutorEstimateParams(gpu_model=gpu_model, is_rented=True),
            ),
            "unrented_8x": await estimate_executor(
                config,
                redis_service,
                snapshot,
                ExecutorEstimateParams(gpu_model=gpu_model, gpu_count=8, is_rented=False),
            ),
        }

    return estimates
