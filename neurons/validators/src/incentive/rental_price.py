"""Rental price incentive algorithm implementation.

This module implements the three-phase rental price incentive algorithm that
rewards unrented high-end GPUs based on their rental market value.

The system uses per-GPU-type caps to dilute incentives when supply exceeds
demand for specific GPU models. Each GPU type has an independent cap value
configured in MAX_UNRENTED_GPUS_BY_TYPE.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import bittensor
from pydantic import BaseModel, Field

from core.utils import _m, get_extra_info, get_logger
from incentive.config import BASE_GPU_MAP

if TYPE_CHECKING:
    from incentive.config import IncentiveConfig
    from services.redis_service import RedisService
from incentive.utils import get_hourly_rate
from incentive.default import DefaultIncentive
from incentive.price_provider import PriceProvider
from services.const import TEMPO, SECONDS_PER_BLOCK, FIXED_RATIO, TOTAL_BURN_EMISSION
from services.task_service import JobResult

logger = get_logger(__name__)


# ── Snapshot models ──────────────────────────────────────────────────────────

class GpuTypeRentalState(BaseModel):
    unrented_count: int
    max_cap: int
    cap_multiplier: float
    weighted_rate_sum: float  # sum(gpu_count * hourly_rate) for this type, cap NOT applied


class RentalMiningState(BaseModel):
    total_gpu_count: int
    total_mining_score: float
    # Per full GPU model totals used by DefaultIncentive.calculate_executor_score.
    # This must be per `JobResult.gpu_model` (not base model), so the default mining
    # score formula can normalize consistently for both real and estimated jobs.
    total_gpu_model_count_map: dict[str, int] = Field(default_factory=dict)


class RentalShareState(BaseModel):
    total_rental_cost: float
    by_gpu_type: dict[str, GpuTypeRentalState]


class RentalPriceSnapshot(BaseModel):
    epoch_subnet_emission: float
    rental_share: float
    burn_share: float
    mining: RentalMiningState
    rental: RentalShareState


# ── Estimate model ────────────────────────────────────────────────────────────

class RentalPriceEstimate(BaseModel):
    gpu_model: str
    base_model: str
    gpu_count: int
    is_rented: bool
    tao_per_epoch: float
    rental_share: float | None = None
    effective_rate: float | None = None
    cap_multiplier: float | None = None
    eligible_for_rental_incentive: bool = True
    mining_share: float | None = None


class RentalPriceIncentive(DefaultIncentive):
    """Rental price incentive algorithm.

    Implements a three-phase algorithm:
    - Phase 1: Exclude unrented eligible GPUs from mining scores
    - Phase 2: Calculate dynamic emission splits based on rental costs
    - Phase 3: Distribute weights across burn/mining/rental pools

    Cap dilution is applied per GPU type based on max_unrented_gpus dictionary.
    Each GPU type has an independent cap, allowing different supply/demand dynamics.
    """

    price_provider: PriceProvider = PriceProvider()

    def __init__(self, *args, snapshot: "RentalPriceSnapshot | None" = None, **kwargs):
        """Initialize rental price incentive algorithm.

        Args:
            config: Incentive configuration with rental_incentive_gpu_types,
                   max_unrented_gpus (dict per GPU type), and rental_prices_per_hour
            redis_service: Redis service for accessing shared state
            burn_service: Burn emission distribution service
            snapshot: Optional snapshot to seed accumulated state (for estimation)
        """
        super().__init__(*args, **kwargs)

        self.unrented_count_by_type: dict[str, int] = {}  # {base_model: raw_gpu_count}
        self.cap_multiplier_by_base_model: dict[str, float] = {}  # cap dilution multiplier per base model
        self.total_rental_cost = 0.0
        self.rental_share = 0.0
        self.burn_share = 0.0
        self._weighted_rate_sum_by_type: dict[str, float] = {}  # {base_model: sum(gpu_count * rate)}
        self.epoch_subnet_emission: float = 0.0
        # Store the snapshot so estimation can derive per-model totals from it.
        self._seed_snapshot = snapshot

        # validate configs
        for base_model in self.config.rental_incentive_gpu_types:
            assert base_model in BASE_GPU_MAP.values(), f"Base model {base_model} not found in BASE_GPU_MAP"

        for gpu_type in self.config.rental_prices_per_hour.keys():
            assert gpu_type in BASE_GPU_MAP.keys(), f"GPU type {gpu_type} not found in BASE_GPU_MAP"

        if snapshot:
            for base_model, state in snapshot.rental.by_gpu_type.items():
                self.unrented_count_by_type[base_model] = state.unrented_count
                self._weighted_rate_sum_by_type[base_model] = state.weighted_rate_sum
            self.total_mining_score = snapshot.mining.total_mining_score
            self.epoch_subnet_emission = snapshot.epoch_subnet_emission

    def get_base_model_for_gpu(self, gpu_model: str) -> str:
        base_model = BASE_GPU_MAP[gpu_model]
        return base_model

    async def _pre_process_job_result(self, hotkey: str, result: JobResult):
        """Process a job result.
        Aggregate metrics from job result.

        Note: max_unrented_gpus is now a dictionary per GPU type.
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
            result.max_cap = self.config.max_unrented_gpus.get(base_model, 0)

            # accumulate raw unrented GPU count and weighted rate sum per base model (only if rate > 0)
            if result.hourly_rate > 0:
                self.unrented_count_by_type[base_model] = (
                    self.unrented_count_by_type.get(base_model, 0) + result.gpu_count
                )
                self._weighted_rate_sum_by_type[base_model] = (
                    self._weighted_rate_sum_by_type.get(base_model, 0) + result.gpu_count * result.hourly_rate
                )

    async def _on_finish_pre_process(self) -> None:
        """Callback after pre-processing all job results.

        - Calculate rental share
        """
        # Step 1: cap multiplier from raw counts
        for base_model, unrented_count in self.unrented_count_by_type.items():
            max_cap = self.config.max_unrented_gpus.get(base_model, 0)
            if unrented_count > 0:
                self.cap_multiplier_by_base_model[base_model] = min(unrented_count, max_cap) / unrented_count

        # Step 2: total_rental_cost from accumulated weighted rate sums
        for base_model, weighted_sum in self._weighted_rate_sum_by_type.items():
            cap_mult = self.cap_multiplier_by_base_model.get(base_model, 0)
            self.total_rental_cost += cap_mult * weighted_sum

        rental_share_raw = await self._calculate_rental_share(self.total_rental_cost)

        # Ensure rental_share doesn't exceed 0.91 (cap at burn emission)
        rental_share_capped = rental_share_raw > TOTAL_BURN_EMISSION
        self.rental_share = min(rental_share_raw, TOTAL_BURN_EMISSION)

        if rental_share_capped:
            logger.warning(
                _m(
                    "Rental share capped at max burn emission",
                    extra={
                        "rental_share_raw": rental_share_raw,
                        "rental_share_capped": self.rental_share,
                        "max_cap": TOTAL_BURN_EMISSION,
                        "hint": f"Rental share would have been {rental_share_raw:.4f} but capped at {TOTAL_BURN_EMISSION}",
                    },
                )
            )

        # Calculate emission splits
        self.burn_share = TOTAL_BURN_EMISSION - self.rental_share
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
        result.total_unrented_by_gpu_type = self.unrented_count_by_type.get(base_model, 0)
        result.cap_dilution_applied = result.total_unrented_by_gpu_type > result.max_cap
        result.rental_share = self.rental_share
        result.burn_share = self.burn_share
        result.total_rental_cost = self.total_rental_cost
        result.unrented_cap_multiplier = self.cap_multiplier_by_base_model.get(base_model, 0)
        result.effective_rate = result.hourly_rate * result.unrented_cap_multiplier

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
                    "unrented_cap_multiplier": result.unrented_cap_multiplier,
                    "effective_rate": result.effective_rate,
                    "total_unrented_by_gpu_type": result.total_unrented_by_gpu_type,
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

        Eligibility is determined by whether the GPU type has a defined cap
        in max_unrented_gpus (per-GPU-type caps).
 _g
        Args:
            total_gpu_model_count_map: Mapping of GPU models to total counts
            job_result: Job execution result to score

        Returns:
            Calculated score (0 for unrented eligible GPUs, normal score otherwise)
        """
        # Check if GPU is unrented and eligible (has defined cap in max_unrented_gpus)
        base_model = self.get_base_model_for_gpu(job_result.gpu_model)
        job_result.eligible_for_rental_share = (
            not job_result.is_rented
            and (base_model in self.config.rental_incentive_gpu_types)
            and (job_result.score > 0 or job_result.job_score > 0)
        )
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
        gpu_model: str,
        gpu_count: int = 1,
        is_rented: bool = False,
        gpu_splitting: bool = False,
        gpu_splitting_min_count: int | None = None,
    ) -> RentalPriceEstimate:
        """Estimate TAO/epoch reward for a hypothetical executor from this instance's snapshot.

        This uses the same internal 3-stage pipeline as real scoring:
        `_pre_process_job_result` -> `_on_finish_pre_process` -> `_post_process_job_result`.
        """
        # Snapshot is required so we can:
        # 1) seed rental-phase state (unrented counts, weighted rate sums, etc.)
        # 2) get per-model GPU totals for DefaultIncentive's normalization.
        if self._seed_snapshot is None:
            raise ValueError("estimate_executor requires RentalPriceIncentive initialized with snapshot=")

        from datura.requests.miner_requests import ExecutorSSHInfo

        base_model = BASE_GPU_MAP.get(gpu_model)
        eligible_for_unrented_estimate = (
            base_model is not None
            and self.config.max_unrented_gpus.get(base_model, 0) > 0
        )
        if base_model is None or (not is_rented and not eligible_for_unrented_estimate):
            return RentalPriceEstimate(
                gpu_model=gpu_model,
                base_model=base_model or gpu_model,
                gpu_count=gpu_count,
                is_rented=is_rented,
                tao_per_epoch=0.0,
                eligible_for_rental_incentive=False,
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
            # Prevent uptime redis lookups for the fake executor. Pipeline still
            # needs get_portion_per_gpu_type() for mining_score normalization.
            collateral_deposited=True,
        )

        await self._pre_process_job_result("estimate", fake_result)
        await self._on_finish_pre_process()
        await self._post_process_job_result("estimate", fake_result)

        return RentalPriceEstimate(
            gpu_model=gpu_model,
            base_model=base_model,
            gpu_count=fake_result.gpu_count,
            is_rented=fake_result.is_rented,
            tao_per_epoch=(fake_result.incentive or 0.0) * self.epoch_subnet_emission,
            rental_share=fake_result.rental_share if not is_rented else None,
            effective_rate=fake_result.effective_rate if not is_rented else None,
            cap_multiplier=fake_result.unrented_cap_multiplier if not is_rented else None,
            eligible_for_rental_incentive=(
                bool(fake_result.eligible_for_rental_share) if not is_rented else True
            ),
            mining_share=self.mining_share if is_rented else None,
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
            Rental emission share (0 to 0.91)
        """
        if total_rental_cost == 0:
            logger.info(
                _m(
                    "Phase 2: No rental cost - rental share is 0",
                    extra={
                        "total_rental_cost": total_rental_cost,
                        "rental_share": 0.0,
                        "reason": "no_unrented_eligible_gpus",
                    },
                )
            )
            return 0.0

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
                total_gpu_model_count_map[result.gpu_model] = (
                    total_gpu_model_count_map.get(result.gpu_model, 0) + result.gpu_count
                )

        total_gpu_count = sum(total_gpu_model_count_map.values())

        by_gpu_type: dict[str, GpuTypeRentalState] = {}
        for base_model, unrented_count in self.unrented_count_by_type.items():
            max_cap = self.config.max_unrented_gpus.get(base_model, 0)
            cap_multiplier = self.cap_multiplier_by_base_model.get(base_model, 0.0)
            weighted_rate_sum = self._weighted_rate_sum_by_type.get(base_model, 0.0)
            by_gpu_type[base_model] = GpuTypeRentalState(
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
                by_gpu_type=by_gpu_type,
            ),
        )


# ── Module-level standalone estimation functions ──────────────────────────────

async def estimate_executor(
    config: IncentiveConfig,
    redis_service: RedisService,
    snapshot: RentalPriceSnapshot,
    gpu_model: str,
    gpu_count: int = 1,
    is_rented: bool = False,
    gpu_splitting: bool = False,
    gpu_splitting_min_count: int | None = None,
) -> RentalPriceEstimate:
    """Estimate TAO/epoch reward for a single hypothetical executor against a snapshot."""
    estimator = RentalPriceIncentive(
        config,
        redis_service,
        jobs_results={},
        total_gpu_model_count_map=snapshot.mining.total_gpu_model_count_map or {},
        snapshot=snapshot,
    )
    return await estimator.estimate_executor(
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        is_rented=is_rented,
        gpu_splitting=gpu_splitting,
        gpu_splitting_min_count=gpu_splitting_min_count,
    )


async def precompute_all_estimates(
    config: IncentiveConfig,
    snapshot: RentalPriceSnapshot,
    redis_service: RedisService,
) -> dict[str, dict]:
    """Precompute rented and unrented estimates for every GPU model in BASE_GPU_MAP.

    Returns a dict keyed by full GPU model name with "rented" and "unrented" estimates.
    """
    gpu_models = list(BASE_GPU_MAP.keys())
    coros = [
        estimate_executor(config, redis_service, snapshot, gpu_model=m, gpu_count=1, is_rented=False)
        for m in gpu_models
    ] + [
        estimate_executor(config, redis_service, snapshot, gpu_model=m, gpu_count=1, is_rented=True)
        for m in gpu_models
    ]
    estimates = await asyncio.gather(*coros)
    n = len(gpu_models)
    return {
        m: {"unrented": estimates[i], "rented": estimates[n + i]}
        for i, m in enumerate(gpu_models)
    }
