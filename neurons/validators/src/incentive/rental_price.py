"""Rental price incentive algorithm implementation.

This module implements the three-phase rental price incentive algorithm that
rewards unrented high-end GPUs based on their rental market value.

The system uses per-GPU-type caps to dilute incentives when supply exceeds
demand for specific GPU models. Each GPU type has an independent cap value
configured in MAX_UNRENTED_GPUS_BY_TYPE.
"""

import bittensor

from core.utils import _m, get_extra_info, get_logger
from incentive.burn_service import BurnService
from incentive.config import IncentiveConfig
from incentive.default import DefaultIncentive
from incentive.price_provider import PriceProvider
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.const import TEMPO, SECONDS_PER_BLOCK, FIXED_RATIO, TOTAL_BURN_EMISSION
from services.redis_service import RedisService
from services.task_service import JobResult

logger = get_logger(__name__)


class RentalPriceIncentive(DefaultIncentive):
    """Rental price incentive algorithm.

    Implements a three-phase algorithm:
    - Phase 1: Exclude unrented eligible GPUs from mining scores
    - Phase 2: Calculate dynamic emission splits based on rental costs
    - Phase 3: Distribute weights across burn/mining/rental pools

    Cap dilution is applied per GPU type based on max_unrented_gpus dictionary.
    Each GPU type has an independent cap, allowing different supply/demand dynamics.
    """

    def __init__(self, *args, **kwargs):
        """Initialize rental price incentive algorithm.

        Args:
            config: Incentive configuration with rental_incentive_gpu_types,
                   max_unrented_gpus (dict per GPU type), and rental_prices_per_hour
            redis_service: Redis service for accessing shared state
            burn_service: Burn emission distribution service
        """
        super().__init__(*args, **kwargs)
        self.price_provider = PriceProvider()

        self.unrented_count_by_type = {}
        self.total_rental_cost = 0.0
        self.rental_share = 0.0
        self.burn_share = 0.0

    async def _pre_process_job_result(self, hotkey: str, result: JobResult):
        """Process a job result.
        Aggregate metrics from job result.

        Note: max_unrented_gpus is now a dictionary per GPU type.
        """
        await super()._pre_process_job_result(hotkey, result)

        if result.score == 0 and result.job_score == 0:
            return

        # Check if GPU is eligible
        if result.gpu_model not in self.config.rental_incentive_gpu_types:
            return

        #  calculate unrented gpu count that's eligible for rental price incentive
        if result.eligible_for_rental_share:
            # update result state
            result.hourly_rate = self.config.rental_prices_per_hour.get(result.gpu_model, 0)
            result.max_cap = self.config.max_unrented_gpus.get(result.gpu_model, 0)

            gpu_type = result.gpu_model
            gpu_count_for_rental_share = min(
                result.max_cap - min(self.unrented_count_by_type.get(gpu_type, 0), result.max_cap),
                result.gpu_count
            )

            # aggregated unrented count by type
            self.unrented_count_by_type[gpu_type] = self.unrented_count_by_type.get(gpu_type, 0) + result.gpu_count

            # aggregate for total rental cost
            self.total_rental_cost += gpu_count_for_rental_share * result.hourly_rate

    async def _on_finish_pre_process(self) -> None:
        """Callback after pre-processing all job results.

        - Calculate rental share 
        """
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
        result.total_unrented_by_gpu_type = self.unrented_count_by_type.get(result.gpu_model, 0)
        # formula - executor's reward will be diluted by the total unrented count for the gpu type
        result.effective_rate = (
            result.hourly_rate * min(result.max_cap, result.total_unrented_by_gpu_type) / result.total_unrented_by_gpu_type
        ) if result.total_unrented_by_gpu_type > 0 else 0
        result.cap_dilution_applied = result.total_unrented_by_gpu_type > result.max_cap
        result.rental_share = self.rental_share # calculated in _on_finish_pre_process
        result.burn_share = self.burn_share # calculated in _on_finish_pre_process
        result.total_rental_cost = self.total_rental_cost # calculated in _on_finish_pre_process

        # calculate incentive score
        result.incentive = (
            result.rental_share * result.gpu_count * result.effective_rate / result.total_rental_cost 
            if result.total_rental_cost > 0 else 0.0
        )

        # update incentive logs
        result.incentive_logs.append(
            _m(
                "Rental price incentive for executor is calculated successfully. Formula: rental_share * gpu_count / total_unrented_count",
                extra=get_extra_info({
                    "hotkey": hotkey,
                    "executor_id": str(result.executor_info.uuid),
                    "gpu_model": result.gpu_model,
                    "gpu_count": result.gpu_count,
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

        Args:
            total_gpu_model_count_map: Mapping of GPU models to total counts
            job_result: Job execution result to score

        Returns:
            Calculated score (0 for unrented eligible GPUs, normal score otherwise)
        """
        # Check if GPU is unrented and eligible (has defined cap in max_unrented_gpus)
        job_result.eligible_for_rental_share = (
            not job_result.is_rented
            and (job_result.gpu_model in self.config.rental_incentive_gpu_types)
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

        # Fetch TAO price and alpha rate
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

        # Calculate epoch subnet emission
        epoch_subnet_emission = TEMPO * tao_price * alpha_rate

        # Calculate rental cost per epoch
        rental_cost_per_epoch = total_rental_cost * (TEMPO * SECONDS_PER_BLOCK) / 3600

        # Calculate rental share (before capping)
        rental_share_raw = rental_cost_per_epoch / FIXED_RATIO / epoch_subnet_emission if epoch_subnet_emission > 0 else 0.0

        logger.info(
            _m(
                "Phase 2: Calculated rental share formula breakdown",
                extra={
                    "total_rental_cost_per_hour": total_rental_cost,
                    "tao_price": tao_price,
                    "alpha_rate": alpha_rate,
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
