"""Rental price incentive algorithm implementation.

This module implements the three-phase rental price incentive algorithm that
rewards unrented high-end GPUs based on their rental market value.
"""

import bittensor

from core.config import settings
from core.utils import _m, get_extra_info, get_logger
from incentive.base import BaseIncentive
from incentive.config import IncentiveConfig
from incentive.price_provider import PriceProvider
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.const import TEMPO, SECONDS_PER_BLOCK, FIXED_RATIO, TOTAL_BURN_EMISSION
from services.redis_service import RedisService
from services.task_service import JobResult

logger = get_logger(__name__)


class RentalPriceIncentive(BaseIncentive):
    """Rental price incentive algorithm.

    Implements a three-phase algorithm:
    - Phase 1: Exclude unrented eligible GPUs from mining scores
    - Phase 2: Calculate dynamic emission splits based on rental costs
    - Phase 3: Distribute weights across burn/mining/rental pools
    """

    def __init__(self, config: IncentiveConfig, redis_service: RedisService):
        """Initialize rental price incentive algorithm.

        Args:
            config: Incentive configuration with eligible_gpu_types,
                   max_unrented_gpus, and rental_prices_per_hour
            redis_service: Redis service for accessing shared state
        """
        super().__init__(config, redis_service)
        self.price_provider = PriceProvider()

    async def calculate_executor_score(
        self,
        total_gpu_model_count_map: dict,
        job_result: JobResult,
    ) -> float:
        """Calculate score for a single executor/job result.

        Phase 1: Unrented eligible GPUs are excluded from mining emission
        by returning score = 0. All other GPUs use normal scoring logic.

        Args:
            total_gpu_model_count_map: Mapping of GPU models to total counts
            job_result: Job execution result to score

        Returns:
            Calculated score (0 for unrented eligible GPUs, normal score otherwise)
        """
        # Check if GPU is unrented and eligible
        is_rented = job_result.is_rented
        is_eligible = job_result.gpu_model in self.config.eligible_gpu_types

        if not is_rented and is_eligible:
            logger.debug(
                _m(
                    "Excluding unrented eligible GPU from mining emission",
                    extra={
                        **get_extra_info(job_result),
                        "gpu_model": job_result.gpu_model,
                        "eligible_gpu_types": self.config.eligible_gpu_types,
                    },
                )
            )
            return 0  # Exclude from mining pool

        # For rented or non-eligible GPUs, use normal scoring logic
        return await self._calculate_default_score(total_gpu_model_count_map, job_result)

    async def _calculate_default_score(
        self,
        total_gpu_model_count_map: dict,
        job_result: JobResult,
    ) -> float:
        """Calculate score using default logic (same as DefaultIncentive).

        This reuses the scoring logic from DefaultIncentive for rented
        and non-eligible GPUs.

        Args:
            total_gpu_model_count_map: Mapping of GPU models to total counts
            job_result: Job execution result to score

        Returns:
            Calculated score using default algorithm
        """
        # Early exit check
        if job_result.score == 0:
            return 0

        # GPU count calculation
        total_gpu_count = total_gpu_model_count_map.get(job_result.gpu_model, 0)
        if total_gpu_count == 0:
            return 0

        # Base score calculation
        score_portion = await self.redis_service.get_portion_per_gpu_type(job_result.gpu_model)
        score = job_result.score * score_portion * job_result.gpu_count / total_gpu_count

        # Multiplier calculation
        multiplier = 1

        # Sysbox runtime multiplier
        if not job_result.sysbox_runtime:
            multiplier *= 1 - settings.PORTION_FOR_SYSBOX

        # Uptime multiplier
        if not job_result.collateral_deposited:
            uptime_in_minutes = await self.redis_service.get_executor_uptime(job_result.executor_info)
            multiplier *= (
                1
                - settings.PORTION_FOR_UPTIME
                + settings.PORTION_FOR_UPTIME
                * min(1, uptime_in_minutes / settings.UPTIME_REQUIRED_MINUTES)
            )

        return score * multiplier

    async def calculate_final_weights(
        self,
        miner_scores: dict[str, float],
        miners: list[bittensor.NeuronInfo],
        last_mechanism_step_block: int | None,
        all_job_results: dict[str, list[JobResult]],
        rented_data: RentedExecutorsResponse,
    ) -> dict[str, float]:
        """Calculate scores with rental price incentive for this cycle.

        This method applies rental price incentive and burning logic, returning
        scores to be accumulated. It does NOT return final weights for blockchain.

        Phase 2: Calculate dynamic emission splits
        - Count unrented GPUs by type
        - Apply cap dilution if needed
        - Calculate rental_share (X) using rental costs and TAO price

        Phase 3: Distribute scores across pools
        - Burn pool: 0.91 - X
        - Mining pool: 0.09 (fixed)
        - Rental pool: X

        Args:
            miner_scores: Mining scores from this cycle only (not accumulated)
            miners: List of miner neuron information
            last_mechanism_step_block: Last mechanism step block number
            all_job_results: All job results by miner hotkey
            rented_data: Rented executors data from backend

        Returns:
            dict[str, float]: Scores with burning and rental incentive applied
        """
        # Phase 2: Count unrented GPUs and calculate rental costs
        unrented_count_by_type = self._count_unrented_gpus(all_job_results, rented_data)
        total_rental_cost = self._calculate_total_rental_cost(unrented_count_by_type)

        # Calculate rental emission share
        rental_share = await self._calculate_rental_share(total_rental_cost)

        # Ensure rental_share doesn't exceed 0.91 (cap at burn emission)
        rental_share = min(rental_share, TOTAL_BURN_EMISSION)

        # Calculate emission splits
        burn_share = TOTAL_BURN_EMISSION - rental_share
        mining_share = 1 - TOTAL_BURN_EMISSION  # 0.09

        logger.info(
            _m(
                "Calculated emission splits",
                extra={
                    "rental_share": rental_share,
                    "burn_share": burn_share,
                    "mining_share": mining_share,
                    "total_rental_cost": total_rental_cost,
                    "unrented_gpus": unrented_count_by_type,
                },
            )
        )

        # Phase 3: Calculate per-miner rental values
        miner_rental_values = self._calculate_miner_rental_values(
            all_job_results, rented_data, unrented_count_by_type
        )

        # Distribute scores across pools
        return self._distribute_scores(
            miners=miners,
            miner_scores=miner_scores,
            miner_rental_values=miner_rental_values,
            burn_share=burn_share,
            mining_share=mining_share,
            rental_share=rental_share,
            last_mechanism_step_block=last_mechanism_step_block,
        )

    def _count_unrented_gpus(
        self,
        all_job_results: dict[str, list[JobResult]],
        rented_data: RentedExecutorsResponse,
    ) -> dict[str, int]:
        """Count unrented GPUs by type.

        Args:
            all_job_results: All job results by miner hotkey
            rented_data: Rented executors data

        Returns:
            Dictionary mapping GPU type to count of unrented GPUs
        """
        unrented_count = {}

        for miner_hotkey, results in all_job_results.items():
            for result in results:
                # Check if GPU is eligible
                if result.gpu_model not in self.config.eligible_gpu_types:
                    continue

                # Check if executor is rented
                executor_id = result.executor_info.uuid
                rented_executor = rented_data.executors.get(executor_id) if rented_data else None
                is_rented = rented_executor is not None and len(rented_executor.pods) > 0

                if not is_rented:
                    gpu_type = result.gpu_model
                    unrented_count[gpu_type] = unrented_count.get(gpu_type, 0) + result.gpu_count

        return unrented_count

    def _calculate_total_rental_cost(
        self,
        unrented_count_by_type: dict[str, int],
    ) -> float:
        """Calculate total rental cost with cap dilution.

        Args:
            unrented_count_by_type: Count of unrented GPUs by type

        Returns:
            Total rental cost per hour in USD
        """
        total_cost = 0.0

        for gpu_type, count in unrented_count_by_type.items():
            # Get hourly rate from config
            hourly_rate = self.config.rental_prices_per_hour.get(gpu_type, 0)
            if hourly_rate == 0:
                logger.warning(
                    _m(
                        "No rental price configured for GPU type",
                        extra={"gpu_type": gpu_type},
                    )
                )
                continue

            # Apply cap dilution
            max_cap = self.config.max_unrented_gpus
            if count > max_cap:
                # All GPUs get diluted reward
                effective_rate = hourly_rate * max_cap / count
                logger.info(
                    _m(
                        "Applying cap dilution",
                        extra={
                            "gpu_type": gpu_type,
                            "count": count,
                            "max_cap": max_cap,
                            "hourly_rate": hourly_rate,
                            "effective_rate": effective_rate,
                        },
                    )
                )
            else:
                effective_rate = hourly_rate

            total_cost += count * effective_rate

        return total_cost

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
            return 0.0

        # Fetch TAO price and alpha rate
        tao_price = await self.price_provider.get_tao_price()
        alpha_rate = await self.price_provider.get_alpha_rate()

        if tao_price is None or alpha_rate is None:
            logger.warning(
                _m(
                    "Failed to fetch TAO price or alpha rate, falling back to 0 rental share",
                    extra={
                        "tao_price": tao_price,
                        "alpha_rate": alpha_rate,
                    },
                )
            )
            return 0.0

        # Calculate epoch subnet emission
        epoch_subnet_emission = TEMPO * tao_price * alpha_rate

        # Calculate rental share
        rental_share = (
            total_rental_cost
            * (TEMPO * SECONDS_PER_BLOCK)
            / 3600
            / FIXED_RATIO
            / epoch_subnet_emission
        )

        logger.info(
            _m(
                "Calculated rental share",
                extra={
                    "total_rental_cost": total_rental_cost,
                    "tao_price": tao_price,
                    "alpha_rate": alpha_rate,
                    "epoch_subnet_emission": epoch_subnet_emission,
                    "rental_share": rental_share,
                },
            )
        )

        return rental_share

    def _calculate_miner_rental_values(
        self,
        all_job_results: dict[str, list[JobResult]],
        rented_data: RentedExecutorsResponse,
        unrented_count_by_type: dict[str, int],
    ) -> dict[str, float]:
        """Calculate rental value for each miner.

        Args:
            all_job_results: All job results by miner hotkey
            rented_data: Rented executors data
            unrented_count_by_type: Total count of unrented GPUs by type

        Returns:
            Dictionary mapping miner hotkey to rental value in USD
        """
        miner_rental_values = {}

        for miner_hotkey, results in all_job_results.items():
            miner_rental_value = 0.0

            for result in results:
                # Check if GPU is eligible
                if result.gpu_model not in self.config.eligible_gpu_types:
                    continue

                # Check if executor is rented
                executor_id = result.executor_info.uuid
                rented_executor = rented_data.executors.get(executor_id) if rented_data else None
                is_rented = rented_executor is not None and len(rented_executor.pods) > 0

                if not is_rented:
                    gpu_type = result.gpu_model
                    hourly_rate = self.config.rental_prices_per_hour.get(gpu_type, 0)
                    if hourly_rate == 0:
                        continue

                    # Apply same dilution as in total cost calculation
                    count = unrented_count_by_type.get(gpu_type, 0)
                    max_cap = self.config.max_unrented_gpus
                    if count > max_cap:
                        effective_rate = hourly_rate * max_cap / count
                    else:
                        effective_rate = hourly_rate

                    miner_rental_value += result.gpu_count * effective_rate

            if miner_rental_value > 0:
                miner_rental_values[miner_hotkey] = miner_rental_value

        return miner_rental_values

    def _distribute_scores(
        self,
        miners: list[bittensor.NeuronInfo],
        miner_scores: dict[str, float],
        miner_rental_values: dict[str, float],
        burn_share: float,
        mining_share: float,
        rental_share: float,
        last_mechanism_step_block: int,
    ) -> dict[str, float]:
        """Distribute scores across burn, mining, and rental pools.

        Args:
            miners: List of miner neuron information
            miner_scores: Mining scores by miner hotkey
            miner_rental_values: Rental values by miner hotkey
            burn_share: Burn emission share
            mining_share: Mining emission share
            rental_share: Rental emission share
            last_mechanism_step_block: Last mechanism step block number

        Returns:
            dict[str, float]: Scores with burning and rental incentive applied
        """
        cycle_scores = {}
        total_mining_score = sum(miner_scores.values())
        total_rental_value = sum(miner_rental_values.values())

        for miner in miners:
            hotkey = miner.hotkey

            # Burners get burn_share
            if settings.ENABLE_NEW_BURN_LOGIC:
                if miner.uid in settings.NEW_BURNERS:
                    cycle_scores[hotkey] = burn_share / len(settings.NEW_BURNERS)
                    continue
            else:
                # Old burn logic
                import random
                main_burner = random.Random(last_mechanism_step_block).choice(settings.BURNERS)
                other_burners = [uid for uid in settings.BURNERS if uid != main_burner]

                if miner.uid == main_burner:
                    cycle_scores[hotkey] = burn_share - (len(settings.BURNERS) - 1) * (burn_share / len(settings.BURNERS))
                    continue
                elif miner.uid in other_burners:
                    cycle_scores[hotkey] = burn_share / len(settings.BURNERS)
                    continue

            # Regular miners get mining + rental scores
            score = 0.0

            # Mining emission (for rented/non-eligible GPUs)
            if hotkey in miner_scores and total_mining_score > 0:
                score += mining_share * miner_scores[hotkey] / total_mining_score

            # Rental emission (for unrented eligible GPUs)
            if hotkey in miner_rental_values and total_rental_value > 0:
                score += rental_share * miner_rental_values[hotkey] / total_rental_value

            cycle_scores[hotkey] = score

        logger.debug(
            _m(
                "Distributed scores across pools",
                extra={
                    "total_mining_score": total_mining_score,
                    "total_rental_value": total_rental_value,
                    "num_miners": len(miners),
                },
            )
        )

        return cycle_scores
