"""Rental price incentive algorithm implementation.

This module implements the three-phase rental price incentive algorithm that
rewards unrented high-end GPUs based on their rental market value.
"""

import bittensor

from core.config import settings
from core.utils import _m, get_logger
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

        logger.debug(
            _m(
                "Phase 1: Evaluating executor for scoring",
                extra={
                    "executor_id": str(job_result.executor_info.uuid),
                    "gpu_model": job_result.gpu_model,
                    "gpu_count": job_result.gpu_count,
                    "is_rented": is_rented,
                    "is_eligible": is_eligible,
                    "eligible_gpu_types": list(self.config.eligible_gpu_types),
                    "pool_assignment": "excluded" if (not is_rented and is_eligible) else "mining",
                },
            )
        )

        if not is_rented and is_eligible:
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
            return 0  # Exclude from mining pool

        # For rented or non-eligible GPUs, use normal scoring logic
        final_score = await self._calculate_default_score(total_gpu_model_count_map, job_result)

        logger.info(
            _m(
                "Executor assigned to mining pool",
                extra={
                    "executor_id": str(job_result.executor_info.uuid),
                    "gpu_model": job_result.gpu_model,
                    "gpu_count": job_result.gpu_count,
                    "is_rented": is_rented,
                    "is_eligible": is_eligible,
                    "score": final_score,
                    "pool": "mining",
                },
            )
        )

        return final_score

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
        if job_result.score == 0 and job_result.job_score == 0:
            logger.debug(
                _m(
                    "Executor score is 0 - early exit",
                    extra={
                        "executor_id": str(job_result.executor_info.uuid),
                        "gpu_model": job_result.gpu_model,
                        "score": job_result.score,
                        "job_score": job_result.job_score,
                        "reason": "job_result_score_zero",
                        "final_score": 0,
                    },
                )
            )
            return 0

        # GPU count calculation
        total_gpu_count = total_gpu_model_count_map.get(job_result.gpu_model, 0)
        if total_gpu_count == 0:
            logger.warning(
                _m(
                    "Total GPU count is 0 for model - cannot calculate score",
                    extra={
                        "executor_id": str(job_result.executor_info.uuid),
                        "gpu_model": job_result.gpu_model,
                        "reason": "total_gpu_count_zero",
                        "final_score": 0,
                    },
                )
            )
            return 0

        # Base score calculation
        score_portion = await self.redis_service.get_portion_per_gpu_type(job_result.gpu_model)
        base_score = job_result.score * score_portion * job_result.gpu_count / total_gpu_count

        # Multiplier calculation
        multiplier = 1
        sysbox_multiplier = 1
        uptime_multiplier = 1
        uptime_minutes = None

        # Sysbox runtime multiplier
        if not job_result.sysbox_runtime:
            sysbox_multiplier = 1 - settings.PORTION_FOR_SYSBOX
            multiplier *= sysbox_multiplier

        # Uptime multiplier
        if not job_result.collateral_deposited:
            uptime_minutes = await self.redis_service.get_executor_uptime(job_result.executor_info)
            uptime_multiplier = (
                1
                - settings.PORTION_FOR_UPTIME
                + settings.PORTION_FOR_UPTIME
                * min(1, uptime_minutes / settings.UPTIME_REQUIRED_MINUTES)
            )
            multiplier *= uptime_multiplier

        final_score = base_score * multiplier

        logger.debug(
            _m(
                "Default score calculation breakdown",
                extra={
                    "executor_id": str(job_result.executor_info.uuid),
                    "gpu_model": job_result.gpu_model,
                    "gpu_count": job_result.gpu_count,
                    "total_gpu_count": total_gpu_count,
                    "job_score": job_result.score,
                    "score_portion": score_portion,
                    "base_score": base_score,
                    "sysbox_runtime": job_result.sysbox_runtime,
                    "sysbox_multiplier": sysbox_multiplier,
                    "collateral_deposited": job_result.collateral_deposited,
                    "uptime_minutes": uptime_minutes,
                    "uptime_multiplier": uptime_multiplier,
                    "total_multiplier": multiplier,
                    "final_score": final_score,
                    "formula": f"{job_result.score} * {score_portion} * {job_result.gpu_count} / {total_gpu_count} * {multiplier}",
                },
            )
        )

        return final_score

    async def calculate_final_weights(
        self,
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
            miners: List of miner neuron information
            last_mechanism_step_block: Last mechanism step block number
            all_job_results: All job results by miner hotkey
            rented_data: Rented executors data from backend

        Returns:
            dict[str, float]: Scores with burning and rental incentive applied
        """
        # Phase 2: Count unrented GPUs and calculate rental costs
        miner_scores = self.temp_miner_scores
        unrented_count_by_type = self._count_unrented_gpus(all_job_results, rented_data)
        total_rental_cost = self._calculate_total_rental_cost(unrented_count_by_type)

        # Calculate rental emission share
        rental_share_raw = await self._calculate_rental_share(total_rental_cost)

        # Ensure rental_share doesn't exceed 0.91 (cap at burn emission)
        rental_share_capped = rental_share_raw > TOTAL_BURN_EMISSION
        rental_share = min(rental_share_raw, TOTAL_BURN_EMISSION)

        if rental_share_capped:
            logger.warning(
                _m(
                    "Rental share capped at max burn emission",
                    extra={
                        "rental_share_raw": rental_share_raw,
                        "rental_share_capped": rental_share,
                        "max_cap": TOTAL_BURN_EMISSION,
                        "hint": f"Rental share would have been {rental_share_raw:.4f} but capped at {TOTAL_BURN_EMISSION}",
                    },
                )
            )

        # Calculate emission splits
        burn_share = TOTAL_BURN_EMISSION - rental_share
        mining_share = 1 - TOTAL_BURN_EMISSION  # 0.09

        logger.info(
            _m(
                "Phase 2: Final emission splits calculated",
                extra={
                    "rental_share": rental_share,
                    "burn_share": burn_share,
                    "mining_share": mining_share,
                    "rental_share_capped": rental_share_capped,
                    "total_rental_cost": total_rental_cost,
                    "unrented_gpus": unrented_count_by_type,
                    "total_unrented": sum(unrented_count_by_type.values()),
                    "validation": f"burn({burn_share:.4f}) + mining({mining_share:.4f}) + rental({rental_share:.4f}) = {burn_share + mining_share + rental_share:.4f}",
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
        eligible_unrented = []
        ineligible_count = 0
        zero_score_count = 0

        for miner_hotkey, results in all_job_results.items():
            for result in results:
                if result.score == 0 and result.job_score == 0:
                    zero_score_count += 1
                    continue
                # Check if GPU is eligible
                if result.gpu_model not in self.config.eligible_gpu_types:
                    ineligible_count += 1
                    continue

                # Check if executor is rented
                executor_id = result.executor_info.uuid
                rented_executor = rented_data.executors.get(executor_id) if rented_data else None
                is_rented = rented_executor is not None and len(rented_executor.pods) > 0

                if not is_rented:
                    gpu_type = result.gpu_model
                    unrented_count[gpu_type] = unrented_count.get(gpu_type, 0) + result.gpu_count
                    eligible_unrented.append({
                        "executor_id": str(executor_id),
                        "miner_hotkey": miner_hotkey,
                        "gpu_model": gpu_type,
                        "gpu_count": result.gpu_count,
                    })

        logger.info(
            _m(
                "Phase 2: Counted unrented eligible GPUs",
                extra={
                    "unrented_count_by_type": unrented_count,
                    "total_unrented_gpus": sum(unrented_count.values()),
                    "num_unrented_executors": len(eligible_unrented),
                    "skipped_ineligible": ineligible_count,
                    "skipped_zero_score": zero_score_count,
                    "eligible_gpu_types": list(self.config.eligible_gpu_types),
                },
            )
        )

        logger.debug(
            _m(
                "Unrented eligible GPU details",
                extra={
                    "unrented_executors": eligible_unrented,
                },
            )
        )

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
        cost_breakdown = []

        for gpu_type, count in unrented_count_by_type.items():
            # Get hourly rate from config
            hourly_rate = self.config.rental_prices_per_hour.get(gpu_type, 0)
            if hourly_rate == 0:
                logger.warning(
                    _m(
                        "No rental price configured for GPU type - skipping",
                        extra={
                            "gpu_type": gpu_type,
                            "count": count,
                            "reason": "missing_rental_price",
                        },
                    )
                )
                continue

            # Apply cap dilution
            max_cap = self.config.max_unrented_gpus
            cap_dilution_applied = count > max_cap
            if cap_dilution_applied:
                # All GPUs get diluted reward
                effective_rate = hourly_rate * max_cap / count
                dilution_factor = max_cap / count
                logger.warning(
                    _m(
                        "Cap dilution applied - unrented count exceeds max cap",
                        extra={
                            "gpu_type": gpu_type,
                            "count": count,
                            "max_cap": max_cap,
                            "hourly_rate": hourly_rate,
                            "effective_rate": effective_rate,
                            "dilution_factor": dilution_factor,
                            "hint": f"Each GPU gets {dilution_factor:.2%} of normal rental rate",
                        },
                    )
                )
            else:
                effective_rate = hourly_rate

            gpu_total_cost = count * effective_rate
            total_cost += gpu_total_cost

            cost_breakdown.append({
                "gpu_type": gpu_type,
                "count": count,
                "hourly_rate": hourly_rate,
                "cap_dilution_applied": cap_dilution_applied,
                "effective_rate": effective_rate,
                "total_cost": gpu_total_cost,
            })

        logger.info(
            _m(
                "Phase 2: Calculated total rental cost",
                extra={
                    "total_rental_cost_per_hour": total_cost,
                    "cost_breakdown": cost_breakdown,
                    "num_gpu_types": len(cost_breakdown),
                },
            )
        )

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
        rental_share_raw = rental_cost_per_epoch / FIXED_RATIO / epoch_subnet_emission

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
        miner_rental_details = []

        for miner_hotkey, results in all_job_results.items():
            miner_rental_value = 0.0
            miner_gpu_details = []

            for result in results:
                if result.score == 0 and result.job_score == 0:
                    continue

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
                    cap_dilution_applied = count > max_cap
                    if cap_dilution_applied:
                        effective_rate = hourly_rate * max_cap / count
                    else:
                        effective_rate = hourly_rate

                    executor_value = result.gpu_count * effective_rate
                    miner_rental_value += executor_value

                    miner_gpu_details.append({
                        "executor_id": str(executor_id),
                        "gpu_model": gpu_type,
                        "gpu_count": result.gpu_count,
                        "hourly_rate": hourly_rate,
                        "effective_rate": effective_rate,
                        "cap_dilution_applied": cap_dilution_applied,
                        "executor_rental_value": executor_value,
                    })

            if miner_rental_value > 0:
                miner_rental_values[miner_hotkey] = miner_rental_value
                miner_rental_details.append({
                    "miner_hotkey": miner_hotkey,
                    "total_rental_value": miner_rental_value,
                    "num_executors": len(miner_gpu_details),
                    "executors": miner_gpu_details,
                })

        logger.info(
            _m(
                "Phase 3: Calculated per-miner rental values",
                extra={
                    "num_miners_with_rental": len(miner_rental_values),
                    "total_rental_value": sum(miner_rental_values.values()),
                    "miner_rental_summary": [
                        {"miner_hotkey": m["miner_hotkey"], "total_rental_value": m["total_rental_value"], "num_executors": m["num_executors"]}
                        for m in miner_rental_details
                    ],
                },
            )
        )

        logger.debug(
            _m(
                "Per-miner rental value details",
                extra={
                    "miner_rental_details": miner_rental_details,
                },
            )
        )

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
        score_breakdown = []
        num_burners = 0
        num_regular_miners = 0

        for miner in miners:
            hotkey = miner.hotkey
            pool_source = None
            mining_score_contribution = 0.0
            rental_score_contribution = 0.0

            # Burners get burn_share
            if settings.ENABLE_NEW_BURN_LOGIC:
                if miner.uid in settings.NEW_BURNERS:
                    burner_score = burn_share / len(settings.NEW_BURNERS)
                    cycle_scores[hotkey] = burner_score
                    pool_source = "burn_pool_new_logic"
                    num_burners += 1

                    logger.debug(
                        _m(
                            "Miner assigned to burn pool (new logic)",
                            extra={
                                "miner_uid": miner.uid,
                                "miner_hotkey": hotkey,
                                "burn_share": burn_share,
                                "num_burners": len(settings.NEW_BURNERS),
                                "score": burner_score,
                                "pool": "burn",
                            },
                        )
                    )
                    continue
            else:
                # Old burn logic
                import random
                main_burner = random.Random(last_mechanism_step_block).choice(settings.BURNERS)
                other_burners = [uid for uid in settings.BURNERS if uid != main_burner]

                if miner.uid == main_burner:
                    main_burner_score = burn_share - (len(settings.BURNERS) - 1) * (burn_share / len(settings.BURNERS))
                    cycle_scores[hotkey] = main_burner_score
                    pool_source = "burn_pool_main_burner"
                    num_burners += 1

                    logger.debug(
                        _m(
                            "Miner assigned as main burner (old logic)",
                            extra={
                                "miner_uid": miner.uid,
                                "miner_hotkey": hotkey,
                                "burn_share": burn_share,
                                "num_burners": len(settings.BURNERS),
                                "score": main_burner_score,
                                "pool": "burn_main",
                            },
                        )
                    )
                    continue
                elif miner.uid in other_burners:
                    other_burner_score = burn_share / len(settings.BURNERS)
                    cycle_scores[hotkey] = other_burner_score
                    pool_source = "burn_pool_other_burner"
                    num_burners += 1

                    logger.debug(
                        _m(
                            "Miner assigned as other burner (old logic)",
                            extra={
                                "miner_uid": miner.uid,
                                "miner_hotkey": hotkey,
                                "burn_share": burn_share,
                                "num_burners": len(settings.BURNERS),
                                "score": other_burner_score,
                                "pool": "burn_other",
                            },
                        )
                    )
                    continue

            # Regular miners get mining + rental scores
            score = 0.0
            num_regular_miners += 1

            # Mining emission (for rented/non-eligible GPUs)
            if hotkey in miner_scores and total_mining_score > 0:
                mining_score_contribution = mining_share * miner_scores[hotkey] / total_mining_score
                score += mining_score_contribution

            # Rental emission (for unrented eligible GPUs)
            if hotkey in miner_rental_values and total_rental_value > 0:
                rental_score_contribution = rental_share * miner_rental_values[hotkey] / total_rental_value
                score += rental_score_contribution

            cycle_scores[hotkey] = score

            if score > 0:
                pool_source = []
                if mining_score_contribution > 0:
                    pool_source.append("mining")
                if rental_score_contribution > 0:
                    pool_source.append("rental")
                pool_source = "+".join(pool_source) if pool_source else "none"

                score_breakdown.append({
                    "miner_uid": miner.uid,
                    "miner_hotkey": hotkey,
                    "mining_pool_score": mining_score_contribution,
                    "rental_pool_score": rental_score_contribution,
                    "total_score": score,
                    "pool_source": pool_source,
                    "mining_raw_score": miner_scores.get(hotkey, 0),
                    "rental_value": miner_rental_values.get(hotkey, 0),
                })

                logger.debug(
                    _m(
                        "Regular miner score breakdown",
                        extra={
                            "miner_uid": miner.uid,
                            "miner_hotkey": hotkey,
                            "mining_pool_contribution": mining_score_contribution,
                            "rental_pool_contribution": rental_score_contribution,
                            "total_score": score,
                            "pool_source": pool_source,
                            "raw_mining_score": miner_scores.get(hotkey, 0),
                            "raw_rental_value": miner_rental_values.get(hotkey, 0),
                            "formula": f"({mining_share} * {miner_scores.get(hotkey, 0)} / {total_mining_score}) + ({rental_share} * {miner_rental_values.get(hotkey, 0)} / {total_rental_value})",
                        },
                    )
                )

        logger.info(
            _m(
                "Phase 3: Distributed scores across pools",
                extra={
                    "total_mining_score": total_mining_score,
                    "total_rental_value": total_rental_value,
                    "mining_share": mining_share,
                    "rental_share": rental_share,
                    "burn_share": burn_share,
                    "num_miners_total": len(miners),
                    "num_burners": num_burners,
                    "num_regular_miners": num_regular_miners,
                    "num_miners_with_scores": len([s for s in cycle_scores.values() if s > 0]),
                    "total_distributed_score": sum(cycle_scores.values()),
                    "validation": f"total_distributed={sum(cycle_scores.values()):.6f}, expected=1.0",
                },
            )
        )

        logger.debug(
            _m(
                "Regular miner score breakdown summary",
                extra={
                    "score_breakdown": score_breakdown[:20],  # Limit to first 20 for readability
                    "total_breakdown_entries": len(score_breakdown),
                },
            )
        )

        return cycle_scores
