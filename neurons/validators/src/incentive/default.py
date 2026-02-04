"""Default incentive algorithm implementation.

This implementation extracts the original score calculation and weight distribution
logic to maintain backward compatibility with the existing system.
"""

import random

import bittensor

from core.config import settings
from core.utils import _m, get_extra_info, get_logger
from incentive.base import BaseIncentive
from incentive.config import IncentiveConfig
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.const import BURNER_EMISSION, TOTAL_BURN_EMISSION
from services.redis_service import RedisService
from services.task_service import JobResult

logger = get_logger(__name__)


class DefaultIncentive(BaseIncentive):
    """Default incentive algorithm.

    Implements the original scoring and weight distribution logic from the validator.
    This maintains backward compatibility with the existing incentive system.
    """

    def __init__(self, config: IncentiveConfig, redis_service: RedisService):
        """Initialize the default incentive algorithm.

        Args:
            config: Incentive configuration
            redis_service: Redis service for accessing shared state
        """
        super().__init__(config, redis_service)

    async def calculate_executor_score(
        self,
        total_gpu_model_count_map: dict,
        job_result: JobResult,
    ) -> float:
        """Calculate score for a single executor/job result.

        This method implements the original calc_job_score() logic from validator.py
        lines 146-199, including GPU count calculation, base scoring, and multipliers
        for sysbox runtime and uptime.

        Args:
            total_gpu_model_count_map: Mapping of GPU models to total counts
            job_result: Job execution result to score

        Returns:
            Calculated score for the executor
        """
        # Early exit check - if job score is 0, return immediately
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

        # Apply multiplier
        score *= multiplier

        # Logging
        logger.debug(
            _m(
                "Calculated executor score",
                extra={
                    "total_gpu_count": total_gpu_count,
                    "score_portion": score_portion,
                    "multiplier": multiplier,
                    "final_score": score,
                },
            )
        )

        return score

    async def calculate_final_weights(
        self,
        miners: list[bittensor.NeuronInfo],
        last_mechanism_step_block: int | None,
        all_job_results: dict[str, list[JobResult]],
        rented_data: RentedExecutorsResponse,
    ) -> dict[str, float]:
        """Calculate scores with burning logic for this cycle.

        This method applies burning logic and returns scores to be accumulated.
        It does NOT return final weights for blockchain - those are calculated
        later in set_weights by normalizing accumulated scores.

        Args:
            miner_scores: Mining scores from this cycle only (not accumulated)
            miners: List of miner neuron information
            last_mechanism_step_block: Last mechanism step block number
            all_job_results: Mapping of miner hotkeys to their job results (unused in default)
            rented_data: Response containing all rented executors (unused in default)

        Returns:
            dict[str, float]: Scores with burning applied for each miner
        """
        miner_scores = self.temp_miner_scores
        cycle_scores = {}
        total_mining_score = sum(miner_scores.values())

        if settings.ENABLE_NEW_BURN_LOGIC:
            # New burn logic
            burners = settings.NEW_BURNERS
            burn_score_per_burner = TOTAL_BURN_EMISSION / len(burners)

            for miner in miners:
                if miner.uid in burners:
                    # Burners get burn emission share
                    cycle_scores[miner.hotkey] = burn_score_per_burner
                else:
                    # Regular miners share mining emission proportionally
                    if total_mining_score > 0:
                        mining_share = (1 - TOTAL_BURN_EMISSION) * miner_scores.get(miner.hotkey, 0.0) / total_mining_score
                    else:
                        mining_share = 0.0
                    cycle_scores[miner.hotkey] = mining_share
        else:
            # Old burn logic with main burner
            main_burner = random.Random(last_mechanism_step_block or 0).choice(settings.BURNERS)
            other_burners = [uid for uid in settings.BURNERS if uid != main_burner]

            main_burner_score = TOTAL_BURN_EMISSION - (len(settings.BURNERS) - 1) * BURNER_EMISSION

            for miner in miners:
                if miner.uid == main_burner:
                    cycle_scores[miner.hotkey] = main_burner_score
                elif miner.uid in other_burners:
                    cycle_scores[miner.hotkey] = BURNER_EMISSION
                else:
                    if total_mining_score > 0:
                        mining_share = (1 - TOTAL_BURN_EMISSION) * miner_scores.get(miner.hotkey, 0.0) / total_mining_score
                    else:
                        mining_share = 0.0
                    cycle_scores[miner.hotkey] = mining_share

        logger.debug(
            _m(
                "Calculated cycle scores with burning",
                extra={
                    "total_mining_score": total_mining_score,
                    "num_miners": len(miners),
                    "enable_new_burn_logic": settings.ENABLE_NEW_BURN_LOGIC,
                },
            )
        )

        return cycle_scores
