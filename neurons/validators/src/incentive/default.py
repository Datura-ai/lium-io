"""Default incentive algorithm implementation.

This implementation extracts the original score calculation and weight distribution
logic to maintain backward compatibility with the existing system.
"""

import random

import bittensor
import numpy as np

from core.config import settings
from core.utils import _m, get_extra_info, get_logger
from incentive.base import BaseIncentive
from incentive.config import IncentiveConfig
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
            logger.debug(
                _m(
                    "Job score is 0, returning 0",
                    extra=get_extra_info(job_result),
                )
            )
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
                    **get_extra_info(job_result),
                    "total_gpu_count": total_gpu_count,
                    "score_portion": score_portion,
                    "multiplier": multiplier,
                    "final_score": score,
                },
            )
        )

        return score

    def calculate_final_weights(
        self,
        miner_scores: dict[str, float],
        miners: list[bittensor.NeuronInfo],
        last_mechanism_step_block: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate final UIDs and weights for blockchain submission.

        This method implements the burning logic from subtensor_client.py lines 336-347,
        including burner selection and weight distribution based on miner scores.

        Args:
            miner_scores: Mapping of miner hotkeys to scores
            miners: List of miner neuron information
            last_mechanism_step_block: Last mechanism step block number

        Returns:
            Tuple of (uids, weights) as numpy arrays
        """
        # Initialize arrays
        uids = np.zeros(len(miners), dtype=np.int64)
        weights = np.zeros(len(miners), dtype=np.float32)

        # Burner selection
        main_burner = random.Random(last_mechanism_step_block).choice(settings.BURNERS)
        other_burners = [uid for uid in settings.BURNERS if uid != main_burner]

        # Weight distribution
        total_score = sum(miner_scores.values())

        for ind, miner in enumerate(miners):
            uids[ind] = miner.uid

            if settings.ENABLE_NEW_BURN_LOGIC:
                # New burn logic
                if miner.uid in settings.NEW_BURNERS:
                    weights[ind] = TOTAL_BURN_EMISSION / len(settings.NEW_BURNERS)
                else:
                    weights[ind] = (
                        0
                        if total_score <= 0
                        else (1 - TOTAL_BURN_EMISSION)
                        * miner_scores.get(miner.hotkey, 0.0)
                        / total_score
                    )
            else:
                # Old burn logic
                if miner.uid == main_burner:
                    weights[ind] = TOTAL_BURN_EMISSION - (len(settings.BURNERS) - 1) * BURNER_EMISSION
                elif miner.uid in other_burners:
                    weights[ind] = BURNER_EMISSION
                else:
                    weights[ind] = (
                        0
                        if total_score <= 0
                        else (1 - TOTAL_BURN_EMISSION)
                        * miner_scores.get(miner.hotkey, 0.0)
                        / total_score
                    )

        logger.debug(
            _m(
                "Calculated final weights",
                extra={
                    "total_score": total_score,
                    "num_miners": len(miners),
                    "main_burner": main_burner,
                    "enable_new_burn_logic": settings.ENABLE_NEW_BURN_LOGIC,
                },
            )
        )

        return uids, weights
