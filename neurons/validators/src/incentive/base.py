"""Abstract base class for incentive algorithms."""

from abc import ABC, abstractmethod

import bittensor

from incentive.config import IncentiveConfig
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.redis_service import RedisService
from services.task_service import JobResult


class BaseIncentive(ABC):
    """Abstract base class for incentive calculation algorithms.

    All incentive algorithms must implement the calculate_executor_score
    and calculate_final_weights methods.
    """

    def __init__(self, config: IncentiveConfig, redis_service: RedisService):
        """Initialize the incentive algorithm.

        Args:
            config: Incentive configuration
            redis_service: Redis service for accessing shared state
        """
        self.config = config
        self.redis_service = redis_service

        self.total_executors = 0
        self.successful_executors = 0
        self.failed_executors = 0

        self.temp_miner_scores = {}

    async def normalize_job_result_score(
        self, all_job_results: dict[str, list[JobResult]], total_gpu_model_count_map: dict,
    ):
        """Normalize the score of a job result.

        Args:
            all_job_results: All job results

        Returns:
            None
        """
        for miner_hotkey, results in all_job_results.items():
            for result in results:
                self.total_executors += 1
                # Replace calc_job_score with incentive.calculate_executor_score
                result.mining_score = await self.calculate_executor_score(
                    total_gpu_model_count_map=total_gpu_model_count_map,
                    job_result=result,
                )
                self.temp_miner_scores[miner_hotkey] = self.temp_miner_scores.get(miner_hotkey, 0) + result.mining_score
                if result.job_score == 1.0:
                    self.successful_executors += 1
                else:
                    self.failed_executors += 1

    @abstractmethod
    async def calculate_executor_score(
        self,
        total_gpu_model_count_map: dict,
        job_result: JobResult,
    ) -> float:
        """Calculate score for a single executor/job result.

        Args:
            total_gpu_model_count_map: Mapping of GPU models to total counts
            job_result: Job execution result to score

        Returns:
            Calculated score for the executor
        """
        raise NotImplementedError

    @abstractmethod
    async def calculate_final_weights(
        self,
        miners: list[bittensor.NeuronInfo],
        last_mechanism_step_block: int | None,
        all_job_results: dict[str, list[JobResult]],
        rented_data: RentedExecutorsResponse,
    ) -> dict[str, float]:
        """Calculate final scores with burning logic applied for this cycle.

        This method receives mining scores from Phase 1 (calculate_executor_score)
        for the current cycle only. It applies burning logic and returns scores
        that will be accumulated across cycles.

        Args:
            miner_scores: Mining scores from this cycle only (not accumulated)
            miners: List of all miners (needed to identify burners by UID)
            last_mechanism_step_block: For burner selection randomization
            all_job_results: Job results from this cycle (for rental calculations)
            rented_data: Response containing all rented executors from backend API

        Returns:
            dict[str, float]: Scores with burning applied for each miner.
                - Burners receive high scores (proportional to TOTAL_BURN_EMISSION)
                - Regular miners receive low scores (proportional to 1 - TOTAL_BURN_EMISSION)
                These scores are accumulated across cycles in validator.miner_scores
        """
        pass
