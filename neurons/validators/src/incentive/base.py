"""Abstract base class for incentive algorithms."""

from abc import ABC, abstractmethod

import bittensor
import numpy as np

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
        miner_scores: dict[str, float],
        miners: list[bittensor.NeuronInfo],
        last_mechanism_step_block: int,
        all_job_results: dict[str, list[JobResult]],
        rented_data: RentedExecutorsResponse,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate final UIDs and weights for blockchain submission.

        Args:
            miner_scores: Mapping of miner hotkeys to scores
            miners: List of miner neuron information
            last_mechanism_step_block: Last mechanism step block number
            all_job_results: Mapping of miner hotkeys to their job results (needed for rental calculations)
            rented_data: Response containing all rented executors from backend API

        Returns:
            Tuple of (uids, weights) as numpy arrays
        """
        pass
