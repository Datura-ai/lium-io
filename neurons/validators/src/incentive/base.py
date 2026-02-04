"""Abstract base class for incentive algorithms."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING  

import bittensor

from incentive.config import IncentiveConfig
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.redis_service import RedisService
from services.task_service import JobResult
from services.const import TOTAL_BURN_EMISSION
if TYPE_CHECKING:
    from incentive.burn_service import BurnService


class BaseIncentive(ABC):
    """Abstract base class for incentive calculation algorithms.

    All incentive algorithms must implement the calculate_executor_score
    and calculate_final_weights methods.
    """

    def __init__(
        self, 
        config: IncentiveConfig, 
        redis_service: RedisService, 
        burn_service: "BurnService", 
        jobs_results: dict[str, list[JobResult]],
        total_gpu_model_count_map: dict,
    ):
        """Initialize the incentive algorithm.

        Args:
            config: Incentive configuration
            redis_service: Redis service for accessing shared state
            burn_service: Burn emission distribution service
        """
        self.config = config
        self.redis_service = redis_service
        self.burn_service = burn_service
        self.job_results = jobs_results
        self.total_gpu_model_count_map = total_gpu_model_count_map

        

    async def caluclate_mining_scores(self):
        """Normalize the score of a job result.

        Args:
            all_job_results: All job results

        Returns:
            None
        """
        for hotkey, results in self.job_results.items():
            for result in results:
                await self._pre_process_job_result(hotkey, result)

        await self._on_finish_pre_process()

        for hotkey, results in self.job_results.items():
            for result in results:
                await self._post_process_job_result(hotkey, result)

    async def _pre_process_job_result(self, hotkey: str, result: JobResult) -> JobResult:
        """Callback before post-processing a job result.

        Args:
            result: Job execution result to process
        """
        pass


    async def _post_process_job_result(self, hotkey: str, result: JobResult) -> JobResult:
        """Callback after post-processing a job result.

        Args:
            result: Job execution result to process
        """
        pass


    async def _on_finish_pre_process(self) -> None:
        """Callback after pre-processing all job results.

        Args:
            None

        Returns:
            None
        """
        pass

    @abstractmethod
    async def calculate_executor_score(
        self,
        job_result: JobResult,
    ) -> JobResult:
        """
        Calculate mining score for a single executor/job result.

        Args:
            job_result: Job execution result to score

        Returns:
            JobResult with calculated mining score
        """
        raise NotImplementedError

    @abstractmethod
    async def calculate_final_weights(
        self,
        miners: list[bittensor.NeuronInfo],
        last_mechanism_step_block: int | None,
        rented_data: RentedExecutorsResponse,
    ) -> dict[str, float]:
        """Calculate final weights with burning logic applied for this cycle.

        This method receives mining scores from Phase 1 (calculate_executor_score)
        for the current cycle only. It applies burning logic and returns weights
        that will be accumulated across cycles.

        Args:
            miners: List of all miners (needed to identify burners by UID)
            last_mechanism_step_block: For burner selection randomization
            rented_data: Response containing all rented executors from backend API

        Returns:
            dict[str, float]: Weights with burning applied for each miner.
                - Burners receive high scores (proportional to TOTAL_BURN_EMISSION)
                - Regular miners receive low scores (proportional to 1 - TOTAL_BURN_EMISSION)
                These weights are accumulated across cycles in validator.miner_scores
        """
        pass
