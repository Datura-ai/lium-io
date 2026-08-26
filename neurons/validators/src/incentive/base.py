"""Abstract base class for incentive algorithms."""

import time
from abc import ABC, abstractmethod

import bittensor
from pydantic import BaseModel
from services.redis_service import RedisService
from services.task_service import JobResult

from incentive.burn_service import BurnService
from incentive.config import IncentiveConfig
from incentive.utils import log_for_monitoring


class BaseIncentive(ABC):
    """Abstract base class for incentive calculation algorithms.

    All incentive algorithms must implement the calculate_executor_score
    and calculate_final_weights methods.
    """

    def __init__(
        self,
        config: IncentiveConfig,
        redis_service: RedisService,
        jobs_results: dict[str, list[JobResult]],
        total_gpu_model_count_map: dict,
        burn_service: BurnService | None = None,
    ):
        """Initialize the incentive algorithm.

        Args:
            config: Incentive configuration
            redis_service: Redis service for accessing shared state
            burn_service: Optional injected burn service, primarily for tests
        """
        self.config = config
        self.redis_service = redis_service
        self.burn_service = burn_service or BurnService()
        self.job_results = jobs_results
        self.total_gpu_model_count_map = total_gpu_model_count_map

    async def calculate_mining_scores(self):
        """Normalize the score of a job result.

        Args:
            all_job_results: All job results

        Returns:
            None
        """
        t1 = time.time()
        for hotkey, results in self.job_results.items():
            for result in results:
                await self._pre_process_job_result(hotkey, result)

        await self._on_finish_pre_process()

        for hotkey, results in self.job_results.items():
            for result in results:
                await self._post_process_job_result(hotkey, result)
                self._split_incentive_across_pools(result)

        log_for_monitoring(self.job_results, t1, getattr(self, "unrented_count_by_bucket", None))

    @staticmethod
    def _split_incentive_across_pools(result: JobResult) -> None:
        """DAH-2467: the single place a scored result gets its rented/idle breakdown.

        A whole node earns from one pool only, so the pair is just its incentive on the
        side it was scored on. A partially rented split node is merged from two portions
        afterwards, which overwrites the pair with the real two-pool split.
        """
        if result.incentive is None:
            return
        rented: float = result.incentive if result.is_rented else 0.0
        result.set_incentive_split(rented, result.incentive - rented)

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

    def get_snapshot(self) -> BaseModel | None:
        """Return a serializable snapshot of the current epoch state.

        Subclasses override this to expose their state for the /incentive-snapshot endpoint.
        """
        return None

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
        current_epoch: int | None = None,
    ) -> dict[str, float]:
        """Calculate final weights with burning logic applied for this cycle.

        This method receives mining scores from Phase 1 (calculate_executor_score)
        for the current cycle only. It applies burning logic and returns weights
        that will be accumulated across cycles.

        Args:
            miners: List of all miners (needed to identify burners by UID)
            last_mechanism_step_block: For burner selection randomization

        Returns:
            dict[str, float]: Weights with burning applied for each miner.
                - Burners receive high scores (proportional to the burn emission share)
                - Regular miners receive low scores (proportional to 1 - burn emission share)
                These weights are accumulated across cycles in validator.miner_scores
        """
        pass
