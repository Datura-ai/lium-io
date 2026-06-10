"""Default incentive algorithm implementation.

This implementation extracts the original score calculation and weight distribution
logic to maintain backward compatibility with the existing system.
"""

from datetime import datetime

import bittensor

from core.config import settings
from core.utils import _m, get_extra_info, get_logger
from incentive.base import BaseIncentive
from services.const import TOTAL_BURN_EMISSION
from services.task_service import JobResult

logger = get_logger(__name__)


class DefaultIncentive(BaseIncentive):
    """Default incentive algorithm.

    Implements the original scoring and weight distribution logic from the validator.
    This maintains backward compatibility with the existing incentive system.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.burn_share = TOTAL_BURN_EMISSION

        # Metrics
        self.total_executors = 0
        self.successful_executors = 0
        self.failed_executors = 0
        
        # Incentive states
        self.total_mining_score = 0
        self.miner_incentives = {}
        self.mining_share = 1 - TOTAL_BURN_EMISSION

    def get_discord_multiplier(self, job_result: JobResult) -> float:
        if datetime.utcnow() < settings.DISCORD_INCENTIVE_CUTOFF:
            return 1
        if job_result.provider_discord_connected:
            return 1
        return 1 - settings.PORTION_FOR_DISCORD

    async def _pre_process_job_result(self, hotkey: str, result: JobResult):
        """Process a job result.

        Args:
            result: Job execution result to process
        """
        self.total_executors += 1
        result = await self.calculate_executor_score(result)
        self.total_mining_score += result.mining_score
        if result.job_score == 1.0:
            self.successful_executors += 1
        else:
            self.failed_executors += 1
        return result

    async def _post_process_job_result(self, hotkey: str, result: JobResult):
        """Process a job result.

        Args:
            result: Job execution result to process
        """
        if result.mining_score is None:
            result.incentive_logs.append(
                _m(
                    "Mining score is not set for job result. This should not happen.",
                    extra=get_extra_info({
                        "hotkey": hotkey,
                        "executor_id": str(result.executor_info.uuid),
                        "score": result.score,
                        "job_score": result.job_score,
                        "gpu_model": result.gpu_model,
                        "gpu_count": result.gpu_count,
                    }),
                ).to_full_string()
            )
            return result

        result.incentive = (self.mining_share * result.mining_score / self.total_mining_score) if self.total_mining_score > 0 else 0.0
        self.miner_incentives[hotkey] = self.miner_incentives.get(hotkey, 0.0) + result.incentive
        result.incentive_logs.append(
            _m(
                "Incentive score is calculated successfully. Formula: mining_share * mining_score / total_mining_score",
                extra=get_extra_info({
                    "hotkey": hotkey,
                    "executor_id": str(result.executor_info.uuid),
                    "mining_score": result.mining_score,
                    "total_mining_score": self.total_mining_score,
                    "mining_share": self.mining_share,
                    "gpu_model": result.gpu_model,
                    "gpu_count": result.gpu_count,
                    "incentive": result.incentive,
                }),
            ).to_full_string()
        )
        return result

    async def calculate_executor_score(
        self,
        job_result: JobResult,
    ) -> JobResult:
        """Calculate mining score for a single executor/job result.

        This method implements the original calc_job_score() logic from validator.py
        lines 146-199, including GPU count calculation, base scoring, and multipliers
        for sysbox runtime and uptime.

        Args:
            job_result: Job execution result to score

        Returns:
            JobResult with calculated mining score
        """
        # Early exit check - if job score is 0, return immediately
        if not job_result.score:
            job_result.mining_score = 0
            return job_result

        # Fallback for deployments using the legacy/default incentive algorithm
        # directly. The active rental_price algorithm handles this before it
        # calls into DefaultIncentive.
        if job_result.is_new_rentals_paused and not job_result.is_rented:
            logger.info(
                _m(
                    "Executor excluded from mining pool - paused for new rentals",
                    extra=get_extra_info(
                        {
                            "executor_id": str(job_result.executor_info.uuid),
                            "gpu_model": job_result.gpu_model,
                            "gpu_count": job_result.gpu_count,
                            "reason": "new_rentals_paused",
                            "score": 0,
                            "pool": "none",
                        }
                    ),
                )
            )
            job_result.mining_score = 0
            return job_result

        # GPU count calculation
        job_result.total_gpu_count = self.total_gpu_model_count_map.get(job_result.gpu_model, 0)
        if not job_result.total_gpu_count:
            job_result.mining_score = 0
            return job_result

        # Base score calculation
        job_result.gpu_portion = await self.redis_service.get_portion_per_gpu_type(job_result.gpu_model)
        job_result.mining_score = job_result.score * job_result.gpu_portion * job_result.gpu_count / job_result.total_gpu_count

        # Sysbox runtime multiplier
        is_rented_after_cutoff = (
            job_result.is_rented
            and job_result.rental_created_at
            and job_result.rental_created_at >= settings.SYSBOX_RENTED_CUTOFF
        )
        if job_result.sysbox_runtime:
            job_result.sysbox_multiplier = 1
        else:
            portion = settings.PORTION_FOR_SYSBOX_RENTED if is_rented_after_cutoff else settings.PORTION_FOR_SYSBOX

            job_result.sysbox_multiplier = 1 - portion

        # Provider Discord connection multiplier
        job_result.discord_multiplier = self.get_discord_multiplier(job_result)

        # Uptime multiplier
        if settings.SKIP_COLLATERAL_PENALTY or job_result.collateral_deposited:
            job_result.uptime_multiplier = 1
        else:
            uptime_in_minutes = await self.redis_service.get_executor_uptime(job_result.executor_info)
            job_result.uptime_multiplier = (
                1
                - settings.PORTION_FOR_UPTIME
                + settings.PORTION_FOR_UPTIME
                * min(1, uptime_in_minutes / settings.UPTIME_REQUIRED_MINUTES)
            )

        # Apply multiplier
        job_result.mining_score *= (
            job_result.sysbox_multiplier
            * job_result.discord_multiplier
            * job_result.uptime_multiplier
        )
        log = _m(
            "Mining score is calculated successfully. Formula: score * gpu_portion * gpu_count / total_gpu_count * sysbox_multiplier * discord_multiplier * uptime_multiplier",
            extra=get_extra_info(
                {
                    "executor_id": str(job_result.executor_info.uuid),
                    "gpu_model": job_result.gpu_model,
                    "gpu_count": job_result.gpu_count,
                    "sysbox_multiplier": job_result.sysbox_multiplier,
                    "discord_multiplier": job_result.discord_multiplier,
                    "provider_discord_connected": job_result.provider_discord_connected,
                    "uptime_multiplier": job_result.uptime_multiplier,
                    "mining_score": job_result.mining_score,
                    "gpu_portion": job_result.gpu_portion,
                    "total_gpu_count": job_result.total_gpu_count,
                    "rental_created": job_result.rental_created_at,
                    "is_rented_after_cutoff": is_rented_after_cutoff,
                }
            ),
        )
        job_result.incentive_logs.append(
            log.to_full_string()
        )
        logger.info(log)
        return job_result

    async def calculate_final_weights(
        self,
        miners: list[bittensor.NeuronInfo],
        last_mechanism_step_block: int | None,
    ) -> dict[str, float]:
        """Calculate scores with burning logic for this cycle.

        This method applies burning logic and returns scores to be accumulated.
        It does NOT return final weights for blockchain - those are calculated
        later in set_weights by normalizing accumulated scores.

        Args:
            miners: List of miner neuron information
            last_mechanism_step_block: Last mechanism step block number

        Returns:
            dict[str, float]: Scores with burning applied for each miner
        """
        # Calculate burn scores using BurnService
        cycle_scores = self.burn_service.calculate_burn_scores(
            miners=miners,
            burn_share=self.burn_share,  # Fixed at 0.91 for default incentive
            last_mechanism_step_block=last_mechanism_step_block,
        )
        for miner in miners:
            # Check if miner is a burner
            if miner.hotkey in cycle_scores:
                continue
            cycle_scores[miner.hotkey] = self.miner_incentives.get(miner.hotkey, 0.0)

        return cycle_scores
