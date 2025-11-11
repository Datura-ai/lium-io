"""Result handling for validator task execution.

This module handles the post-pipeline processing: persisting verification data to Redis
and converting the validation context into a JobResult for reporting.
"""

import logging
from typing import Optional

from core.utils import _m
from payload_models.payloads import MinerJobRequestPayload
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason
from datura.requests.miner_requests import ExecutorSSHInfo
from services.redis_service import RedisService

from .models import JobResult
from .pipeline import Context

logger = logging.getLogger(__name__)


class ResultHandler:
    """Handles post-pipeline result processing and persistence."""

    def __init__(self, redis_service: RedisService):
        """Initialize result handler with Redis service.

        Args:
            redis_service: Redis service for persisting verification data
        """
        self.redis_service = redis_service

    async def handle_result(
        self,
        context: Context,
        miner_info: MinerJobRequestPayload,
        executor_info: ExecutorSSHInfo,
        verified_job_info: dict,
        log_text: str,
        success: bool,
    ) -> JobResult:
        """Handle task result by persisting verification data and building JobResult.

        Args:
            context: Final pipeline context containing all validation state
            miner_info: Miner job request payload
            executor_info: Executor SSH connection info
            verified_job_info: Previous verification job info from Redis
            log_text: Log message for this result
            success: Whether validation succeeded

        Returns:
            JobResult containing all validation outcomes
        """
        # Log the result
        logger.info(
            _m(
                "Handle task result",
                extra={
                    "miner_hotkey": miner_info.miner_hotkey,
                    "executor_id": executor_info.uuid,
                    "success": success,
                    "score": context.score,
                    "job_score": context.job_score,
                },
            )
        )

        # Determine log status and log appropriately
        if success:
            log_status = "info"
            logger.info(log_text)
        else:
            log_status = "warning"
            logger.warning(log_text)

        # Persist verification data to Redis
        await self._persist_verification_data(
            miner_hotkey=miner_info.miner_hotkey,
            executor_id=executor_info.uuid,
            verified_job_info=verified_job_info,
            context=context,
            success=success,
        )

        # Parse GPU model and count from gpu_model_count string
        gpu_model: Optional[str] = None
        gpu_count = 0
        gpu_model_count = context.state.gpu_model_count or ""

        if gpu_model_count and ":" in gpu_model_count:
            parts = gpu_model_count.split(":")
            gpu_model = parts[0]
            gpu_count = int(parts[1])

        # Build and return JobResult
        return JobResult(
            spec=context.state.specs,
            executor_info=executor_info,
            score=context.score,
            job_score=context.job_score,
            collateral_deposited=context.collateral_deposited,
            job_batch_id=miner_info.job_batch_id,
            log_status=log_status,
            log_text=str(log_text),
            gpu_model=gpu_model,
            gpu_count=gpu_count,
            sysbox_runtime=context.state.sysbox_runtime,
            ssh_pub_keys=context.ssh_pub_keys,
        )

    async def _persist_verification_data(
        self,
        miner_hotkey: str,
        executor_id: str,
        verified_job_info: dict,
        context: Context,
        success: bool,
    ) -> None:
        """Persist verification data to Redis based on validation outcome.

        Args:
            miner_hotkey: Miner's hotkey
            executor_id: Executor UUID
            verified_job_info: Previous verification info
            context: Pipeline context with verification state
            success: Whether validation succeeded
        """
        if success:
            # On success, update verification data if we have GPU info
            gpu_model_count = context.state.gpu_model_count or ""
            gpu_uuids = context.state.gpu_uuids or ""

            if gpu_model_count and gpu_uuids:
                await self.redis_service.set_verified_job_info(
                    miner_hotkey=miner_hotkey,
                    executor_id=executor_id,
                    prev_info=verified_job_info,
                    success=True,
                    spec=gpu_model_count,
                    uuids=gpu_uuids,
                )
        else:
            # On failure, either clear or mark as failed
            if context.clear_verified_job_info:
                reason = self._resolve_clear_reason(context.clear_verified_job_reason)
                await self.redis_service.clear_verified_job_info(
                    miner_hotkey=miner_hotkey,
                    executor_id=executor_id,
                    prev_info=verified_job_info,
                    reason=reason,
                )
            else:
                await self.redis_service.set_verified_job_info(
                    miner_hotkey=miner_hotkey,
                    executor_id=executor_id,
                    prev_info=verified_job_info,
                    success=False,
                )

    @staticmethod
    def _resolve_clear_reason(reason_value: Optional[str]) -> ResetVerifiedJobReason:
        """Resolve clear reason string to enum value.

        Args:
            reason_value: String representation of clear reason

        Returns:
            ResetVerifiedJobReason enum value
        """
        if not reason_value:
            return ResetVerifiedJobReason.DEFAULT

        try:
            return ResetVerifiedJobReason(reason_value)
        except ValueError:
            return ResetVerifiedJobReason.DEFAULT
