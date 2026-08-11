"""Result handling for validator task execution.

This module handles the post-pipeline processing: persisting verification data to Redis
and converting the validation context into a JobResult for reporting.
"""

import logging
from datetime import datetime
from typing import Optional

from core.utils import _m, get_extra_info
from payload_models.payloads import MinerJobRequestPayload
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason
from datura.requests.miner_requests import ExecutorSSHInfo
from services.gpu_spec_table import normalize_gpu_model
from services.redis_service import INSPECTOR_EVENT_CHANNEL, RedisService

from .models import JobResult
from .pipeline import Context

logger = logging.getLogger(__name__)


def _canonicalize_published_gpu_names(specs: dict) -> dict:
    gpu = specs.get("gpu")
    if not isinstance(gpu, dict):
        return specs

    details = gpu.get("details")
    if not isinstance(details, list):
        return specs

    canonical_details = [
        {**detail, "name": normalize_gpu_model(detail.get("name"))}
        if isinstance(detail, dict) and detail.get("name")
        else detail
        for detail in details
    ]
    if canonical_details == details:
        return specs

    return {**specs, "gpu": {**gpu, "details": canonical_details}}


class ResultHandler:
    """Handles post-pipeline result processing and persistence."""

    def __init__(self, redis_service: RedisService, dry_run: bool = False):
        """Initialize result handler with Redis service.

        Args:
            redis_service: Redis service for persisting verification data
            dry_run: If True, skip Redis writes
        """
        self.redis_service = redis_service
        self.dry_run = dry_run

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

        # Persist verification data to Redis (unless in DRY_RUN mode)
        if not self.dry_run:
            await self._persist_verification_data(
                miner_hotkey=miner_info.miner_hotkey,
                executor_id=executor_info.uuid,
                verified_job_info=verified_job_info,
                context=context,
                success=success,
            )
        else:
            logger.info("DRY_RUN: Skipping Redis persistence")

        if not self.dry_run and context.state.inspector_event:
            await self._publish_inspector_event(
                context.state.inspector_event,
                miner_info.miner_hotkey,
            )

        # Parse GPU model and count from gpu_model_count string
        gpu_model: Optional[str] = None
        gpu_count = 0
        gpu_model_count = context.state.gpu_model_count or ""

        if gpu_model_count and ":" in gpu_model_count:
            parts = gpu_model_count.split(":")
            gpu_model = parts[0]
            gpu_count = int(parts[1])

        is_spot = bool(
            context.state.rented_data
            and executor_info.uuid in context.state.rented_data.spot_executor_ids
        )
        is_new_rentals_paused = bool(
            context.state.rented_data
            and executor_info.uuid in context.state.rented_data.new_rentals_paused_executor_ids
        )
        default_job_owner = (
            context.state.rented_data.get_default_job_owner(executor_info.uuid)
            if context.state.rented_data
            else None
        )
        provider_discord_connected = self._get_provider_discord_connected(context)

        # add TDX attestation and spot tier to specs (propagated to compute-app
        # via MACHINE_SPEC_CHANNEL → executor.specs)
        specs = _canonicalize_published_gpu_names(
            {
                **(context.state.specs or {}),
                "tdx_attestation_passed": context.tdx_attestation_passed,
                "is_spot": is_spot,
            }
        )
        # G1 — NVIDIA CC GPU attestation outcome. Only added when a verification
        # was actually performed (None → key omitted), mirroring gpu_metrics.
        # Rides executor.specs to the backend like tdx_attestation_passed.
        if context.gpu_attestation_passed is not None:
            specs["gpu_attestation_passed"] = context.gpu_attestation_passed
        # DAH-2582 — the hardware-bound identities this attestation established. Published so
        # the backend can dedup on attested identity rather than only on the GPU UUIDs it
        # scrapes: a scraped UUID is asserted by software on the host, while these came out of
        # signed evidence, so the backend's duplicate detection stops depending on the honesty
        # of the node it is checking. Omitted entirely for a node that attested nothing, which
        # keeps a non-CVM executor's specs exactly as they were.
        if context.state.attested_identities:
            specs["attested_identities"] = {
                identity_class: list(values)
                for identity_class, values in context.state.attested_identities.items()
                if values
            }
        # FP32 TFLOPS metrics (device-0 representative), captured by CapabilityCheck.
        # Only added when present; absent → fail-safe (TFLOPS disabled/failed). The
        # backend types this nested object as MachineSpecs.gpu_metrics.
        if context.state.gpu_metrics is not None:
            specs["gpu_metrics"] = context.state.gpu_metrics

        # DAH-2265 Plan 2: advisory cached-template signal — whether this executor has the
        # recommended default image pre-pulled (so DOCKER_PULL is a no-op for default-template
        # rentals). Captured by CachedTemplateVerificationCheck. Only published when measured
        # (None → skipped/fail-open, key omitted). Rides executor.specs like gpu_metrics; no
        # scoring impact.
        if context.state.recommended_image_cached is not None:
            specs["recommended_image_cached"] = context.state.recommended_image_cached

        # DAH-2265 digest: advisory digest-match for the recommended image. False = the node
        # holds STALE content under an unchanged tag (re-pull needed). Only published when
        # compared (None → not cached / no backend digest / unreadable, key omitted). Rides
        # executor.specs like recommended_image_cached; no scoring impact.
        if context.state.recommended_image_digest_match is not None:
            specs["recommended_image_digest_match"] = context.state.recommended_image_digest_match

        # NVIDIA driver version string reported by NVML (e.g. "580.95.05"). Used by the
        # minimum-driver requirement gate.
        nvidia_driver_version = str(specs.get("gpu", {}).get("driver") or "")
        inspector_outcome = (
            context.state.inspector_event.get("outcome")
            if context.state.inspector_event
            else "SKIPPED"
        )

        # Build and return JobResult
        return JobResult(
            spec=specs,
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
            nvidia_driver_version=nvidia_driver_version,
            supports_gpu_splitting=context.state.supports_gpu_splitting,
            gpu_splitting_min_count=context.state.gpu_splitting_min_count,
            ssh_pub_keys=context.ssh_pub_keys,
            is_rented=context.rented,
            is_spot=is_spot,
            is_new_rentals_paused=is_new_rentals_paused,
            is_provider_banned=context.is_provider_banned,
            provider_discord_connected=provider_discord_connected,
            rental_created_at=self._get_rental_created_at(context),
            default_job_owner=default_job_owner,
            tdx_attestation_passed=context.tdx_attestation_passed,
            gpu_attestation_passed=context.gpu_attestation_passed,
            inspector_outcome=inspector_outcome,
        )

    @staticmethod
    def _get_rental_created_at(context: Context) -> datetime | None:
        rented_data = context.state.rented_data
        if not rented_data or not context.rented:
            return None
        rented_executor = rented_data.executors.get(context.executor.uuid)
        if not rented_executor or not rented_executor.pods:
            return None
        created_dates = [p.created_at for p in rented_executor.pods if p.created_at]
        return max(created_dates) if created_dates else None

    @staticmethod
    def _get_provider_discord_connected(context: Context) -> bool:
        rented_data = context.state.rented_data
        if not rented_data:
            return True

        connected_executor_ids = rented_data.provider_discord_connected_executor_ids
        if connected_executor_ids is None:
            return True

        return str(context.executor.uuid) in connected_executor_ids

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

    async def _publish_inspector_event(self, event: dict, miner_hotkey: str) -> None:
        payload = {
            **event,
            "miner_hotkey": miner_hotkey,
        }
        try:
            await self.redis_service.publish(
                INSPECTOR_EVENT_CHANNEL,
                payload,
            )
        except Exception as e:
            logger.error(
                _m(
                    "Failed to publish inspector event",
                    extra=get_extra_info(
                        {
                            "miner_hotkey": miner_hotkey,
                            "executor_id": event.get("executor_id"),
                            "error": str(e),
                        }
                    ),
                ),
                exc_info=True,
            )
