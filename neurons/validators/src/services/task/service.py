import logging
from typing import TYPE_CHECKING, Annotated

import bittensor
from clients.backend_client import BackendClient
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.attestation_service import AttestationError, AttestationNonce, AttestationService
from services.collateral_contract_service import CollateralContractService
from services.executor_connectivity_service import ExecutorConnectivityService
from services.interactive_shell_service import InteractiveShellService
from services.matrix_validation_service import ValidationService
from services.redis_service import RedisService
from services.verifyx_validation_service import VerifyXValidationService

from core.config import settings
from core.utils import _m, get_extra_info
from services.ssh_service import SSHService

from .models import JobResult
from .pipeline_factory import PipelineFactory
from .result_handler import ResultHandler

if TYPE_CHECKING:
    # Deferred: docker_service imports this package, so a runtime import here is a cycle.
    from services.docker_service import DockerService

logger = logging.getLogger(__name__)

class TaskService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
        validation_service: Annotated[ValidationService, Depends(ValidationService)],
        verifyx_validation_service: Annotated[VerifyXValidationService, Depends(VerifyXValidationService)],
        collateral_contract_service: Annotated[CollateralContractService, Depends(CollateralContractService)],
        executor_connectivity_service: Annotated[ExecutorConnectivityService, Depends(ExecutorConnectivityService)],
        backend_client: Annotated[BackendClient, Depends(BackendClient)],
        attestation_service: Annotated[AttestationService, Depends(AttestationService)],
        docker_service: "DockerService",
    ):
        self.ssh_service = ssh_service
        self.redis_service = redis_service
        self.attestation_service = attestation_service
        self.wallet = settings.get_bittensor_wallet()

        # Initialize pipeline factory with all required services
        self.pipeline_factory = PipelineFactory(
            ssh_service=ssh_service,
            redis_service=redis_service,
            validation_service=validation_service,
            verifyx_validation_service=verifyx_validation_service,
            collateral_contract_service=collateral_contract_service,
            executor_connectivity_service=executor_connectivity_service,
            backend_client=backend_client,
            docker_service=docker_service,
        )

    async def create_task(
        self,
        miner_info: MinerJobRequestPayload,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
        public_key: str,
        encrypted_files: MinerJobEnryptedFiles,
        rented_data: RentedExecutorsResponse,
        default_docker_image_digests: dict[str, str],
        attestation_nonce: AttestationNonce | None = None,
    ):
        """New pipeline-based validation task implementation."""
        attestation_digest = None
        tee_type = None
        attestation_passed = False
        gpu_attestation_passed = None

        try:
            # Decrypt private key
            private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)

            # Prepare attestation host policy before SSH connection
            try:
                host_policy = await self.attestation_service.prepare_host_policy(
                    executor_info,
                    nonce=attestation_nonce,
                )
                known_hosts_policy = host_policy.known_hosts
                attestation_digest = host_policy.attestation_digest
                tee_type = host_policy.tee_type
                gpu_attestation_passed = host_policy.gpu_attestation_passed
                # G1 invariant: a verified-bad GPU can never surface as passed
                # (HostPolicyResult.attestation_passed requires tdx ∧ gpu-not-False).
                attestation_passed = host_policy.attestation_passed
            except AttestationError as exc:
                log_text = _m(
                    "Attestation failed",
                    extra=get_extra_info({
                        "job_batch_id": miner_info.job_batch_id,
                        "miner_hotkey": miner_info.miner_hotkey,
                        "executor_uuid": executor_info.uuid,
                        "error": str(exc),
                    }),
                )
                logger.error(log_text)
                raise

            async with InteractiveShellService(
                host=executor_info.address,
                username=executor_info.ssh_username,
                private_key=private_key,
                port=executor_info.ssh_port,
                known_hosts=known_hosts_policy,
            ) as shell:
                # Build validation context
                base_ctx = await self.pipeline_factory.build_context(
                    shell=shell,
                    miner_info=miner_info,
                    executor_info=executor_info,
                    keypair=keypair,
                    private_key=private_key,
                    public_key=public_key,
                    encrypted_files=encrypted_files,
                    rented_data=rented_data,
                    default_docker_image_digests=default_docker_image_digests,
                    tdx_attestation_passed=attestation_passed,
                    gpu_attestation_passed=gpu_attestation_passed,
                )

                # Build and run validation pipeline
                # Use dry run pipeline if DRY_RUN mode is enabled to avoid state changes
                if settings.DRY_RUN:
                    checks = self.pipeline_factory.build_dry_run_checks()
                else:
                    checks = self.pipeline_factory.build_checks()
                pipeline = self.pipeline_factory.build_pipeline(checks)
                ok, events, last_context = await pipeline.run(base_ctx)

                # Determine log_text and success based on ok status
                # Always use the last event for structured logging (Grafana consumption)
                last_event = events[-1]
                log_text = _m(last_event.event, extra=last_event.model_dump())
                success = False if not ok else last_context.success

                # Handle result using ResultHandler
                result_handler = ResultHandler(self.redis_service, dry_run=settings.DRY_RUN)
                result = await result_handler.handle_result(
                    context=last_context,
                    miner_info=miner_info,
                    executor_info=executor_info,
                    verified_job_info=base_ctx.verified,  # From original context
                    log_text=log_text.to_full_string(),
                    success=success,
                )
                result.attestation_digest = attestation_digest
                result.tee_type = tee_type
                result.gpu_attestation_passed = gpu_attestation_passed
                return result

        except Exception as e:
            log_text = _m(
                "Pipeline validation error",
                extra=get_extra_info({
                    "job_batch_id": miner_info.job_batch_id,
                    "miner_hotkey": miner_info.miner_hotkey,
                    "executor_uuid": executor_info.uuid,
                    "executor_ip_address": executor_info.address,
                    "executor_port": executor_info.port,
                    "ssh_user": executor_info.ssh_username,
                    "ssh_port": executor_info.ssh_port,
                    "error": str(e),
                })
            )
            logger.error(log_text, exc_info=True)
            return JobResult(
                spec=None,
                executor_info=executor_info,
                score=0,
                job_score=0,
                collateral_deposited=False,
                job_batch_id=miner_info.job_batch_id,
                log_status="error",
                log_text=log_text.to_full_string(),
                gpu_model=None,
                gpu_count=0,
                sysbox_runtime=False,
                attestation_digest=attestation_digest,
                tee_type=tee_type,
                gpu_attestation_passed=gpu_attestation_passed,
            )


TaskServiceDep = Annotated[TaskService, Depends(TaskService)]
