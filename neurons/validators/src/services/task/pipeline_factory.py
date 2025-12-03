"""Factory for building validation pipelines.

This module provides a clean interface for constructing the validation pipeline,
including context setup and check instantiation.
"""

import logging
import uuid
from typing import cast

import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload

from core.config import settings
from daos.port_mapping_dao import PortMappingDao
from services.collateral_contract_service import CollateralContractService
from services.const import GPU_MODEL_RATES, LIB_NVIDIA_ML_DIGESTS, MAX_GPU_COUNT
from services.executor_connectivity_service import ExecutorConnectivityService
from services.interactive_shell_service import InteractiveShellService
from services.matrix_validation_service import ValidationService
from services.redis_service import RENTAL_SUCCEED_MACHINE_SET, RedisService
from services.ssh_service import SSHService
from services.verifyx_validation_service import VerifyXValidationService

from .checks import (
    BannedGpuCheck,
    CapabilityCheck,
    CollateralCheck,
    DuplicateExecutorCheck,
    FinalizeCheck,
    GpuCountCheck,
    GpuFingerprintCheck,
    GpuModelValidCheck,
    GpuUsageCheck,
    MachineSpecScrapeCheck,
    NvmlDigestCheck,
    PortConnectivityCheck,
    PortCountCheck,
    ScoreCheck,
    SpecChangeCheck,
    StartGPUMonitorCheck,
    TenantEnforcementCheck,
    UploadFilesCheck,
    VerifyXCheck,
)
from .pipeline import (
    Check,
    Context,
    ContextConfig,
    ContextServices,
    ContextState,
    LoggerSink,
    Pipeline,
)
from .runner import SSHCommandRunner
from .score_calculator import calculate_scores

logger = logging.getLogger(__name__)

JOB_LENGTH = 300


class PipelineFactory:
    """Factory for building and configuring validation pipelines."""

    def __init__(
        self,
        ssh_service: SSHService,
        redis_service: RedisService,
        validation_service: ValidationService,
        verifyx_validation_service: VerifyXValidationService,
        collateral_contract_service: CollateralContractService,
        executor_connectivity_service: ExecutorConnectivityService,
        port_mapping_dao: PortMappingDao,
    ):
        """Initialize pipeline factory with required services.

        Args:
            ssh_service: SSH service for remote operations
            redis_service: Redis service for state management
            validation_service: Matrix validation service
            verifyx_validation_service: VerifyX validation service
            collateral_contract_service: Collateral contract service
            executor_connectivity_service: Executor connectivity service
            port_mapping_dao: Port mapping DAO
        """
        self.ssh_service = ssh_service
        self.redis_service = redis_service
        self.validation_service = validation_service
        self.verifyx_validation_service = verifyx_validation_service
        self.collateral_contract_service = collateral_contract_service
        self.executor_connectivity_service = executor_connectivity_service
        self.port_mapping_dao = port_mapping_dao

    async def build_context(
        self,
        shell: InteractiveShellService,
        miner_info: MinerJobRequestPayload,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
        public_key: str,
        encrypted_files: MinerJobEnryptedFiles,
    ) -> Context:
        """Build the base validation context with all configuration.

        Args:
            shell: Interactive shell service for SSH operations
            miner_info: Miner job request payload
            executor_info: Executor SSH connection info
            keypair: Validator's bittensor keypair
            private_key: Decrypted private key for SSH
            public_key: Public key for SSH
            encrypted_files: Encrypted validation files

        Returns:
            Configured Context ready for pipeline execution
        """
        runner = SSHCommandRunner(shell.ssh_client, max_retries=1)
        verified_job_info = await self.redis_service.get_verified_job_info(
            executor_info.uuid
        )

        default_extra = {
            "job_batch_id": miner_info.job_batch_id,
            "miner_hotkey": miner_info.miner_hotkey,
            "executor_uuid": executor_info.uuid,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
            "version": settings.VERSION,
            "rented": False,
            "renting_in_progress": False,
        }

        is_rental_succeed = await self.redis_service.is_elem_exists_in_set(
            RENTAL_SUCCEED_MACHINE_SET, executor_info.uuid
        )

        pipeline_id = str(uuid.uuid4())

        return Context(
            pipeline_id=pipeline_id,
            executor=executor_info,
            miner_hotkey=miner_info.miner_hotkey,
            ssh=shell.ssh_client,
            runner=runner,
            verified=verified_job_info,
            settings={"version": settings.VERSION},
            encrypt_key=encrypted_files.encrypt_key,
            default_extra=default_extra,
            services=ContextServices(
                ssh=self.ssh_service,
                redis=self.redis_service,
                collateral=self.collateral_contract_service,
                validation=self.validation_service,
                verifyx=self.verifyx_validation_service,
                connectivity=self.executor_connectivity_service,
                shell=shell,
                port_mapping=self.port_mapping_dao,
                score_calculator=calculate_scores,
            ),
            config=ContextConfig(
                executor_root=executor_info.root_dir,
                compute_rest_app_url=settings.COMPUTE_REST_API_URL,
                gpu_monitor_script_relative="src/gpus_utility.py",
                machine_scrape_filename=encrypted_files.machine_scrape_file_name,
                machine_scrape_timeout=JOB_LENGTH,
                obfuscation_keys=encrypted_files.all_keys,
                validator_keypair=keypair,
                max_gpu_count=MAX_GPU_COUNT,
                gpu_model_rates=GPU_MODEL_RATES,
                nvml_digest_map=LIB_NVIDIA_ML_DIGESTS,
                enable_no_collateral=settings.ENABLE_NO_COLLATERAL,
                verifyx_enabled=settings.ENABLE_VERIFYX,
                port_private_key=private_key,
                port_public_key=public_key,
                job_batch_id=miner_info.job_batch_id,
            ),
            state=ContextState(upload_local_dir=encrypted_files.tmp_directory),
            is_rental_succeed=is_rental_succeed,
        )

    @staticmethod
    def build_checks() -> list[Check]:
        """Build the standard validation check pipeline.

        Returns:
            Ordered list of validation checks to execute
        """
        return cast(
            list[Check],
            [
                StartGPUMonitorCheck(),
                UploadFilesCheck(),
                MachineSpecScrapeCheck(),
                GpuCountCheck(),
                GpuModelValidCheck(),
                NvmlDigestCheck(),
                SpecChangeCheck(),
                GpuFingerprintCheck(),
                BannedGpuCheck(),
                DuplicateExecutorCheck(),
                CollateralCheck(),
                # PortConnectivityCheck(),
                TenantEnforcementCheck(),
                GpuUsageCheck(),
                VerifyXCheck(),
                CapabilityCheck(),
                PortCountCheck(),
                ScoreCheck(),
                FinalizeCheck(),
            ],
        )

    def build_pipeline(self, checks: list[Check]) -> Pipeline:
        """Build a pipeline with the given checks.

        Args:
            checks: List of validation checks

        Returns:
            Configured Pipeline ready to run
        """
        return Pipeline(checks, sink=LoggerSink(logger))
