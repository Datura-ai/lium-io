"""Factory for building validation pipelines.

This module provides a clean interface for constructing the validation pipeline,
including context setup and check instantiation.
"""

import logging
import uuid
from typing import cast

import bittensor
from clients.backend_client import BackendClient
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.collateral_contract_service import CollateralContractService
from services.const import GPU_MODEL_RATES, LIB_NVIDIA_ML_DIGESTS, MAX_GPU_COUNT
from services.container_cleanup import ContainerCleanup
from services.default_docker_image_digest_service import DefaultDockerImageDigestService
from services.executor_connectivity_service import ExecutorConnectivityService
from services.inspector_validation_service import InspectorValidationService
from services.interactive_shell_service import InteractiveShellService
from services.matrix_validation_service import ValidationService
from services.redis_service import RENTAL_SUCCEED_MACHINE_SET, RedisService
from services.verifyx_validation_service import VerifyXValidationService

from core.config import settings
from services.ssh_service import SSHService

from .checks import (
    BannedGpuCheck,
    CachedTemplateVerificationCheck,
    CapabilityCheck,
    CollateralCheck,
    CustomBuildOrphanSweepCheck,
    DuplicateExecutorCheck,
    FinalizeCheck,
    GpuCountCheck,
    GpuFingerprintCheck,
    GpuModelValidCheck,
    GpuPowerLimitCheck,
    GpuUsageCheck,
    GpuVramPrecheck,
    InspectorRentedCheck,
    MachineSpecScrapeCheck,
    NvmlDigestCheck,
    PortConnectivityCheck,
    PortCountCheck,
    RentalVerificationCheck,
    ScoreCheck,
    SpecChangeCheck,
    StaleContainerCleanupCheck,
    StartGPUMonitorCheck,
    SysboxRequiredCheck,
    TdxHostCheck,
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

# DAH-2211: singleton for per-executor cadence tracking across pipeline cycles.
# `build_checks()` is a staticmethod called per cycle, so a fresh instance per
# call would lose the "last swept at" timestamps. The check itself is stateless
# wrt input data and only reads ctx, so a shared instance is safe.
_CUSTOM_BUILD_ORPHAN_SWEEP_SINGLETON = CustomBuildOrphanSweepCheck()


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
        backend_client: BackendClient,
        default_docker_image_digest_service: DefaultDockerImageDigestService,
    ):
        """Initialize pipeline factory with required services.

        Args:
            ssh_service: SSH service for remote operations
            redis_service: Redis service for state management
            validation_service: Matrix validation service
            verifyx_validation_service: VerifyX validation service
            collateral_contract_service: Collateral contract service
            executor_connectivity_service: Executor connectivity service
            backend_client: Backend API client
        """
        self.ssh_service = ssh_service
        self.redis_service = redis_service
        self.validation_service = validation_service
        self.verifyx_validation_service = verifyx_validation_service
        self.inspector_validation_service = InspectorValidationService()
        self.collateral_contract_service = collateral_contract_service
        self.executor_connectivity_service = executor_connectivity_service
        self.backend_client = backend_client
        self.default_docker_image_digest_service = default_docker_image_digest_service

        dry_run = settings.DRY_RUN or settings.CONTAINER_CLEANUP_DRY_RUN
        logger.info(f"ContainerCleanup dry_run={dry_run}")
        self.container_cleanup = ContainerCleanup(dry_run=dry_run)

    async def build_context(
        self,
        shell: InteractiveShellService,
        miner_info: MinerJobRequestPayload,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
        public_key: str,
        encrypted_files: MinerJobEnryptedFiles,
        rented_data: RentedExecutorsResponse,
        tdx_attestation_passed: bool = False,
        gpu_attestation_passed: bool | None = None,
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
            tdx_attestation_passed: Whether TDX attestation passed
            gpu_attestation_passed: NVIDIA CC GPU attestation outcome (None = not performed)

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
            miner_address=miner_info.miner_address,
            miner_port=miner_info.miner_port,
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
                inspector=self.inspector_validation_service,
                connectivity=self.executor_connectivity_service,
                shell=shell,
                score_calculator=calculate_scores,
                backend=self.backend_client,
                default_docker_image_digests=self.default_docker_image_digest_service,
                container_cleanup=self.container_cleanup,
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
                inspector_enabled=settings.ENABLE_INSPECTOR,
                port_private_key=private_key,
                port_public_key=public_key,
                job_batch_id=miner_info.job_batch_id,
            ),
            state=ContextState(
                upload_local_dir=encrypted_files.tmp_directory,
                rented_data=rented_data,
            ),
            is_rental_succeed=is_rental_succeed,
            tdx_attestation_passed=tdx_attestation_passed,
            gpu_attestation_passed=gpu_attestation_passed,
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
                # Pure-data model<->VRAM gate. No SSH/GPU dependency, so it runs
                # here — before the rented short-circuit (TenantEnforcementCheck)
                # — to gate rented and idle executors alike.
                GpuVramPrecheck(),
                GpuPowerLimitCheck(),
                NvmlDigestCheck(),
                SpecChangeCheck(),
                GpuFingerprintCheck(),
                BannedGpuCheck(),
                DuplicateExecutorCheck(),
                CollateralCheck(),
                # Reap orphaned (non-rented) rental containers BEFORE the port checks.
                # A pod container that outlives its rental (e.g. BROKEN_BY_PROVIDER, which the
                # platform deliberately does not tear down) keeps binding the rental port range.
                # Cleaning it here frees those ports so PortConnectivityCheck can bind them this
                # cycle; otherwise PortCountCheck (fatal) halts the pipeline before the cleanup
                # that used to live in TenantEnforcementCheck ever runs -> executor stuck at 0.
                StaleContainerCleanupCheck(),
                # DAH-2211 Phase 3.4(ii): orphan sweep for `lium-build-*`
                # image/scratch artifacts left behind by validator crashes or
                # aborted releases. Internally throttled to once-per-6h per
                # executor, so this is cheap to run every cycle. Singleton
                # instance so cadence state persists across `build_checks()`
                # calls.
                _CUSTOM_BUILD_ORPHAN_SWEEP_SINGLETON,
                PortConnectivityCheck(),
                PortCountCheck(),
                # DAH-2313: require sysbox before an unrented executor is allowed on the network.
                # Runs after PortConnectivityCheck, which overwrites ctx.state.sysbox_runtime with
                # the authoritative probe result used for scoring (not the earlier scrape hint).
                SysboxRequiredCheck(),
                InspectorRentedCheck(),
                TenantEnforcementCheck(),
                GpuUsageCheck(),
                VerifyXCheck(),
                TdxHostCheck(),
                CapabilityCheck(),
                # DAH-2265 Plan 2: advisory, non-fatal — observes whether the executor has
                # the recommended default image pre-pulled (DOCKER_PULL no-op). Runs here,
                # after specs/gpu_model/driver are populated and the GPU is validated, on the
                # idle valid-executor population. No scoring impact; fails open on any error.
                CachedTemplateVerificationCheck(),
                RentalVerificationCheck(),
                ScoreCheck(),
                FinalizeCheck(),
            ],
        )

    @staticmethod
    def build_dry_run_checks() -> list[Check]:
        """Build a dry run validation pipeline that excludes state-changing checks.

        This pipeline skips checks that would modify executor state or interfere with
        production operations:
        - StartGPUMonitorCheck: Would start processes on the executor
        - VerifyXCheck: Would use GPU to run some tests and interfere with prod validation


        Returns:
            Ordered list of validation checks safe for dry run mode
        """
        return cast(
            list[Check],
            [
                # StartGPUMonitorCheck(),  # SKIP: Starts processes on executor
                UploadFilesCheck(),
                MachineSpecScrapeCheck(),
                GpuCountCheck(),
                GpuModelValidCheck(),
                GpuVramPrecheck(),
                GpuPowerLimitCheck(),
                NvmlDigestCheck(),
                SpecChangeCheck(),
                GpuFingerprintCheck(),
                BannedGpuCheck(),
                DuplicateExecutorCheck(),
                CollateralCheck(),
                # StaleContainerCleanupCheck(),  # SKIP: removes containers on the executor
                PortConnectivityCheck(),
                PortCountCheck(),
                # DAH-2313: require sysbox before an unrented executor is allowed on the network.
                # Runs after PortConnectivityCheck, which overwrites ctx.state.sysbox_runtime with
                # the authoritative probe result used for scoring (not the earlier scrape hint).
                SysboxRequiredCheck(),
                TenantEnforcementCheck(),
                GpuUsageCheck(),
                # VerifyXCheck(),
                TdxHostCheck(),
                CapabilityCheck(),
                RentalVerificationCheck(),
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
