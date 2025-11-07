import logging
import uuid
from typing import Annotated, cast

import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload

from core.config import settings
from core.utils import _m, get_extra_info
from daos.port_mapping_dao import PortMappingDao
from services.const import (
    GPU_MODEL_RATES,
    MAX_GPU_COUNT,
    LIB_NVIDIA_ML_DIGESTS,
)
from services.executor_connectivity_service import ExecutorConnectivityService
from services.redis_service import (
    RedisService,
    RENTAL_SUCCEED_MACHINE_SET,
)
from services.ssh_service import SSHService
from services.interactive_shell_service import InteractiveShellService
from services.matrix_validation_service import ValidationService
from services.verifyx_validation_service import VerifyXValidationService
from services.collateral_contract_service import CollateralContractService

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
    TenantEnforcementCheck,
    ScoreCheck,
    StartGPUMonitorCheck,
    SpecChangeCheck,
    UploadFilesCheck,
    VerifyXCheck,
)
from .models import JobResult
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
from .result_handler import ResultHandler

logger = logging.getLogger(__name__)

JOB_LENGTH = 300

class TaskService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
        validation_service: Annotated[ValidationService, Depends(ValidationService)],
        verifyx_validation_service: Annotated[VerifyXValidationService, Depends(VerifyXValidationService)],
        collateral_contract_service: Annotated[CollateralContractService, Depends(CollateralContractService)],
        executor_connectivity_service: Annotated[ExecutorConnectivityService, Depends(ExecutorConnectivityService)],
        port_mapping_dao: Annotated[PortMappingDao, Depends(PortMappingDao)],
    ):
        self.ssh_service = ssh_service
        self.redis_service = redis_service
        self.validation_service = validation_service
        self.verifyx_validation_service = verifyx_validation_service
        self.collateral_contract_service = collateral_contract_service
        self.wallet = settings.get_bittensor_wallet()

        self.executor_connectivity_service = executor_connectivity_service
        self.port_mapping_dao = port_mapping_dao

    async def create_task(
        self,
        miner_info: MinerJobRequestPayload,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
        public_key: str,
        encrypted_files: MinerJobEnryptedFiles,
    ):
        """New pipeline-based validation task implementation."""
        try:
            # Decrypt private key
            private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)

            async with InteractiveShellService(
                host=executor_info.address,
                username=executor_info.ssh_username,
                private_key=private_key,
                port=executor_info.ssh_port,
            ) as shell:

                runner = SSHCommandRunner(shell.ssh_client, max_retries=1)

                verified_job_info = await self.redis_service.get_verified_job_info(executor_info.uuid)

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

                base_ctx = Context(
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

                checks = cast(
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
                        TenantEnforcementCheck(),
                        GpuUsageCheck(),
                        PortConnectivityCheck(),
                        VerifyXCheck(),
                        CapabilityCheck(),
                        PortCountCheck(),
                        ScoreCheck(),
                        FinalizeCheck(),
                    ],
                )

                ok, events, last_context = await Pipeline(checks, sink=LoggerSink(logger)).run(base_ctx)

                # Determine log_text and success based on ok status
                # Always use the last event for structured logging (Grafana consumption)
                last_event = events[-1]
                log_text = _m(last_event.event, extra=last_event.model_dump())
                success = False if not ok else last_context.success

                # Handle result using ResultHandler
                result_handler = ResultHandler(self.redis_service)
                return await result_handler.handle_result(
                    context=last_context,
                    miner_info=miner_info,
                    executor_info=executor_info,
                    verified_job_info=verified_job_info,
                    log_text=str(log_text),
                    success=success,
                )

        except Exception as e:
            logger.error(
                _m(
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
                ),
                exc_info=True,
            )
            return JobResult(
                spec=None,
                executor_info=executor_info,
                score=0,
                job_score=0,
                collateral_deposited=False,
                job_batch_id=miner_info.job_batch_id,
                log_status="error",
                log_text=str(e),
                gpu_model=None,
                gpu_count=0,
                sysbox_runtime=False,
            )


TaskServiceDep = Annotated[TaskService, Depends(TaskService)]
