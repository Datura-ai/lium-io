import logging
from typing import Annotated

import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload

from core.config import settings
from core.utils import _m, get_extra_info
from daos.port_mapping_dao import PortMappingDao
from services.collateral_contract_service import CollateralContractService
from services.executor_connectivity_service import ExecutorConnectivityService
from services.interactive_shell_service import InteractiveShellService
from services.matrix_validation_service import ValidationService
from services.redis_service import RedisService
from services.ssh_service import SSHService
from services.verifyx_validation_service import VerifyXValidationService

from .models import JobResult
from .pipeline_factory import PipelineFactory
from .result_handler import ResultHandler

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
        port_mapping_dao: Annotated[PortMappingDao, Depends(PortMappingDao)],
    ):
        self.ssh_service = ssh_service
        self.redis_service = redis_service
        self.wallet = settings.get_bittensor_wallet()

        # Initialize pipeline factory with all required services
        self.pipeline_factory = PipelineFactory(
            ssh_service=ssh_service,
            redis_service=redis_service,
            validation_service=validation_service,
            verifyx_validation_service=verifyx_validation_service,
            collateral_contract_service=collateral_contract_service,
            executor_connectivity_service=executor_connectivity_service,
            port_mapping_dao=port_mapping_dao,
        )

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
                # Build validation context
                base_ctx = await self.pipeline_factory.build_context(
                    shell=shell,
                    miner_info=miner_info,
                    executor_info=executor_info,
                    keypair=keypair,
                    private_key=private_key,
                    public_key=public_key,
                    encrypted_files=encrypted_files,
                )

                # Build and run validation pipeline
                pipeline = self.pipeline_factory.build_pipeline()
                ok, events, last_context = await pipeline.run(base_ctx)

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
                    verified_job_info=base_ctx.verified,  # From original context
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
