import asyncio
import json
import os

from daos.port_mapping_dao import PortMappingDao
from payload_models.payloads import MinerJobRequestPayload
from clients.backend_client import BackendClient

from core.config import settings
from incentive.factory import IncentiveFactory
from core.utils import _m, get_extra_info, get_logger
from clients.subtensor_client import SubtensorClient
from services.docker_service import DockerService
from services.executor_connectivity.container_runner import ContainerRunner
from services.executor_connectivity.dind_probe import DindProbe, DindVerifier
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import PortSelector
from services.executor_connectivity.port_tester import PortTester
from services.executor_connectivity.port_verifiers import BatchVerifier, FallbackVerifier
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.persister import PortResultPersister
from services.executor_connectivity_service import ExecutorConnectivityService
from services.file_encrypt_service import FileEncryptService
from services.miner_service import MinerService
from services.redis_service import PENDING_PODS_PREFIX, RedisService
from services.ssh_service import SSHService
from services.task_service import TaskService, JobResult
from services.matrix_validation_service import ValidationService
from services.verifyx_validation_service import VerifyXValidationService
from services.collateral_contract_service import CollateralContractService
from services.attestation_service import AttestationService
from services.const import IS_NOT_DEPOSITED_SCORE_MULTIPLIER


logger = get_logger(__name__)

SYNC_CYCLE = 12
WEIGHT_MAX_COUNTER = 6
MINER_SCORES_KEY = "miner_scores"


class Validator:
    def __init__(self):

        self.should_exit = False
        self.last_job_run_blocks = 0
        self.default_extra = {}

        self.miner_scores = {}

        # set incentive algorithm from setting
        self.incentive = settings.incentive

    async def initiate_services(self):
        # initiate subtensor client
        self.subtensor_client = await SubtensorClient.initialize()

        ssh_service = SSHService()
        self.redis_service = RedisService()
        self.file_encrypt_service = FileEncryptService(ssh_service=ssh_service)
        self.validation_service = ValidationService()
        self.verifyx_validation_service = VerifyXValidationService()
        self.collateral_contract_service = CollateralContractService()
        self.attestation_service = AttestationService()
        self.port_mapping_dao = PortMappingDao()

        # Backend client for API requests
        keypair = settings.get_bittensor_wallet().get_hotkey()
        self.backend_client = BackendClient(
            base_url=settings.COMPUTE_REST_API_URL or "",
            keypair=keypair,
        )

        port_tester = PortTester()
        runner = ContainerRunner()
        self.executor_connectivity_service = ExecutorConnectivityService(
            orchestrator=ConnectivityOrchestrator(
                PortSelector(),
                PortProbe(
                    BatchVerifier(port_tester, runner),
                    FallbackVerifier(port_tester, runner),
                ),
                DindProbe(DindVerifier(ssh_service)),
            ),
            persister=PortResultPersister(self.port_mapping_dao),
        )

        task_service = TaskService(
            ssh_service=ssh_service,
            redis_service=self.redis_service,
            validation_service=self.validation_service,
            verifyx_validation_service=self.verifyx_validation_service,
            collateral_contract_service=self.collateral_contract_service,
            executor_connectivity_service=self.executor_connectivity_service,
            port_mapping_dao=self.port_mapping_dao,
            backend_client=self.backend_client,
            attestation_service=self.attestation_service,
        )
        self.docker_service = DockerService(
            ssh_service=ssh_service,
            redis_service=self.redis_service,
            port_mapping_dao=self.port_mapping_dao,
            attestation_service=self.attestation_service,
        )
        self.miner_service = MinerService(
            ssh_service=ssh_service,
            task_service=task_service,
            redis_service=self.redis_service,
            port_mapping_dao=self.port_mapping_dao,
            attestation_service=self.attestation_service,
        )

        # init miner_scores
        try:
            if await self.subtensor_client.should_set_weights():
                self.miner_scores = {}
            else:
                miner_scores_json = await self.redis_service.get(MINER_SCORES_KEY)
                if miner_scores_json is None:
                    logger.info(
                        _m(
                            "[initiate_services] No data found in Redis for MINER_SCORES_KEY, initializing empty miner_scores.",
                            extra=get_extra_info(self.default_extra),
                        ),
                    )
                    self.miner_scores = {}
                else:
                    self.miner_scores = json.loads(miner_scores_json)

            # remove pod renting-in-progress status
            await self.redis_service.delete(PENDING_PODS_PREFIX)
        except Exception as e:
            logger.error(
                _m(
                    "[initiate_services] Failed to initialize miner_scores",
                    extra=get_extra_info({**self.default_extra, "error": str(e)}),
                ),
            )
            self.miner_scores = {}

        logger.info(
            _m(
                "[initiate_services] miner scores",
                extra=get_extra_info(
                    {
                        **self.default_extra,
                        **self.miner_scores,
                    }
                ),
            ),
        )

    async def sync(self):
        try:
            logger.info(
                _m(
                    "[sync] Syncing at subtensor",
                    extra=get_extra_info(self.default_extra),
                ),
            )

            # fetch miners
            miners = await self.subtensor_client.get_miners()

            try:
                if await self.subtensor_client.should_set_weights():
                    if settings.DRY_RUN:
                        logger.info(
                            _m(
                                "[sync] DRY_RUN: Skipping set_weights to Bittensor",
                                extra=get_extra_info({**self.default_extra, "miner_scores": self.miner_scores}),
                            )
                        )
                    else:
                        await self.subtensor_client.set_weights(miner_scores=self.miner_scores)
                    self.miner_scores = {}
            except Exception as e:
                logger.error(
                    _m(
                        "[sync] Error setting weights",
                        extra=get_extra_info(
                            {
                                **self.default_extra,
                                "error": str(e),
                            }
                        ),
                    ),
                )

            current_block = self.subtensor_client.get_current_block()
            logger.info(
                _m(
                    "[sync] Current block",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "current_block": current_block,
                        }
                    ),
                ),
            )

            if current_block - self.last_job_run_blocks >= settings.BLOCKS_FOR_JOB:
                job_block = (current_block // settings.BLOCKS_FOR_JOB) * settings.BLOCKS_FOR_JOB
                job_batch_id = await self.subtensor_client.get_time_from_block(job_block)

                # Fetch all rented executors from backend API
                rented_executors = await self.backend_client.get_all_rented_executors()
                if rented_executors is None:
                    logger.error(
                        _m(
                            "[sync] Failed to fetch rented executors, skipping this iteration",
                            extra=get_extra_info(self.default_extra),
                        ),
                    )
                    return

                logger.info(
                    _m(
                        "[sync] Send jobs to miners",
                        extra=get_extra_info(
                            {
                                **self.default_extra,
                                "miners": len(miners),
                                "current_block": current_block,
                                "job_batch_id": job_batch_id,
                            }
                        ),
                    ),
                )

                self.last_job_run_blocks = current_block

                encrypted_files = self.file_encrypt_service.ecrypt_miner_job_files()

                task_info = {}

                # request jobs
                jobs = [
                    asyncio.create_task(
                        self.miner_service.request_job_to_miner(
                            payload=MinerJobRequestPayload(
                                job_batch_id=job_batch_id,
                                miner_hotkey=miner.hotkey,
                                miner_coldkey=miner.coldkey,
                                miner_address=miner.axon_info.ip,
                                miner_port=miner.axon_info.port,
                            ),
                            encrypted_files=encrypted_files,
                            rented_data=rented_executors,
                        )
                    )
                    for miner in miners
                ]

                for miner, job in zip(miners, jobs):
                    task_info[job] = {
                        "miner_hotkey": miner.hotkey,
                        "miner_address": miner.axon_info.ip,
                        "miner_port": miner.axon_info.port,
                        "job_batch_id": job_batch_id,
                    }

                try:
                    total_gpu_model_count_map = {}
                    all_job_results = {}
                    miner_coldkeys = {}

                    # Run all jobs with asyncio.wait and set a timeout
                    done, pending = await asyncio.wait(jobs, timeout=settings.JOB_TIME_OUT - 50)

                    # Process completed jobs
                    for task in done:
                        try:
                            result = task.result()
                            if result:
                                miner_hotkey = result.get("miner_hotkey")
                                job_results: list[JobResult] = result.get("results", [])
                                miner_coldkey = result.get("miner_coldkey")

                                logger.info(
                                    _m(
                                        "[sync] Job_Result",
                                        extra=get_extra_info(
                                            {
                                                **self.default_extra,
                                                "miner_hotkey": miner_hotkey,
                                                "results": len(job_results)
                                            }
                                        ),
                                    ),
                                )

                                all_job_results[miner_hotkey] = job_results
                                miner_coldkeys[miner_hotkey] = miner_coldkey

                                for job_result in job_results:
                                    if job_result.gpu_model and job_result.gpu_count and (job_result.score > 0 or job_result.job_score > 0):
                                        total_gpu_model_count_map[job_result.gpu_model] = total_gpu_model_count_map.get(job_result.gpu_model, 0) + job_result.gpu_count

                            else:
                                info = task_info.get(task, {})
                                miner_hotkey = info.get("miner_hotkey", "unknown")
                                job_batch_id = info.get("job_batch_id", "unknown")
                                logger.error(
                                    _m(
                                        "[sync] No_Job_Result",
                                        extra=get_extra_info(
                                            {
                                                **self.default_extra,
                                                "miner_hotkey": miner_hotkey,
                                                "job_batch_id": job_batch_id,
                                            }
                                        ),
                                    ),
                                )

                        except Exception as e:
                            logger.error(
                                _m(
                                    "[sync] Error processing job result",
                                    extra=get_extra_info(
                                        {
                                            **self.default_extra,
                                            "job_batch_id": job_batch_id,
                                            "error": str(e),
                                        }
                                    ),
                                ),
                            )

                    # Handle pending jobs (those that did not complete within the timeout)
                    if pending:
                        for task in pending:
                            info = task_info.get(task, {})
                            miner_hotkey = info.get("miner_hotkey", "unknown")
                            job_batch_id = info.get("job_batch_id", "unknown")

                            logger.error(
                                _m(
                                    "[sync] Job_Timeout",
                                    extra=get_extra_info(
                                        {
                                            **self.default_extra,
                                            "miner_hotkey": miner_hotkey,
                                            "job_batch_id": job_batch_id,
                                        }
                                    ),
                                ),
                            )
                            task.cancel()

                    try:
                        open_fd_count = len(os.listdir('/proc/self/fd'))
                    except FileNotFoundError:
                        open_fd_count = -1
                        logger.error("Failed to get open file descriptor count. Defaulting to -1.")

                    logger.info(
                        _m(
                            "Total GPU model count",
                            extra=get_extra_info(
                                {
                                    **self.default_extra,
                                    "job_batch_id": job_batch_id,
                                    "data": total_gpu_model_count_map,
                                }
                            ),
                        ),
                    )

                    incentive = IncentiveFactory.create(
                        config=self.incentive,
                        redis_service=self.redis_service,
                        # pass results from synthetic job
                        jobs_results=all_job_results,
                        total_gpu_model_count_map=total_gpu_model_count_map,
                    )
                    await incentive.calculate_mining_scores()

                    # PHASE 2: Calculate final weights with burning logic per cycle
                    miners = await self.subtensor_client.get_miners()
                    last_mechanism_step_block = self.subtensor_client.get_last_mechansim_step_block()

                    cycle_scores = await incentive.calculate_final_weights(
                        miners=miners,
                        last_mechanism_step_block=last_mechanism_step_block,
                    )

                    # PHASE 3: Accumulate scores with burning applied
                    for miner_hotkey, score in cycle_scores.items():
                        self.miner_scores[miner_hotkey] = self.miner_scores.get(miner_hotkey, 0) + score

                    # Publish machine specs
                    for miner_hotkey, results in incentive.job_results.items():
                        miner_coldkey = miner_coldkeys.get(miner_hotkey)
                        if miner_coldkey:
                            await self.miner_service.publish_machine_specs(results, miner_hotkey, miner_coldkey)

                    logger.info(
                        _m(
                            "[sync] All Jobs finished",
                            extra=get_extra_info(
                                {
                                    **self.default_extra,
                                    "job_batch_id": job_batch_id,
                                    "cycle_scores": cycle_scores,
                                    "accumulated_miner_scores": self.miner_scores,
                                    "open_fd_count": open_fd_count,
                                    "total_executors": incentive.total_executors,
                                    "successful_executors": incentive.successful_executors,
                                    "failed_executors": incentive.failed_executors,
                                }
                            ),
                        ),
                    )

                except Exception as e:
                    logger.error(
                        _m(
                            "[sync] Unexpected error",
                            extra=get_extra_info(
                                {
                                    **self.default_extra,
                                    "job_batch_id": job_batch_id,
                                    "error": str(e),
                                }
                            ),
                        ),
                        exc_info=True,
                    )
            else:
                remaining_blocks = (
                    current_block // settings.BLOCKS_FOR_JOB + 1
                ) * settings.BLOCKS_FOR_JOB - current_block

                logger.info(
                    _m(
                        "[sync] Remaining blocks for next job",
                        extra=get_extra_info(
                            {
                                **self.default_extra,
                                "remaining_blocks": remaining_blocks,
                                "last_job_run_blocks": self.last_job_run_blocks,
                                "current_block": current_block,
                            }
                        ),
                    ),
                )
        except Exception as e:
            logger.error(
                _m(
                    "[sync] Unknown error",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "error": str(e),
                        }
                    ),
                ),
                exc_info=True,
            )

    async def start(self):
        logger.info(
            _m(
                "[start] Starting Validator in background",
                extra=get_extra_info({**self.default_extra, "dry_run": settings.DRY_RUN}),
            ),
        )
        try:
            await self.initiate_services()
            self.should_exit = False

            while not self.should_exit:
                await self.sync()
                self.should_exit = settings.DRY_RUN
                if not settings.DRY_RUN:
                    await asyncio.sleep(SYNC_CYCLE)

        except KeyboardInterrupt:
            logger.info(
                _m(
                    "[start] Validator killed by keyboard interrupt",
                    extra=get_extra_info(self.default_extra),
                ),
            )
            exit()
        except Exception as e:
            logger.info(
                _m(
                    "[start] Unknown error",
                    extra=get_extra_info({**self.default_extra, "error": str(e)}),
                ),
            )

    async def stop(self):
        logger.info(
            _m(
                "[stop] Stopping Validator process",
                extra=get_extra_info(self.default_extra),
            ),
        )

        self.should_exit = True

        # Cleanup subtensor client
        if hasattr(self, 'subtensor_client'):
            await SubtensorClient.shutdown()

        try:
            await self.redis_service.set(MINER_SCORES_KEY, json.dumps(self.miner_scores))
        except Exception as e:
            logger.info(
                _m(
                    "[stop] Failed to save miner_scores",
                    extra=get_extra_info({**self.default_extra, "error": str(e)}),
                ),
            )
