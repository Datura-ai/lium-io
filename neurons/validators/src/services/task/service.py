import asyncio
import json
import logging
import random
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Optional, Union, cast

import asyncssh
import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload

from core.config import settings
from core.utils import _m, context, get_extra_info, StructuredMessage
from daos.port_mapping_dao import PortMappingDao
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason
from services.const import (
    GPU_MODEL_RATES,
    MAX_GPU_COUNT,
    UNRENTED_MULTIPLIER,
    LIB_NVIDIA_ML_DIGESTS,
    GPU_UTILIZATION_LIMIT,
    GPU_MEMORY_UTILIZATION_LIMIT,
    MIN_PORT_COUNT,
)
from services.executor_connectivity_service import ExecutorConnectivityService
from services.redis_service import (
    RedisService,
    DUPLICATED_MACHINE_SET,
    RENTAL_SUCCEED_MACHINE_SET,
    AVAILABLE_PORT_MAPS_PREFIX,
)
from services.ssh_service import SSHService
from services.interactive_shell_service import InteractiveShellService
from services.matrix_validation_service import ValidationService
from services.verifyx_validation_service import VerifyXValidationService
from services.collateral_contract_service import CollateralContractService
from services.file_encrypt_service import ORIGINAL_KEYS

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

logger = logging.getLogger(__name__)

JOB_LENGTH = 300
SCORE_PORTION_FOR_OLD_CONTRACT = 0

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

    async def is_script_running(
        self, ssh_client: asyncssh.SSHClientConnection, script_path: str
    ) -> bool:
        """
        Check if a specific script is running.

        Args:
            ssh_client: SSH client instance
            script_path: Full path to the script (e.g., '/root/app/gpus_utility.py')


        Returns:
            bool: True if script is running, False otherwise
        """
        try:
            result = await ssh_client.run(f'ps aux | grep "python.*{script_path}"', timeout=10)
            # Filter out the grep process itself
            processes = [line for line in result.stdout.splitlines() if "grep" not in line]

            logger.info(f"{script_path} running status: {bool(processes)}")
            return bool(processes)
        except Exception as e:
            logger.error(f"Error checking {script_path} status: {e}")
            return False

    async def start_script(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        script_path: str,
        command_args: dict,
        executor_info: ExecutorSSHInfo,
    ) -> bool:
        """
        Start a script with specified arguments.

        Args:
            ssh_client: SSH client instance
            script_path: Full path to the script (e.g., '/root/app/gpus_utility.py')
            command_args: Dictionary of argument names and values

        Returns:
            bool: True if script started successfully, False otherwise
        """
        try:
            # Build command string from arguments
            args_string = " ".join([f"--{key} {value}" for key, value in command_args.items()])
            await ssh_client.run("pip install aiohttp click pynvml psutil", timeout=30)
            command = (
                f"nohup {executor_info.python_path} {script_path} {args_string} > /dev/null 2>&1 & "
            )
            # Run the script
            result = await ssh_client.run(command, timeout=50, check=True)
            logger.info(f"Started {script_path}: {result}")
            return True
        except Exception as e:
            logger.error(f"Error starting script {script_path}: {e}", exc_info=True)
            return False

    def validate_docker_image_digests(self, docker_digests, docker_hub_digests):
        # Check if the list is empty
        if not docker_digests:
            return False

        # Get unique digests
        unique_digests = list({item["digest"] for item in docker_digests})

        # Check for duplicates
        if len(unique_digests) != len(docker_digests):
            return False

        # Check if any digest is invalid
        for digest in unique_digests:
            if digest not in docker_hub_digests.values():
                return False

        return True

    def check_fingerprints_changed(self, prev_uuids, gpu_uuids):
        try:
            if not prev_uuids:
                return False
            prev_uuids = sorted(prev_uuids.split(','))
            gpu_uuids = sorted(gpu_uuids.split(','))

            return ",".join(prev_uuids) != ",".join(gpu_uuids)
        except Exception as e:
            logger.error(f"Error checking fingerprints changed: {e}")
            return False

    async def check_banned_guids(self, guids: list[str]):
        banned_guids = await self.redis_service.get_banned_guids()
        return any(guid in banned_guids for guid in guids)

    async def get_available_port_count(
        self, miner_hotkey: str, executor_id: str
    ) -> int:
        """Get count_ports of available ports from DB, fallback to Redis if needed.

        Returns:
            Count of available ports
        """
        extra = {"miner_hotkey": miner_hotkey, "executor_id": executor_id}

        try:
            count_ports = await self.port_mapping_dao.get_successful_ports_count(executor_id)
            if count_ports >= MIN_PORT_COUNT:
                logger.info(_m(f"Retrieved {count_ports} ports count_ports from DB", extra=extra))
                return count_ports

            logger.warning(_m(f"only {count_ports} in DB, fallback to Redis", extra=extra))

        except Exception as e:
            logger.error(_m("DB error, fallback to Redis", extra={**extra, "error": str(e)}), exc_info=True)

        # Fallback to Redis
        port_map_key = f"{AVAILABLE_PORT_MAPS_PREFIX}:{miner_hotkey}:{executor_id}"
        port_maps_bytes = await self.redis_service.lrange(port_map_key)
        return len([tuple(map(int, pm.decode().split(","))) for pm in port_maps_bytes])

    async def check_pod_running(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        container_name: str,
        executor_info: ExecutorSSHInfo,
    ):
        # check container running or not
        is_pod_running = False

        command = f"/usr/bin/docker ps -q -f name={container_name}"
        result = await ssh_client.run(command)
        if result.stdout.strip():
            is_pod_running = True
        else:
            # # remove pod in redis
            # await self.redis_service.remove_rented_machine(executor_info)
            logger.error(
                _m(
                    "Pod not found, but redis is saying it's rented",
                    extra={
                        "container_name": container_name,
                        "executor_id": executor_info.uuid,
                        "address": executor_info.address,
                        "port": executor_info.port,
                    }
                )
            )

        # get ssh pub keys
        command = f"/usr/bin/docker exec -i {container_name} sh -c 'cat ~/.ssh/authorized_keys'"
        result = await ssh_client.run(command)
        if result.stdout.strip():
            return is_pod_running, result.stdout.strip().split('\n')

        return is_pod_running, []

    async def _handle_task_result(
        self,
        miner_info: MinerJobRequestPayload,
        executor_info: ExecutorSSHInfo,
        spec: dict | None,
        score: float,
        job_score: float,
        collateral_deposited: bool,
        log_text: object,
        verified_job_info: dict,
        success: bool = True,
        clear_verified_job_info: bool = False,
        gpu_model_count: str = '',
        gpu_uuids: str = '',
        sysbox_runtime: bool = False,
        ssh_pub_keys: list[str] | None = None,
        clear_verified_job_reason: ResetVerifiedJobReason = ResetVerifiedJobReason.DEFAULT,
    ):
        logger.info(_m("Handle task result: ", extra={
            "miner_hotkey": miner_info.miner_hotkey,
            "executor_id": executor_info.uuid,
            "success": success,
            "score": score,
            "job_score": job_score,
        }))
        if success:
            log_status = "info"
            logger.info(log_text)

            if gpu_model_count and gpu_uuids:
                await self.redis_service.set_verified_job_info(
                    miner_hotkey=miner_info.miner_hotkey,
                    executor_id=executor_info.uuid,
                    prev_info=verified_job_info,
                    success=True,
                    spec=gpu_model_count,
                    uuids=gpu_uuids,
                )
        else:
            log_status = "warning"
            logger.warning(log_text)

            if clear_verified_job_info:
                await self.redis_service.clear_verified_job_info(
                    miner_hotkey=miner_info.miner_hotkey,
                    executor_id=executor_info.uuid,
                    prev_info=verified_job_info,
                    reason=clear_verified_job_reason,
                )
            else:
                await self.redis_service.set_verified_job_info(
                    miner_hotkey=miner_info.miner_hotkey,
                    executor_id=executor_info.uuid,
                    prev_info=verified_job_info,
                    success=success,
                )

        gpu_model = None
        gpu_count = 0
        if gpu_model_count and ':' in gpu_model_count:
            gpu_model_count_info = gpu_model_count.split(':')
            gpu_model = gpu_model_count_info[0]
            gpu_count = int(gpu_model_count_info[1])

        return JobResult(
            spec=spec,
            executor_info=executor_info,
            score=score,
            job_score=job_score,
            collateral_deposited=collateral_deposited,
            job_batch_id=miner_info.job_batch_id,
            log_status=log_status,
            log_text=str(log_text),
            gpu_model=gpu_model,
            gpu_count=gpu_count,
            sysbox_runtime=sysbox_runtime,
            ssh_pub_keys=ssh_pub_keys,
        )

    def check_gpu_usage(
        self,
        gpu_details: list[dict],
        gpu_processes: list[dict],
        default_extra: dict,
        rented: bool = False,
    ) -> tuple[bool, str | None | StructuredMessage]:
        # check gpu usages
        for detail in gpu_details:
            gpu_utilization = detail.get("gpu_utilization", GPU_UTILIZATION_LIMIT)
            gpu_memory_utilization = detail.get("memory_utilization", GPU_MEMORY_UTILIZATION_LIMIT)
            if len(gpu_processes) > 0 and (gpu_utilization >= GPU_UTILIZATION_LIMIT or gpu_memory_utilization > GPU_MEMORY_UTILIZATION_LIMIT):
                log_text = _m(
                    "GPU busy outside validator" if not rented else "Tenant container does not own GPU",
                    extra=get_extra_info({
                        **default_extra,
                        "reason_code": "GPU_USAGE_HIGH" if not rented else "GPU_USAGE_OUTSIDE_TENANT",
                        "severity": "warning",
                        "what_we_saw": {
                            "gpu_utilization": f"{gpu_utilization}%",
                            "vram_utilization": f"{gpu_memory_utilization}%",
                            "process_count": len(gpu_processes),
                        },
                        "gpu_processes": gpu_processes,
                        "impact": "Validation skipped; score set to 0" if not rented else "Validation failed; score set to 0",
                        "remediation": (
                            "Stop all GPU processes and re-run your node. If using Docker, ensure no host processes are running."
                        ) if not rented else (
                            "Terminate host-level GPU processes, make sure nvidia-smi doesn't show any running processes."
                        ),
                    }),
                )
                return False, log_text

        return True, None

    def calc_scores(
        self,
        gpu_model: str,
        collateral_deposited: bool,
        is_rental_succeed: bool,
        contract_version: str,
        rented: bool = False,
        port_count: int = 0,
    ) -> Union[float, float, str]:
        warning_messages = []
        job_score = 1
        actual_score = 1
        
        def _return_value(actual_score, job_score, warning_messages):
            return actual_score, 1 if rented else job_score, (" WARNING: " + " | ".join(warning_messages)) if warning_messages else ""

        if not is_rental_succeed:
            actual_score = 0
            warning_messages.append("Score set to 0 pending rental verification")

        if port_count < MIN_PORT_COUNT and not rented:
            actual_score = 0
            job_score = 0
            warning_messages.append(f"Insufficient ports: {port_count} available, {MIN_PORT_COUNT} required")

        if gpu_model in settings.COLLATERAL_EXCLUDED_GPU_TYPES:
            return _return_value(actual_score, job_score, warning_messages)

        if not collateral_deposited:
            if settings.ENABLE_NO_COLLATERAL:
                warning_messages.append("No collateral deposited")
            else:
                actual_score = 0
                job_score = 0
                warning_messages.append("Collateral required but not deposited")
        elif contract_version and contract_version != settings.get_latest_contract_version() and not settings.ENABLE_NO_COLLATERAL:
            actual_score = actual_score * SCORE_PORTION_FOR_OLD_CONTRACT
            job_score = job_score * SCORE_PORTION_FOR_OLD_CONTRACT
            warning_messages.append(f"Outdated contract version (current: {contract_version}, latest: {settings.get_latest_contract_version()})")

        return _return_value(actual_score, job_score, warning_messages)

    async def create_task_old(
        self,
        miner_info: MinerJobRequestPayload,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
        public_key: str,
        encrypted_files: MinerJobEnryptedFiles,
    ):
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
        verified_job_info = await self.redis_service.get_verified_job_info(executor_info.uuid)
        prev_spec = verified_job_info.get('spec', '')
        prev_uuids = verified_job_info.get('uuids', '')

        is_rental_succeed = await self.redis_service.is_elem_exists_in_set(
            RENTAL_SUCCEED_MACHINE_SET, executor_info.uuid
        )

        try:
            logger.info(_m("Start job on an executor", extra=get_extra_info(default_extra)))

            private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)

            async with InteractiveShellService(
                host=executor_info.address,
                username=executor_info.ssh_username,
                private_key=private_key,
                port=executor_info.ssh_port,
            ) as shell:
                # start gpus_utility.py
                program_id = str(uuid.uuid4())
                command_args = {
                    "program_id": program_id,
                    "signature": f"0x{keypair.sign(program_id.encode()).hex()}",
                    "executor_id": executor_info.uuid,
                    "validator_hotkey": keypair.ss58_address,
                    "compute_rest_app_url": settings.COMPUTE_REST_API_URL,
                }
                script_path = f"{executor_info.root_dir}/src/gpus_utility.py"
                if not await self.is_script_running(shell.ssh_client, script_path):
                    await self.start_script(shell.ssh_client, script_path, command_args, executor_info)

                # upload temp directory
                random_length = random.randint(5, 15)
                remote_dir = f"{executor_info.root_dir}/{self.ssh_service.generate_random_string(length=random_length, string_only=True)}"
                await shell.upload_directory(encrypted_files.tmp_directory, remote_dir)

                remote_machine_scrape_file_path = (
                    f"{remote_dir}/{encrypted_files.machine_scrape_file_name}"
                )

                logger.info(
                    _m(
                        "Uploaded files to run job",
                        extra=get_extra_info(default_extra),
                    )
                )

                await shell.ssh_client.run(f"chmod +x {remote_machine_scrape_file_path}")
                machine_specs, _ = await self._run_task(
                    ssh_client=shell.ssh_client,
                    miner_hotkey=miner_info.miner_hotkey,
                    executor_info=executor_info,
                    command=f"{remote_machine_scrape_file_path}",
                )
                if not machine_specs:
                    raise Exception("No machine specs found")

                machine_spec = json.loads(
                    self.ssh_service.decrypt_payload(
                        encrypted_files.encrypt_key, machine_specs[0].strip()
                    )
                )

                # de-obfuscate machine_spec
                all_keys = encrypted_files.all_keys
                reverse_all_keys = {v: k for k, v in all_keys.items()}

                updated_machine_spec = self.update_keys(machine_spec, reverse_all_keys)
                updated_machine_spec = self.update_keys(updated_machine_spec, ORIGINAL_KEYS)

                machine_spec = {**updated_machine_spec}

                gpu_model = None
                if machine_spec.get("gpu", {}).get("count", 0) > 0:
                    details = machine_spec["gpu"].get("details", [])
                    if len(details) > 0:
                        gpu_model = details[0].get("name", None)

                gpu_count = machine_spec.get("gpu", {}).get("count", 0)
                gpu_details = machine_spec.get("gpu", {}).get("details", [])
                gpu_model_count = f'{gpu_model}:{gpu_count}'

                nvidia_driver = machine_spec.get("gpu", {}).get("driver", "")
                libnvidia_ml = machine_spec.get("md5_checksums", {}).get("libnvidia_ml", "")

                docker_version = machine_spec.get("docker", {}).get("version", "")
                docker_digest = machine_spec.get("md5_checksums", {}).get("docker", "")

                ram = machine_spec.get("ram", {}).get("total", 0)
                storage = machine_spec.get("hard_disk", {}).get("free", 0)

                gpu_processes = machine_spec.get("gpu_processes", [])

                sysbox_runtime = machine_spec.get("sysbox_runtime", False)
                vram = 0
                for detail in gpu_details:
                    vram += detail.get("capacity", 0) * 1024

                gpu_uuids = ','.join([detail.get('uuid', '') for detail in gpu_details])

                logger.info(
                    _m(
                        "Machine spec scraped",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                "gpu_model": gpu_model,
                                "gpu_count": gpu_count,
                                "nvidia_driver": nvidia_driver,
                                "libnvidia_ml": libnvidia_ml,
                                **verified_job_info,
                            }
                        ),
                    ),
                )

                collateral_deposited, collateral_contract_error_message, contract_version = await self.collateral_contract_service.is_eligible_executor(
                    miner_hotkey=miner_info.miner_hotkey,
                    executor_uuid=executor_info.uuid,
                    gpu_model=gpu_model,
                    gpu_count=gpu_count
                )
                default_extra = {
                    **default_extra,
                    "collateral_deposited": collateral_deposited,
                    "collateral_contract_error_message": collateral_contract_error_message,
                }

                if gpu_count > MAX_GPU_COUNT:
                    log_text = _m(
                        "GPU count exceeds policy",
                        extra=get_extra_info({
                            **default_extra,
                            "reason_code": "GPU_COUNT_EXCEEDS_MAX",
                            "severity": "error",
                            "what_we_saw": {
                                "count": gpu_count,
                                "max": MAX_GPU_COUNT,
                            },
                            "impact": "Score set to 0",
                            "remediation": "Reduce visible GPU count (e.g., CUDA_VISIBLE_DEVICES) to policy limit.",
                        }),
                    )

                    return await self._handle_task_result(
                        miner_info=miner_info,
                        executor_info=executor_info,
                        spec=machine_spec,
                        score=0,
                        job_score=0,
                        collateral_deposited=collateral_deposited,
                        log_text=log_text,
                        verified_job_info=verified_job_info,
                        success=False,
                        clear_verified_job_info=False,
                    )

                if not GPU_MODEL_RATES.get(gpu_model) or gpu_count == 0 or len(gpu_details) != gpu_count:
                    extra_info = {
                        **default_extra,
                        "os_version": machine_spec.get("os", ""),
                        "nvidia_cfg": machine_spec.get("nvidia_cfg", ""),
                        "docker_cfg": machine_spec.get("docker_cfg", ""),
                        "gpu_count": gpu_count,
                        "gpu_details_length": len(gpu_details),
                        "gpu_scrape_error": machine_spec.get("gpu_scrape_error", ""),
                        "nvidia_cfg_scrape_error": machine_spec.get("nvidia_cfg_scrape_error", ""),
                        "docker_cfg_scrape_error": machine_spec.get("docker_cfg_scrape_error", ""),
                    }
                    if gpu_model:
                        extra_info["gpu_model"] = gpu_model

                    extra_info["reason_code"] = "GPU_SCRAPE_EMPTY"
                    extra_info["severity"] = "warning"
                    extra_info["what_we_saw"] = {
                        "gpu_count": gpu_count,
                        "details_len": len(gpu_details),
                    }
                    extra_info["impact"] = "Job skipped; score set to 0"
                    extra_info["remediation"] = (
                        "1) Ensure `nvidia-smi` works on host. "
                        "2) Ensure Docker can see GPUs: "
                        "`docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi`. "
                        "3) Install and configure nvidia-container-toolkit."
                    )
                    extra_info["help_uri"] = "https://github.com/NVIDIA/nvidia-container-toolkit"

                    log_text = _m(
                        "No usable GPUs detected",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                **extra_info,
                            }
                        ),
                    )

                    return await self._handle_task_result(
                        miner_info=miner_info,
                        executor_info=executor_info,
                        spec=machine_spec,
                        score=0,
                        job_score=0,
                        collateral_deposited=collateral_deposited,
                        log_text=log_text,
                        verified_job_info=verified_job_info,
                        success=False,
                        clear_verified_job_info=False,
                    )

                if nvidia_driver and LIB_NVIDIA_ML_DIGESTS.get(nvidia_driver) != libnvidia_ml:
                    log_text = _m(
                        "NVML library digest mismatch",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                "reason_code": "NVML_DIGEST_MISMATCH",
                                "severity": "error",
                                "what_we_saw": {
                                    "driver": nvidia_driver,
                                    "expected_md5": LIB_NVIDIA_ML_DIGESTS.get(nvidia_driver),
                                    "actual_md5": libnvidia_ml,
                                },
                                "gpu_model": gpu_model,
                                "gpu_count": gpu_count,
                                "impact": "Score set to 0; previous verification cleared",
                                "remediation": "Reinstall NVIDIA driver for this version, ensure libnvidia-ml matches. Avoid LD_PRELOAD/overrides of NVML.",
                            }
                        ),
                    )

                    return await self._handle_task_result(
                        miner_info=miner_info,
                        executor_info=executor_info,
                        spec=machine_spec,
                        score=0,
                        job_score=0,
                        collateral_deposited=collateral_deposited,
                        log_text=log_text,
                        verified_job_info=verified_job_info,
                        success=False,
                        clear_verified_job_info=True,
                    )

                if prev_spec and prev_spec != gpu_model_count:
                    log_text = _m(
                        "GPU inventory changed",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                "reason_code": "SPEC_CHANGED",
                                "severity": "warning",
                                "what_we_saw": {
                                    "previous": prev_spec,
                                    "current": gpu_model_count,
                                },
                                "impact": "Verification reset; score set to 0",
                                "remediation": "Keep a stable GPU configuration between checks. If you hot-plugged GPUs or changed MIG, revert or re-verify.",
                            }
                        ),
                    )

                    return await self._handle_task_result(
                        miner_info=miner_info,
                        executor_info=executor_info,
                        spec=machine_spec,
                        score=0,
                        job_score=0,
                        collateral_deposited=collateral_deposited,
                        log_text=log_text,
                        verified_job_info=verified_job_info,
                        success=False,
                        clear_verified_job_info=True,
                    )

                if self.check_fingerprints_changed(prev_uuids, gpu_uuids):
                    log_text = _m(
                        "GPU fingerprints changed",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                "reason_code": "GPU_UUID_CHANGED",
                                "severity": "warning",
                                "what_we_saw": {
                                    "previous": prev_uuids,
                                    "current": gpu_uuids,
                                },
                                "impact": "Verification reset; score set to 0",
                                "remediation": "Ensure the same physical GPUs are attached. Power-cycling or different PCIe mapping can change order; ensure stable mapping.",
                            }
                        ),
                    )

                    return await self._handle_task_result(
                        miner_info=miner_info,
                        executor_info=executor_info,
                        spec=machine_spec,
                        score=0,
                        job_score=0,
                        collateral_deposited=collateral_deposited,
                        log_text=log_text,
                        verified_job_info=verified_job_info,
                        success=False,
                        clear_verified_job_info=True,
                    )

                if await self.check_banned_guids(gpu_uuids.split(',')):
                    log_text = _m(
                        "GPU model temporarily ineligible",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                "reason_code": "GPU_BANNED",
                                "severity": "warning",
                                "what_we_saw": {
                                    "gpu_uuids": gpu_uuids,
                                },
                                "impact": "Score set to 0; verification cleared",
                                "remediation": "Use eligible GPUs per current marketplace policy or wait for policy updates.",
                            }
                        ),
                    )

                    return await self._handle_task_result(
                        miner_info=miner_info,
                        executor_info=executor_info,
                        spec=machine_spec,
                        score=0,
                        job_score=0,
                        collateral_deposited=collateral_deposited,
                        log_text=log_text,
                        verified_job_info=verified_job_info,
                        success=False,
                        clear_verified_job_info=True,
                    )

                logger.info(
                    _m(
                        f"Got GPU Model: {gpu_model}, count: {gpu_count}",
                        extra=get_extra_info(default_extra),
                    ),
                )

                # check duplicated
                is_duplicated = await self.redis_service.is_elem_exists_in_set(
                    DUPLICATED_MACHINE_SET, f"{miner_info.miner_hotkey}:{executor_info.uuid}"
                )
                if is_duplicated:
                    log_text = _m(
                        "Duplicate executor registration",
                        extra=get_extra_info({
                            **default_extra,
                            "reason_code": "EXECUTOR_DUPLICATE",
                            "severity": "warning",
                            "what_we_saw": {
                                "executor_uuid": executor_info.uuid,
                                "hotkey": miner_info.miner_hotkey,
                            },
                            "impact": "Score set to 0; verification cleared",
                            "remediation": "Deregister duplicates or ensure each executor has a unique UUID and host identity.",
                        }),
                    )

                    return await self._handle_task_result(
                        miner_info=miner_info,
                        executor_info=executor_info,
                        spec=machine_spec,
                        score=0,
                        job_score=0,
                        collateral_deposited=collateral_deposited,
                        log_text=log_text,
                        verified_job_info=verified_job_info,
                        success=False,
                        clear_verified_job_info=True,
                    )

                # check rented status
                rented_machine = await self.redis_service.get_rented_machine(executor_info)
                if rented_machine and rented_machine.get("container_name", ""):
                    default_extra = {
                        **default_extra,
                        "rented": True,
                    }
                    container_name = rented_machine.get("container_name", "")
                    is_pod_running, ssh_pub_keys = await self.check_pod_running(
                        ssh_client=shell.ssh_client,
                        container_name=container_name,
                        executor_info=executor_info,
                    )
                    if not is_pod_running:
                        log_text = _m(
                            "Pod not running",
                            extra=get_extra_info(
                                {
                                    **default_extra,
                                    "reason_code": "POD_NOT_RUNNING",
                                    "severity": "error",
                                    "what_we_saw": {
                                        "container": container_name,
                                        "status": "not found in `docker ps`",
                                    },
                                    "impact": "Score set to 0; verification cleared",
                                    "remediation": f"Start the container and keep it healthy: `docker start {container_name}`. Investigate container exit logs: `docker logs --tail=200 {container_name}`.",
                                }
                            ),
                        )

                        return await self._handle_task_result(
                            miner_info=miner_info,
                            executor_info=executor_info,
                            spec=machine_spec,
                            score=0,
                            job_score=0,
                            collateral_deposited=collateral_deposited,
                            log_text=log_text,
                            verified_job_info=verified_job_info,
                            success=False,
                            gpu_model_count=gpu_model_count,
                            clear_verified_job_info=True,
                            clear_verified_job_reason=ResetVerifiedJobReason.POD_NOT_RUNNING,
                        )

                    # check gpu running out side of containers
                    gpu_running_outside = False
                    for process in gpu_processes:
                        gpu_process_container = process.get('container_name', None)
                        if not gpu_process_container or gpu_process_container != container_name:
                            gpu_running_outside = True
                            break

                    if not rented_machine.get("owner_flag", False) and gpu_running_outside:
                        # check gpu usages
                        is_gpu_usage_ok, log_text = self.check_gpu_usage(
                            gpu_details=gpu_details,
                            gpu_processes=gpu_processes,
                            default_extra=default_extra,
                            rented=True,
                        )
                        if not is_gpu_usage_ok:
                            return await self._handle_task_result(
                                miner_info=miner_info,
                                executor_info=executor_info,
                                spec=machine_spec,
                                score=0,
                                job_score=0,
                                collateral_deposited=collateral_deposited,
                                log_text=log_text,
                                verified_job_info=verified_job_info,
                                success=False,
                                gpu_model_count=gpu_model_count,
                                clear_verified_job_info=False,
                                ssh_pub_keys=ssh_pub_keys,
                            )

                    # get available port count from DB (fallback to Redis)
                    port_count = await self.get_available_port_count(
                        miner_info.miner_hotkey, executor_info.uuid
                    )
                    machine_spec = {
                        **machine_spec,
                        "available_port_count": port_count,
                    }

                    actual_score, job_score, warning_message = self.calc_scores(
                        gpu_model=gpu_model,
                        collateral_deposited=collateral_deposited,
                        is_rental_succeed=is_rental_succeed,
                        contract_version=contract_version,
                        rented=True,
                    )

                    log_text = _m(
                        "Executor already rented",
                        extra=get_extra_info({
                            **default_extra,
                            "reason_code": "RENTED",
                            "severity": "info",
                            "what_we_saw": {
                                "contract_version": contract_version,
                                "collateral": collateral_deposited,
                            },
                            "impact": f"Reported rented score={job_score} (actual={actual_score})",
                            "remediation": f"No action needed.{warning_message}",
                            "actual_score": actual_score,
                            "is_rental_succeed": is_rental_succeed,
                            "job_score": job_score,
                        }),
                    )

                    return await self._handle_task_result(
                        miner_info=miner_info,
                        executor_info=executor_info,
                        spec=machine_spec,
                        score=actual_score,
                        job_score=job_score,
                        collateral_deposited=collateral_deposited,
                        log_text=log_text,
                        verified_job_info=verified_job_info,
                        success=True,
                        gpu_model_count=gpu_model_count,
                        clear_verified_job_info=False,
                        sysbox_runtime=sysbox_runtime,
                        ssh_pub_keys=ssh_pub_keys,
                    )

                # check gpu usages
                is_gpu_usage_ok, log_text = self.check_gpu_usage(
                    gpu_details=gpu_details,
                    gpu_processes=gpu_processes,
                    default_extra=default_extra,
                    rented=False,
                )
                if not is_gpu_usage_ok:
                    return await self._handle_task_result(
                        miner_info=miner_info,
                        executor_info=executor_info,
                        spec=machine_spec,
                        score=0,
                        job_score=0,
                        collateral_deposited=collateral_deposited,
                        log_text=log_text,
                        verified_job_info=verified_job_info,
                        success=False,
                        gpu_model_count=gpu_model_count,
                        clear_verified_job_info=False,
                    )

                renting_in_progress = await self.redis_service.renting_in_progress(miner_info.miner_hotkey, executor_info.uuid)
                if not renting_in_progress and not rented_machine:
                    default_extra = {
                        **default_extra,
                        "renting_in_progress": True,
                    }
                    docker_connection_check_result = await self.executor_connectivity_service.verify_ports(
                        ssh_client=shell.ssh_client,
                        job_batch_id=miner_info.job_batch_id,
                        miner_hotkey=miner_info.miner_hotkey,
                        executor_info=executor_info,
                        private_key=private_key,
                        public_key=public_key,
                        sysbox_runtime=sysbox_runtime,
                    )

                    sysbox_runtime = docker_connection_check_result.sysbox_runtime
                    machine_spec = {
                        **machine_spec,
                        "sysbox_runtime": sysbox_runtime,
                    }
                    if not docker_connection_check_result.success:
                        return await self._handle_task_result(
                            miner_info=miner_info,
                            executor_info=executor_info,
                            spec=machine_spec,
                            score=0,
                            job_score=0,
                            collateral_deposited=collateral_deposited,
                            log_text=docker_connection_check_result.log_text,
                            verified_job_info=verified_job_info,
                            success=False,
                            gpu_model_count=gpu_model_count,
                            clear_verified_job_info=False,
                        )

                # docker_digests = machine_spec.get("docker", {}).get("containers", [])
                # is_docker_valid = self.validate_docker_image_digests(docker_digests, docker_hub_digests)
                # if not is_docker_valid:
                #     return await self.handle_task_failure(
                #         ssh_client, remote_dir, miner_info, executor_info, machine_spec,
                #         "Docker digests are not valid", verified_job_info, {**default_extra, "docker_digests": docker_digests}, True
                #     )

                if settings.ENABLE_VERIFYX:
                    verifyx_result = await self.verifyx_validation_service.validate_verifyx_and_process_job(
                        shell=shell, executor_info=executor_info,
                        default_extra=default_extra, machine_spec=machine_spec
                    )

                    if verifyx_result.data and verifyx_result.data.get("success"):
                        # Direct update on success
                        machine_spec.update({
                            "ram": verifyx_result.data.get("ram"),
                            "hard_disk": verifyx_result.data.get("hard_disk"),
                            "network": verifyx_result.data.get("network")
                        })
                        default_extra.update({
                            "verifyx_success": True,
                            "verifyx_data": verifyx_result.data
                        })
                    else:
                        error_msg = verifyx_result.error or (verifyx_result.data.get('errors') if verifyx_result.data else 'Unknown errors')
                        log_text = _m(
                            "VerifyX validation failed",
                            extra=get_extra_info({
                                **default_extra,
                                "reason_code": "VERIFYX_FAILED",
                                "severity": "error",
                                "what_we_saw": {
                                    "errors": error_msg,
                                },
                                "impact": "Score set to 0",
                                "remediation": "Run VerifyX locally to debug. Ensure network, disk and RAM probes pass within timeouts.",
                            })
                        )
                        return await self._handle_task_result(
                            miner_info=miner_info,
                            executor_info=executor_info,
                            spec=machine_spec,
                            score=0,
                            job_score=0,
                            collateral_deposited=collateral_deposited,
                            log_text=log_text,
                            verified_job_info=verified_job_info,
                            success=False,
                            gpu_model_count=gpu_model_count,
                            clear_verified_job_info=False,
                        )

                    logger.info(_m("Verifyx validation processed", extra=get_extra_info(default_extra)))

                is_valid = await self.validation_service.validate_gpu_model_and_process_job(
                    ssh_client=shell.ssh_client,
                    executor_info=executor_info,
                    default_extra=default_extra,
                    machine_spec=machine_spec
                )

                if not is_valid:
                    log_text = _m(
                        "GPU capability verification failed",
                        extra=get_extra_info({
                            **default_extra,
                            "reason_code": "GPU_VERIFY_FAILED",
                            "severity": "error",
                            "impact": "Score set to 0",
                            "remediation": "Run: `docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi` and ensure it succeeds; then retry validation.",
                        }),
                    )
                    return await self._handle_task_result(
                        miner_info=miner_info,
                        executor_info=executor_info,
                        spec=machine_spec,
                        score=0,
                        job_score=0,
                        collateral_deposited=collateral_deposited,
                        log_text=log_text,
                        verified_job_info=verified_job_info,
                        success=False,
                        gpu_model_count=gpu_model_count,
                        clear_verified_job_info=False,
                    )

                # get available port count from DB (fallback to Redis)
                port_count = await self.get_available_port_count(
                    miner_info.miner_hotkey, executor_info.uuid
                )
                machine_spec = {
                    **machine_spec,
                    "available_port_count": port_count,
                }

                actual_score, job_score, warning_message = self.calc_scores(
                    gpu_model=gpu_model,
                    collateral_deposited=collateral_deposited,
                    is_rental_succeed=is_rental_succeed,
                    contract_version=contract_version,
                    rented=False,
                    port_count=port_count,
                )

                success = True if actual_score > 0 else False

                log_text = _m(
                    "Validation task completed",
                    extra=get_extra_info(
                        {
                            **default_extra,
                            "reason_code": "VALIDATION_COMPLETED",
                            "severity": "info" if success else "warning",
                            "what_we_saw": {
                                "gpu_model": gpu_model,
                                "gpu_count": gpu_count,
                                "contract_version": contract_version,
                            },
                            "impact": f"Job score={job_score}, actual score={actual_score}",
                            "remediation": f"No action needed.{warning_message}" if success else f"Address issues:{warning_message}",
                            "job_score": job_score,
                            "actual_score": actual_score,
                            "is_rental_succeed": is_rental_succeed,
                            "unrented_multiplier": UNRENTED_MULTIPLIER,
                            "sysbox_runtime": sysbox_runtime,
                        }
                    ),
                )

                logger.debug(
                    _m(
                        "SSH connection closed for executor",
                        extra=get_extra_info(default_extra),
                    ),
                )

                return await self._handle_task_result(
                    miner_info=miner_info,
                    executor_info=executor_info,
                    spec=machine_spec,
                    score=actual_score,
                    job_score=job_score,
                    collateral_deposited=collateral_deposited,
                    log_text=log_text,
                    verified_job_info=verified_job_info,
                    success=success,
                    clear_verified_job_info=False,
                    gpu_model_count=gpu_model_count,
                    gpu_uuids=gpu_uuids,
                    sysbox_runtime=sysbox_runtime,
                )
        except Exception as e:
            log_status = "error"
            log_text = _m(
                "Task orchestration error",
                extra=get_extra_info({
                    **default_extra,
                    "reason_code": "TASK_ERROR",
                    "severity": "error",
                    "what_we_saw": {
                        "exception": str(e)[:200],
                    },
                    "impact": "Score set to 0",
                    "remediation": "Check SSH connectivity, file paths, and executor logs. Retry after fixing environment.",
                }),
            )

            try:
                await self.redis_service.set_verified_job_info(
                    miner_hotkey=miner_info.miner_hotkey,
                    executor_id=executor_info.uuid,
                    prev_info=verified_job_info,
                    success=False,
                )
            except:
                pass

            logger.error(
                log_text,
                exc_info=True,
            )

            return JobResult(
                spec=None,
                executor_info=executor_info,
                score=0,
                job_score=0,
                job_batch_id=miner_info.job_batch_id,
                log_status=log_status,
                log_text=str(log_text),
                gpu_model=None,
                gpu_count=0,
                sysbox_runtime=False,
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

                base_ctx = Context(
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
                        score_calculator=self.calc_scores,
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

                def _resolve_reason(reason_value: str | None) -> ResetVerifiedJobReason:
                    if not reason_value:
                        return ResetVerifiedJobReason.DEFAULT
                    try:
                        return ResetVerifiedJobReason(reason_value)
                    except ValueError:
                        return ResetVerifiedJobReason.DEFAULT

                # Determine log_text and success based on ok status
                # Always use the last event for structured logging (Grafana consumption)
                last_event = events[-1]
                log_text = _m(last_event.event, extra=last_event.model_dump())
                success = False if not ok else last_context.success

                # Single call to handle result
                return await self._handle_task_result(
                    miner_info=miner_info,
                    executor_info=executor_info,
                    spec=last_context.state.specs or None,
                    score=last_context.score,
                    job_score=last_context.job_score,
                    collateral_deposited=last_context.collateral_deposited,
                    log_text=log_text,
                    verified_job_info=verified_job_info,
                    success=success,
                    clear_verified_job_info=last_context.clear_verified_job_info,
                    gpu_model_count=last_context.state.gpu_model_count or "",
                    gpu_uuids=last_context.state.gpu_uuids or "",
                    sysbox_runtime=last_context.state.sysbox_runtime,
                    ssh_pub_keys=last_context.ssh_pub_keys,
                    clear_verified_job_reason=_resolve_reason(last_context.clear_verified_job_reason),
                )

        except Exception as e:
            logger.error(
                _m(
                    "Pipeline validation error",
                    extra=get_extra_info({
                        "job_batch_id": miner_info.job_batch_id,
                        "miner_hotkey": miner_info.miner_hotkey,
                        "executor_uuid": executor_info.uuid,
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

    async def _run_task(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo,
        command: str,
        timeout: int = JOB_LENGTH,
    ) -> tuple[list[str] | None, str | None]:
        try:
            executor_name = f"{executor_info.uuid}_{executor_info.address}_{executor_info.port}"
            default_extra = {
                "executor_uuid": executor_info.uuid,
                "executor_ip_address": executor_info.address,
                "executor_port": executor_info.port,
                "miner_hotkey": miner_hotkey,
                "command": command[:100] + ("..." if len(command) > 100 else ""),
                "version": settings.VERSION,
            }
            context.set(f"[_run_task][{executor_name}]")
            logger.info(
                _m(
                    "Running task for executor",
                    extra=default_extra,
                ),
            )
            result = await ssh_client.run(command, timeout=timeout)
            results = result.stdout.splitlines()
            errors = result.stderr.splitlines()

            actual_errors = [error for error in errors if "warning" not in error.lower()]

            if len(results) == 0 and len(actual_errors) > 0:
                logger.error(_m("Remote command failed", extra=get_extra_info({
                    **default_extra,
                    "reason_code": "COMMAND_EXEC_ERROR",
                    "severity": "error",
                    "what_we_saw": {
                        "stderr_lines": len(actual_errors),
                    },
                    "errors": actual_errors,
                    "impact": "Task aborted"
                })))
                return None, str(actual_errors)

            if len(results) == 0:
                logger.error(_m("Remote command failed", extra=get_extra_info({
                    **default_extra,
                    "reason_code": "COMMAND_EXEC_ERROR",
                    "severity": "error",
                    "what_we_saw": {
                        "result": "No output",
                    },
                    "impact": "Task aborted",
                })))
                return None, "No results"

            return results, None
        except Exception as e:
            logger.error(
                _m("Run task error to executor", extra=get_extra_info(default_extra)),
                exc_info=True,
            )

            return None, str(e)

    def update_keys(self, d, key_mapping):
        updated_dict = {}
        for key, value in d.items():
            # Get the original key using the reverse mapping
            original_key = key_mapping.get(key)  # Default to the same key if not found
            # Recursively update keys if the value is a dictionary
            if isinstance(value, dict):
                updated_dict[original_key] = self.update_keys(value, key_mapping)
            elif isinstance(value, list):
                updated_list = []
                for item in value:
                    if isinstance(item, dict):
                        updated_list.append(self.update_keys(item, key_mapping))
                    else:
                        updated_list.append(item)
                updated_dict[original_key] = updated_list
            else:
                updated_dict[original_key] = value
        return updated_dict


TaskServiceDep = Annotated[TaskService, Depends(TaskService)]
