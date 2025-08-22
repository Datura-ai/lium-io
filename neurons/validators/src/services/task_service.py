import asyncio
import json
import logging
import random
import uuid
from typing import Annotated
from pydantic import BaseModel

import asyncssh
import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload

from core.config import settings
from core.utils import _m, context, get_extra_info
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason
from services.const import (
    GPU_MODEL_RATES,
    MAX_GPU_COUNT,
    UNRENTED_MULTIPLIER,
    LIB_NVIDIA_ML_DIGESTS,
    DOCKER_DIGEST,
    PYTHON_DIGEST,
    GPU_UTILIZATION_LIMIT,
    GPU_MEMORY_UTILIZATION_LIMIT,
    MIN_PORT_COUNT,
    BATCH_PORT_VERIFICATION_SIZE,
    DOCKER_DIND_IMAGE,
)
from services.redis_service import (
    RedisService,
    DUPLICATED_MACHINE_SET,
    RENTAL_SUCCEED_MACHINE_SET,
    AVAILABLE_PORT_MAPS_PREFIX,
)
from services.ssh_service import SSHService
from services.interactive_shell_service import InteractiveShellService
from services.matrix_validation_service import ValidationService
from services.collateral_contract_service import CollateralContractService
from services.file_encrypt_service import ORIGINAL_KEYS

logger = logging.getLogger(__name__)

JOB_LENGTH = 300


class JobResult(BaseModel):
    spec: dict | None = None
    executor_info: ExecutorSSHInfo
    score: float
    job_score: float
    collateral_deposited: bool = False
    job_batch_id: str
    log_status: str
    log_text: str
    gpu_model: str | None = None
    gpu_count: int = 0
    sysbox_runtime: bool = False
    ssh_pub_keys: list[str] | None = None


class DockerConnectionCheckResult(BaseModel):
    success: bool
    log_text: str | None = None
    sysbox_runtime: bool


class TaskService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
        validation_service: Annotated[ValidationService, Depends(ValidationService)],
        collateral_contract_service: Annotated[CollateralContractService, Depends(CollateralContractService)],
    ):
        self.ssh_service = ssh_service
        self.redis_service = redis_service
        self.validation_service = validation_service
        self.collateral_contract_service = collateral_contract_service
        self.wallet = settings.get_bittensor_wallet()

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

    def get_available_port_maps(
        self,
        executor_info: ExecutorSSHInfo,
        batch_size: int = 3,
    ) -> list[tuple[int, int]]:
        """Get a list of available port maps for batch verification."""
        if executor_info.port_mappings:
            port_mappings: list[tuple[int, int]] = json.loads(executor_info.port_mappings)
            port_mappings = [
                (internal_port, external_port)
                for internal_port, external_port in port_mappings
                if internal_port != executor_info.ssh_port
                and external_port != executor_info.ssh_port
            ]

            # Return up to batch_size port mappings
            return random.sample(port_mappings, min(batch_size, len(port_mappings)))

        # Generate ports from range
        if executor_info.port_range:
            if "-" in executor_info.port_range:
                min_port, max_port = map(
                    int, (part.strip() for part in executor_info.port_range.split("-"))
                )
                ports = list(range(min_port, max_port + 1))
            else:
                ports = list(
                    map(int, (part.strip() for part in executor_info.port_range.split(",")))
                )
        else:
            # Default range if port_range is empty
            ports = list(range(40000, 65536))

        ports = [port for port in ports if port != executor_info.ssh_port]

        if not ports:
            return []

        # Select random ports for batch verification
        selected_ports = random.sample(ports, min(batch_size, len(ports)))
        return [(port, port) for port in selected_ports]

    async def batch_verify_ports(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        job_batch_id: str,
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo,
        private_key: str,
        public_key: str,
        sysbox_runtime: bool = False,
    ) -> DockerConnectionCheckResult:
        """Verify multiple ports concurrently."""
        default_extra = {
            "job_batch_id": job_batch_id,
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_info.uuid,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "ssh_username": executor_info.ssh_username,
            "ssh_port": executor_info.ssh_port,
            "version": settings.VERSION,
            "sysbox_runtime": sysbox_runtime,
        }
        try:
            # remove all containers that has conatiner_ prefix in its name, since it's unrented
            command = '/usr/bin/docker ps -a --filter "name=^/container_" --format "{{.Names}}"'
            result = await ssh_client.run(command)
            if result.stdout.strip():
                container_names = " ".join(result.stdout.strip().split("\n"))

                logger.info(
                    _m(
                        "Cleaning existing docker containers",
                        extra=get_extra_info({
                            **default_extra,
                            "container_names": container_names,
                        }),
                    ),
                )

                command = f'/usr/bin/docker rm {container_names} -f'
                await ssh_client.run(command)

                command = f'/usr/bin/docker volume prune -af'
                await ssh_client.run(command)

            port_maps = self.get_available_port_maps(executor_info, BATCH_PORT_VERIFICATION_SIZE)
            if not port_maps:
                return DockerConnectionCheckResult(
                    success=False,
                    log_text="No port available for docker container",
                    sysbox_runtime=sysbox_runtime,
                )

            log_text = _m(
                "Verifying multiple ports",
                extra=get_extra_info({
                    **default_extra,
                    "port_maps": port_maps,
                }),
            )
            logger.info(log_text)

            tasks = [
                self.verify_single_port(
                    ssh_client, job_batch_id, miner_hotkey, executor_info,
                    private_key, public_key, internal_port, external_port, sysbox_runtime
                )
                for internal_port, external_port in port_maps
            ]

            # Execute all port verification tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter successful port verifications
            successful_ports = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Port verification failed for {port_maps[i]}: {result}")
                    continue

                if result.success:  # result is True if port verification succeeded
                    successful_ports.append(port_maps[i])
                    sysbox_runtime = result.sysbox_runtime

            if not successful_ports:
                return DockerConnectionCheckResult(
                    success=False,
                    log_text="No port available for docker container",
                    sysbox_runtime=sysbox_runtime,
                )

            # set port on redis
            key = f"{AVAILABLE_PORT_MAPS_PREFIX}:{miner_hotkey}:{executor_info.uuid}"
            for internal_port, external_port in successful_ports:
                port_map = f"{internal_port},{external_port}"

                # delete all the same port_maps in the list
                await self.redis_service.lrem(key=key, element=port_map)

                # insert port_map in the list
                await self.redis_service.lpush(key, port_map)

                # keep the latest 10 port maps
                port_maps = await self.redis_service.lrange(key)
                if len(port_maps) > 10:
                    await self.redis_service.rpop(key)

            return DockerConnectionCheckResult(
                success=True,
                log_text="All ports verified successfully",
                sysbox_runtime=sysbox_runtime,
            )
        except Exception as e:
            return DockerConnectionCheckResult(
                success=False,
                log_text=str(e),
                sysbox_runtime=sysbox_runtime,
            )

    async def verify_single_port(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        job_batch_id: str,
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo,
        private_key: str,
        public_key: str,
        internal_port: int,
        external_port: int,
        sysbox_runtime: bool = False,
    ) -> DockerConnectionCheckResult:
        default_extra = {
            "job_batch_id": job_batch_id,
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_info.uuid,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "ssh_username": executor_info.ssh_username,
            "ssh_port": executor_info.ssh_port,
            "version": settings.VERSION,
            "sysbox_runtime": sysbox_runtime,
            "internal_port": internal_port,
            "external_port": external_port,
        }

        container_name = f"container_{miner_hotkey}_{external_port}"

        try:
            docker_cmd = f"sh -c 'mkdir -p ~/.ssh && echo \"{public_key}\" >> ~/.ssh/authorized_keys && ssh-keygen -A && service ssh start && tail -f /dev/null'"
            command = (
                f'/usr/bin/docker run -d '
                f'{"--runtime=sysbox-runc " if sysbox_runtime else ""}'
                f'--name {container_name} --gpus all '
                f'-p {internal_port}:22 '
                f'{DOCKER_DIND_IMAGE} '
                f'{docker_cmd}'
            )

            result = await ssh_client.run(command)
            if result.exit_status != 0:
                error_message = result.stderr.strip() if result.stderr else "No error message available"
                log_text = _m(
                    "Error creating docker connection",
                    extra=get_extra_info({
                        **default_extra,
                        "error": error_message
                    }),
                )
                logger.error(log_text, exc_info=True)

                try:
                    command = f"/usr/bin/docker rm {container_name} -f"
                    await ssh_client.run(command)
                except Exception as e:
                    pass

                return DockerConnectionCheckResult(
                    success=False,
                    log_text=str(log_text),
                    sysbox_runtime=sysbox_runtime,
                )

            await asyncio.sleep(5)

            pkey = asyncssh.import_private_key(private_key)
            async with asyncssh.connect(
                host=executor_info.address,
                port=external_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=None,
            ) as _:
                log_text = _m(
                    "Connected into docker container",
                    extra=default_extra,
                )
                logger.info(log_text)

                if sysbox_runtime:
                    command = f"docker pull hello-world"
                    result = await ssh_client.run(command)
                    if result.exit_status != 0:
                        error_message = result.stderr.strip() if result.stderr else "No error message available"
                        logger.error(
                            _m(
                                "Error DinD not working",
                                extra=get_extra_info({**default_extra, "error": error_message}),
                            ),
                            exc_info=True,
                        )
                        sysbox_runtime = False

            command = f"/usr/bin/docker rm {container_name} -f"
            await ssh_client.run(command)

            return DockerConnectionCheckResult(
                success=True,
                log_text=str(log_text),
                sysbox_runtime=sysbox_runtime,
            )
        except Exception as e:
            log_text = _m(
                "Error connection docker container",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            try:
                command = f"/usr/bin/docker rm {container_name} -f"
                await ssh_client.run(command)
            except Exception as e:
                pass

            return DockerConnectionCheckResult(
                success=False,
                log_text=str(log_text),
                sysbox_runtime=sysbox_runtime,
            )

    async def check_pod_running(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        container_name: str,
        executor_info: ExecutorSSHInfo,
    ):
        # check container running or not
        command = f"/usr/bin/docker exec -i {container_name} sh -c 'cat ~/.ssh/authorized_keys'"
        result = await ssh_client.run(command)
        if result.stdout.strip():
            return result.stdout.strip().split('\n')

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

        return []

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
    ) -> tuple[bool, str | None]:
        # check gpu usages
        for detail in gpu_details:
            gpu_utilization = detail.get("gpu_utilization", GPU_UTILIZATION_LIMIT)
            gpu_memory_utilization = detail.get("memory_utilization", GPU_MEMORY_UTILIZATION_LIMIT)
            if len(gpu_processes) > 0 and (gpu_utilization >= GPU_UTILIZATION_LIMIT or gpu_memory_utilization > GPU_MEMORY_UTILIZATION_LIMIT):
                log_text = _m(
                    "High gpu utilization detected" if not rented else "GPU is using in some other places on the rented machine",
                    extra=get_extra_info({
                        **default_extra,
                        "gpu_utilization": gpu_utilization,
                        "gpu_memory_utilization": gpu_memory_utilization,
                        "gpu_processes": gpu_processes,
                    }),
                )
                return False, log_text

        return True, None

    async def create_task(
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
                docker_checksums = await shell.get_checksums_over_scp('/usr/bin/docker')
                if docker_checksums != DOCKER_DIGEST:
                    logger.info(
                        _m(
                            "Docker checksum",
                            extra=get_extra_info({
                                **default_extra,
                                "checksum": docker_checksums,
                                "DOCKER_DIGEST": DOCKER_DIGEST
                            }),
                        )
                    )
                    # raise Exception("Docker is altered")

                python_checksums = await shell.get_checksums_over_scp('/usr/bin/python')
                if python_checksums != PYTHON_DIGEST or executor_info.python_path != '/usr/bin/python':
                    logger.info(
                        _m(
                            "Python checksum",
                            extra=get_extra_info({
                                **default_extra,
                                "checksum": python_checksums,
                                "PYTHON_DIGEST": PYTHON_DIGEST
                            }),
                        )
                    )
                    # raise Exception("Python is altered")

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

                # get available port maps
                port_map_key = f"{AVAILABLE_PORT_MAPS_PREFIX}:{miner_info.miner_hotkey}:{executor_info.uuid}"
                port_maps = await self.redis_service.lrange(port_map_key)
                machine_spec = {
                    **updated_machine_spec,
                    "available_port_maps": [port_map.decode().split(",") for port_map in port_maps],
                }

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

                collateral_deposited, collateral_contract_error_message = await self.collateral_contract_service.is_eligible_executor(
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
                        f"GPU count({gpu_count}) is greater than the maximum allowed ({MAX_GPU_COUNT}).",
                        extra=get_extra_info(default_extra),
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
                        extra_info["help_text"] = (
                            "If you have the gpu machine and encountering this issue consistently, "
                            "then please pull the latest version of github repository and follow the installation guide here: "
                            "https://github.com/Datura-ai/compute-subnet/tree/main/neurons/executor. "
                            "Also, please configure the nvidia-container-runtime correctly. Check out here: "
                            "https://stackoverflow.com/questions/72932940/failed-to-initialize-nvml-unknown-error-in-docker-after-few-hours "
                            "https://bobcares.com/blog/docker-failed-to-initialize-nvml-unknown-error/"
                        )

                    log_text = _m(
                        f"No GPU in list or GPU count({gpu_count}) is 0. No need to run job.",
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
                        "Nvidia driver is altered",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                "gpu_model": gpu_model,
                                "gpu_count": gpu_count,
                                "nvidia_driver": nvidia_driver,
                                "libnvidia_ml": libnvidia_ml,
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
                        "Machine spec is changed",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                "prev_spec": prev_spec,
                                "current_spec": gpu_model_count,
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
                        "GPUs are changed",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                "prev_uuids": prev_uuids,
                                "gpu_uuids": gpu_uuids,
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
                        "Your GPUs are banned due to low rental-rate in the site.",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                "prev_uuids": prev_uuids,
                                "gpu_uuids": gpu_uuids,
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
                        f"Executor is duplicated",
                        extra=get_extra_info(default_extra),
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
                    container_name = rented_machine.get("container_name", "")
                    ssh_pub_keys = await self.check_pod_running(
                        ssh_client=shell.ssh_client,
                        container_name=container_name,
                        executor_info=executor_info,
                    )
                    if not ssh_pub_keys:
                        log_text = _m(
                            "Pod is not running",
                            extra=get_extra_info(
                                {
                                    **default_extra,
                                    "container_name": container_name,
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

                    # In backend, there are 2 scores. actual score and job score.
                    # job score is the score which executor gets when matrix multiply is finished.
                    # actual score is the score which executor gets for incentive
                    # In rented executor, there should be no job score. But we can't give actual score to executor until it pass rental check.
                    # So, if executor is rented but didn't pass rental check, we can give 0 for actual score and 1 for job score, because if both scores are 0, executor will be flagged as invalid in backend.
                    job_score = 0
                    actual_score = 1
                    log_msg = "Executor is already rented."

                    if not collateral_deposited and settings.ENABLE_COLLATERAL_CONTRACT and not settings.ENABLE_NEW_INCENTIVE_ALGO:
                        job_score = 1
                        actual_score = 0
                        log_msg = "Executor is rented. But not eligible from collateral contract."
                    elif not is_rental_succeed:
                        job_score = 1
                        actual_score = 0
                        log_msg = "Executor is rented. Set score 0 until it's verified by rental check"
                    elif not collateral_deposited and not settings.ENABLE_COLLATERAL_CONTRACT and not settings.ENABLE_NEW_INCENTIVE_ALGO:
                        log_msg = "Executor is rented. But not eligible from collateral contract. Will not have score very soon."

                    log_text = _m(
                        log_msg,
                        extra=get_extra_info({**default_extra, "actual_score": actual_score, "is_rental_succeed": is_rental_succeed, "job_score": job_score}),
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
                    docker_connection_check_result = await self.batch_verify_ports(
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

                is_valid = await self.validation_service.validate_gpu_model_and_process_job(
                    ssh_client=shell.ssh_client,
                    executor_info=executor_info,
                    default_extra=default_extra,
                    machine_spec=machine_spec
                )

                if not is_valid:
                    log_text = _m(
                        "GPU Verification failed",
                        extra=get_extra_info(default_extra),
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

                # get available port maps
                port_maps = await self.redis_service.lrange(port_map_key)
                machine_spec = {
                    **machine_spec,
                    "available_port_maps": [port_map.decode().split(",") for port_map in port_maps],
                }

                job_score = 1
                actual_score = 1
                log_msg = "Train task is finished."

                if not collateral_deposited and settings.ENABLE_COLLATERAL_CONTRACT and not settings.ENABLE_NEW_INCENTIVE_ALGO:
                    actual_score = 0
                    job_score = 0
                    log_msg = "Train task is finished. But not eligible from collateral contract."
                elif len(port_maps) < MIN_PORT_COUNT:
                    actual_score = 0
                    log_msg = f"Current port maps: {len(port_maps)}. Minimum required: {MIN_PORT_COUNT}."
                elif not is_rental_succeed:
                    actual_score = 0
                    log_msg = "Train task is finished. Set score 0 until it's verified by rental check"
                elif not collateral_deposited and not settings.ENABLE_COLLATERAL_CONTRACT and not settings.ENABLE_NEW_INCENTIVE_ALGO:
                    log_msg = "Train task is finished. But not eligible from collateral contract. Will not have score very soon."

                success = True if actual_score > 0 else False

                log_text = _m(
                    log_msg,
                    extra=get_extra_info(
                        {
                            **default_extra,
                            "job_score": job_score,
                            "acutal_score": actual_score,
                            "gpu_model": gpu_model,
                            "gpu_count": gpu_count,
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
                "Error creating task for executor",
                extra=get_extra_info({**default_extra, "error": str(e)}),
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

            actual_errors = [error for error in errors if "warnning" not in error.lower()]

            if len(results) == 0 and len(actual_errors) > 0:
                logger.error(_m("Failed to execute command!", extra=get_extra_info({**default_extra, "errors": actual_errors})))
                return None, str(actual_errors)

            if len(results) == 0:
                logger.error(_m("Failed to execute command!", extra=get_extra_info({**default_extra, "error": "No results"})))
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
