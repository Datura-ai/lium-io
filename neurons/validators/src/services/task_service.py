import os
import asyncio
import json
import logging
import time
import random
from typing import Annotated, Optional, Tuple

import asyncssh
import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from payload_models.payloads import MinerJobRequestPayload, MinerJobEnryptedFiles

from core.utils import _m, context, get_extra_info
from services.const import (
    DOWNLOAD_SPEED_WEIGHT,
    GPU_MAX_SCORES,
    JOB_TAKEN_TIME_WEIGHT,
    MAX_DOWNLOAD_SPEED,
    MAX_UPLOAD_SPEED,
    UPLOAD_SPEED_WEIGHT,
    MAX_GPU_COUNT,
    UNRENTED_MULTIPLIER,
    HASHCAT_CONFIGS,
    LIB_NVIDIA_ML_DIGESTS,
    DOCKER_DIGESTS,
    GPU_UTILIZATION_LIMIT,
    GPU_MEMORY_UTILIZATION_LIMIT,
)
from services.redis_service import (
    RedisService,
    RENTED_MACHINE_SET,
    DUPLICATED_MACHINE_SET,
    AVAILABLE_PORT_MAPS_PREFIX,
)
from services.ssh_service import SSHService
from services.hash_service import HashService

logger = logging.getLogger(__name__)

JOB_LENGTH = 300


class TaskService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
    ):
        self.ssh_service = ssh_service
        self.redis_service = redis_service
        self.is_valid = True

    async def upload_directory(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        local_dir: str,
        remote_dir: str
    ):
        """Uploads a directory recursively to a remote server using AsyncSSH."""
        async with ssh_client.start_sftp_client() as sftp_client:
            for root, dirs, files in os.walk(local_dir):
                relative_dir = os.path.relpath(root, local_dir)
                remote_path = os.path.join(remote_dir, relative_dir)

                # Create remote directory if it doesn't exist
                result = await ssh_client.run(f'mkdir -p {remote_path}')
                if result.exit_status != 0:
                    raise Exception(f"Failed to create directory {remote_path}: {result.stderr}")

                # Upload files
                upload_tasks = []
                for file in files:
                    local_file = os.path.join(root, file)
                    remote_file = os.path.join(remote_path, file)
                    upload_tasks.append(sftp_client.put(local_file, remote_file))

                # Await all upload tasks for the current directory
                await asyncio.gather(*upload_tasks)

    def validate_digests(self, docker_digests, docker_hub_digests):
        # Check if the list is empty
        if not docker_digests:
            return False

        # Get unique digests
        unique_digests = list({item['digest'] for item in docker_digests})

        # Check for duplicates
        if len(unique_digests) != len(docker_digests):
            return False

        # Check if any digest is invalid
        for digest in unique_digests:
            if digest not in docker_hub_digests.values():
                return False

        return True

    async def clear_remote_directory(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        remote_dir: str
    ):
        try:
            await ssh_client.run(f"rm -rf {remote_dir}", timeout=10)
        except:
            pass

    def get_available_port_map(
        self,
        executor_info: ExecutorSSHInfo,
    ) -> Optional[Tuple[int, int]]:
        if executor_info.port_mappings:
            port_mappings: list[Tuple[int, int]] = json.loads(executor_info.port_mappings)
            port_mappings = [
                (internal_port, external_port)
                for internal_port, external_port in port_mappings
                if internal_port != executor_info.ssh_port and external_port != executor_info.ssh_port
            ]

            if not port_mappings:
                return None

            return random.choice(port_mappings)

        if executor_info.port_range:
            if '-' in executor_info.port_range:
                min_port, max_port = map(int, (part.strip() for part in executor_info.port_range.split('-')))
                ports = list(range(min_port, max_port + 1))
            else:
                ports = list(map(int, (part.strip() for part in executor_info.port_range.split(','))))
        else:
            # Default range if port_range is empty
            ports = list(range(40000, 65536))

        ports = [port for port in ports if port != executor_info.ssh_port]

        if not ports:
            return None

        internal_port = random.choice(ports)

        return internal_port, internal_port

    async def docker_connection_check(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        job_batch_id: str,
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo,
        private_key: str,
        public_key: str,
    ):
        port_map = self.get_available_port_map(executor_info)
        if port_map is None:
            log_text = _m(
                "No port available for docker container",
                extra=get_extra_info({
                    "job_batch_id": job_batch_id,
                    "miner_hotkey": miner_hotkey,
                    "executor_uuid": executor_info.uuid,
                    "executor_ip_address": executor_info.address,
                    "executor_port": executor_info.port,
                    "ssh_username": executor_info.ssh_username,
                    "ssh_port": executor_info.ssh_port,
                }),
            )
            log_status = "error"
            logger.error(log_text, exc_info=True)

            return False, log_text, log_status

        internal_port, external_port = port_map
        executor_name = f"{executor_info.uuid}_{executor_info.address}_{executor_info.port}"
        default_extra = {
            "job_batch_id": job_batch_id,
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_info.uuid,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "ssh_username": executor_info.ssh_username,
            "ssh_port": executor_info.ssh_port,
            "internal_port": internal_port,
            "external_port": external_port,
        }
        context.set(f"[_docker_connection_check][{executor_name}]")

        container_name = f"container_{miner_hotkey}"

        try:
            log_text = _m(
                "Creating docker container",
                extra=default_extra,
            )
            log_status = "info"
            logger.info(log_text)

            docker_cmd = f"sh -c 'mkdir -p ~/.ssh && echo \"{public_key}\" >> ~/.ssh/authorized_keys && ssh-keygen -A && service ssh start && tail -f /dev/null'"
            command = f"docker run -d --name {container_name} -p {internal_port}:22 daturaai/compute-subnet-executor:latest {docker_cmd}"

            result = await ssh_client.run(command, timeout=20)
            if result.exit_status != 0:
                log_text = _m(
                    "Error creating docker connection",
                    extra=get_extra_info(default_extra),
                )
                log_status = "error"
                logger.error(log_text, exc_info=True)

                try:
                    command = f"docker rm {container_name} -f"
                    await ssh_client.run(command, timeout=20)
                except:
                    pass

                return False, log_text, log_status

            await asyncio.sleep(3)

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

                # set port on redis
                key = f"{AVAILABLE_PORT_MAPS_PREFIX}:{miner_hotkey}:{executor_info.uuid}"
                port_map = f"{internal_port},{external_port}"

                # delete all the same port_maps in the list
                await self.redis_service.lrem(key=key, element=port_map)

                # insert port_map in the list
                await self.redis_service.lpush(key, port_map)

                # keep the latest 10 port maps
                port_maps = await self.redis_service.lrange(key)
                if len(port_maps) > 10:
                    await self.redis_service.rpop(key)

            command = f"docker rm {container_name} -f"
            await ssh_client.run(command, timeout=20)

            return True, log_text, log_status
        except Exception as e:
            log_text = _m(
                "Error connection docker container",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            log_status = "error"
            logger.error(log_text, exc_info=True)

            try:
                command = f"docker rm {container_name} -f"
                await ssh_client.run(command, timeout=20)
            except:
                pass

            return False, log_text, log_status

    async def create_task(
        self,
        miner_info: MinerJobRequestPayload,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
        public_key: str,
        encypted_files: MinerJobEnryptedFiles,
        docker_hub_digests: dict[str, str]
    ):
        default_extra = {
            "job_batch_id": miner_info.job_batch_id,
            "miner_hotkey": miner_info.miner_hotkey,
            "executor_uuid": executor_info.uuid,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }
        try:
            logger.info(_m("Start job on an executor", extra=get_extra_info(default_extra)))

            private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
            pkey = asyncssh.import_private_key(private_key)

            async with asyncssh.connect(
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=None,
            ) as ssh_client:
                remote_dir = f"{executor_info.root_dir}/temp"
                await ssh_client.run(f"rm -rf {remote_dir}")
                await ssh_client.run(f"mkdir -p {remote_dir}")

                # upload temp directory
                await self.upload_directory(ssh_client, encypted_files.tmp_directory, remote_dir)

                remote_machine_scrape_file_path = f"{remote_dir}/{encypted_files.machine_scrape_file_name}"
                remote_score_file_path = f"{remote_dir}/{encypted_files.score_file_name}"

                logger.info(
                    _m(
                        "Uploaded files to run job",
                        extra=get_extra_info(default_extra),
                    ),
                )

                machine_specs, _ = await self._run_task(
                    ssh_client=ssh_client,
                    miner_hotkey=miner_info.miner_hotkey,
                    executor_info=executor_info,
                    command=f"chmod +x {remote_machine_scrape_file_path} && {remote_machine_scrape_file_path}"
                )
                if not machine_specs:
                    log_status = "warning"
                    log_text = _m("No machine specs found", extra=get_extra_info(default_extra))
                    logger.warning(log_text)

                    await self.clear_remote_directory(ssh_client, remote_dir)

                    return (
                        None,
                        executor_info,
                        0,
                        0,
                        miner_info.job_batch_id,
                        log_status,
                        log_text,
                    )

                machine_spec = json.loads(self.ssh_service.decrypt_payload(encypted_files.encrypt_key, machine_specs[0].strip()))

                gpu_model = None
                if machine_spec.get("gpu", {}).get("count", 0) > 0:
                    details = machine_spec["gpu"].get("details", [])
                    if len(details) > 0:
                        gpu_model = details[0].get("name", None)

                max_score = 0
                if gpu_model:
                    max_score = GPU_MAX_SCORES.get(gpu_model, 0)

                gpu_count = machine_spec.get("gpu", {}).get("count", 0)
                gpu_details = machine_spec.get("gpu", {}).get("details", [])

                nvidia_driver = machine_spec.get("gpu", {}).get("driver", '')
                libnvidia_ml = machine_spec.get('md5_checksums', {}).get('libnvidia_ml', '')

                docker_version = machine_spec.get("docker", {}).get("version", '')
                docker_digest = machine_spec.get('md5_checksums', {}).get('docker', '')
                container_id = machine_spec.get('docker', {}).get('container_id', '')

                logger.info(
                    _m(
                        "Machine spec scraped",
                        extra=get_extra_info({
                            **default_extra,
                            "gpu_model": gpu_model,
                            "gpu_count": gpu_count,
                            "nvidia_driver": nvidia_driver,
                            "libnvidia_ml": libnvidia_ml,
                        }),
                    ),
                )

                if gpu_count > MAX_GPU_COUNT:
                    log_status = "warning"
                    log_text = _m(
                        f"GPU count({gpu_count}) is greater than the maximum allowed ({MAX_GPU_COUNT}).",
                        extra=get_extra_info(default_extra),
                    )
                    logger.warning(log_text)

                    await self.clear_remote_directory(ssh_client, remote_dir)

                    return (
                        machine_spec,
                        executor_info,
                        0,
                        0,
                        miner_info.job_batch_id,
                        log_status,
                        log_text,
                    )

                if max_score == 0 or gpu_count == 0 or len(gpu_details) != gpu_count:
                    extra_info = {
                        **default_extra,
                        "os_version": machine_spec.get('os', ''),
                        "nvidia_cfg": machine_spec.get('nvidia_cfg', ''),
                        "docker_cfg": machine_spec.get('docker_cfg', ''),
                        "gpu_scrape_error": machine_spec.get('gpu_scrape_error', ''),
                        "nvidia_cfg_scrape_error": machine_spec.get('nvidia_cfg_scrape_error', ''),
                        "docker_cfg_scrape_error": machine_spec.get('docker_cfg_scrape_error', '')
                    }
                    if gpu_model:
                        extra_info["gpu_model"] = gpu_model
                        extra_info["help_text"] = (
                            "If you have the gpu machine and encountering this issue consistantly, "
                            "then please pull the latest version of github repository and follow the installation guide here: "
                            "https://github.com/Datura-ai/compute-subnet/tree/main/neurons/executor. "
                            "Also, please configure the nvidia-container-runtime correctly. Check out here: "
                            "https://stackoverflow.com/questions/72932940/failed-to-initialize-nvml-unknown-error-in-docker-after-few-hours "
                            "https://bobcares.com/blog/docker-failed-to-initialize-nvml-unknown-error/"
                        )

                    log_text = _m(
                        f"Max Score({max_score}) or GPU count({gpu_count}) is 0. No need to run job.",
                        extra=get_extra_info({
                            **default_extra,
                            **extra_info,
                        }),
                    )
                    log_status = "warning"
                    logger.warning(log_text)

                    await self.clear_remote_directory(ssh_client, remote_dir)

                    return (
                        machine_spec,
                        executor_info,
                        0,
                        0,
                        miner_info.job_batch_id,
                        log_status,
                        log_text,
                    )

                if not docker_version or DOCKER_DIGESTS.get(docker_version) != docker_digest:
                    log_status = "warning"
                    log_text = _m(
                        f"Docker is altered",
                        extra=get_extra_info({
                            **default_extra,
                            "docker_version": docker_version,
                            "docker_digest": docker_digest,
                            "container_id": container_id,
                        }),
                    )
                    logger.warning(log_text)

                    await self.clear_remote_directory(ssh_client, remote_dir)

                    return (
                        machine_spec,
                        executor_info,
                        0,
                        0,
                        miner_info.job_batch_id,
                        log_status,
                        log_text,
                    )

                if nvidia_driver and LIB_NVIDIA_ML_DIGESTS.get(nvidia_driver) != libnvidia_ml:
                    log_status = "warning"
                    log_text = _m(
                        f"Nvidia driver is altered",
                        extra=get_extra_info({
                            **default_extra,
                            "gpu_model": gpu_model,
                            "gpu_count": gpu_count,
                            "nvidia_driver": nvidia_driver,
                            "libnvidia_ml": libnvidia_ml,
                        }),
                    )
                    logger.warning(log_text)

                    await self.clear_remote_directory(ssh_client, remote_dir)

                    return (
                        machine_spec,
                        executor_info,
                        0,
                        0,
                        miner_info.job_batch_id,
                        log_status,
                        log_text,
                    )

                logger.info(
                    _m(
                        f"Got GPU specs: {gpu_model} with max score: {max_score}",
                        extra=get_extra_info(default_extra),
                    ),
                )

                # check duplicated
                is_duplicated = await self.redis_service.is_elem_exists_in_set(
                    DUPLICATED_MACHINE_SET, f"{miner_info.miner_hotkey}:{executor_info.uuid}"
                )
                if is_duplicated:
                    log_status = "warning"
                    log_text = _m(
                        f"Executor is duplicated",
                        extra=get_extra_info(default_extra),
                    )
                    logger.warning(log_text)

                    await self.clear_remote_directory(ssh_client, remote_dir)

                    return (
                        machine_spec,
                        executor_info,
                        0,
                        0,
                        miner_info.job_batch_id,
                        log_status,
                        log_text,
                    )

                # check rented status
                is_rented = await self.redis_service.is_elem_exists_in_set(
                    RENTED_MACHINE_SET, f"{miner_info.miner_hotkey}:{executor_info.uuid}"
                )
                if is_rented:
                    score = max_score * gpu_count
                    log_text = _m(
                        "Executor is already rented.",
                        extra=get_extra_info({**default_extra, "score": score}),
                    )
                    log_status = "info"
                    logger.info(log_text)

                    await self.clear_remote_directory(ssh_client, remote_dir)

                    return (
                        machine_spec,
                        executor_info,
                        score,
                        0,
                        miner_info.job_batch_id,
                        log_status,
                        log_text,
                    )
                else:
                    # check gpu usages
                    for detail in gpu_details:
                        gpu_utilization = detail.get("gpu_utilization", GPU_UTILIZATION_LIMIT)
                        gpu_memory_utilization = detail.get("memory_utilization", GPU_MEMORY_UTILIZATION_LIMIT)
                        if gpu_utilization >= GPU_UTILIZATION_LIMIT or gpu_memory_utilization > GPU_MEMORY_UTILIZATION_LIMIT:
                            log_status = "warning"
                            log_text = _m(
                                f"High gpu utilization detected:",
                                extra=get_extra_info({
                                    **default_extra,
                                    "gpu_utilization": gpu_utilization,
                                    "gpu_memory_utilization": gpu_memory_utilization,
                                }),
                            )
                            logger.warning(log_text)

                            await self.clear_remote_directory(ssh_client, remote_dir)

                            return (
                                machine_spec,
                                executor_info,
                                0,
                                0,
                                miner_info.job_batch_id,
                                log_status,
                                log_text,
                            )

                    # if not rented, check docker digests
                    docker_digests = machine_spec.get("docker", {}).get("containers", [])
                    self.is_valid = self.validate_digests(docker_digests, docker_hub_digests)
                    if not self.is_valid:
                        log_text = _m(
                            "Docker digests are not valid",
                            extra=get_extra_info({
                                **default_extra,
                                "docker_digests": docker_digests
                            }),
                        )
                        log_status = "error"

                        logger.warning(log_text)

                        await self.clear_remote_directory(ssh_client, remote_dir)

                        return (
                            None,
                            executor_info,
                            0,
                            0,
                            miner_info.job_batch_id,
                            log_status,
                            log_text,
                        )

                    # if not rented, check renting ports
                    success, log_text, log_status = await self.docker_connection_check(
                        ssh_client=ssh_client,
                        job_batch_id=miner_info.job_batch_id,
                        miner_hotkey=miner_info.miner_hotkey,
                        executor_info=executor_info,
                        private_key=private_key,
                        public_key=public_key,
                    )
                    if not success:
                        await self.clear_remote_directory(ssh_client, remote_dir)

                        return (
                            None,
                            executor_info,
                            0,
                            0,
                            miner_info.job_batch_id,
                            log_status,
                            log_text,
                        )

                # scoring
                hashcat_config = HASHCAT_CONFIGS[gpu_model]
                if not hashcat_config:
                    log_text = _m(
                        "No config for hashcat",
                        extra=get_extra_info(default_extra),
                    )
                    log_status = "error"

                    logger.warning(log_text)

                    await self.clear_remote_directory(ssh_client, remote_dir)

                    return (
                        None,
                        executor_info,
                        0,
                        0,
                        miner_info.job_batch_id,
                        log_status,
                        log_text,
                    )

                num_digits = hashcat_config.get('digits', 11)
                avg_job_time = hashcat_config.get("average_time")[gpu_count - 1 if gpu_count <= 8 else 7] if hashcat_config.get("average_time") else 60
                hash_service = HashService.generate(
                    gpu_count=gpu_count,
                    num_digits=num_digits,
                    timeout=int(avg_job_time * 2.5)
                )
                start_time = time.time()

                results, err = await self._run_task(
                    ssh_client=ssh_client,
                    miner_hotkey=miner_info.miner_hotkey,
                    executor_info=executor_info,
                    command=f"export PYTHONPATH={executor_info.root_dir}:$PYTHONPATH && {executor_info.python_path} {remote_score_file_path} '{hash_service.payload}'",
                )
                if not results:
                    log_text = _m(
                        "No result from training job task.",
                        extra=get_extra_info(default_extra),
                    )
                    log_status = "warning"
                    logger.warning(log_text)

                    await self.clear_remote_directory(ssh_client, remote_dir)

                    return (
                        machine_spec,
                        executor_info,
                        0,
                        0,
                        miner_info.job_batch_id,
                        log_status,
                        log_text,
                    )

                end_time = time.time()
                job_taken_time = end_time - start_time

                result = json.loads(results[0])
                answer = result["answer"]

                score = 0

                logger.info(
                    _m(
                        f"Results from training job task: {str(result)}",
                        extra=get_extra_info(default_extra),
                    ),
                )
                log_text = ""
                log_status = ""

                if err is not None:
                    log_status = "error"
                    log_text = _m(
                        f"Error executing task on executor: {err}",
                        extra=get_extra_info(default_extra),
                    )
                    logger.error(log_text)

                elif answer != hash_service.answer:
                    log_status = "error"
                    log_text = _m(
                        f"Hashcat incorrect Answer",
                        extra=get_extra_info(default_extra),
                    )
                    logger.error(log_text)

                # elif job_taken_time > avg_job_time * 2:
                #     log_status = "error"
                #     log_text = _m(
                #         f"Incorrect Answer",
                #         extra=get_extra_info(default_extra),
                #     )
                #     logger.error(log_text)

                else:
                    logger.info(
                        _m(
                            "Job taken time for executor",
                            extra=get_extra_info(
                                {**default_extra, "job_taken_time": job_taken_time}
                            ),
                        ),
                    )

                    upload_speed = machine_spec.get("network", {}).get("upload_speed", 0)
                    download_speed = machine_spec.get("network", {}).get("download_speed", 0)

                    # Ensure upload_speed and download_speed are not None
                    upload_speed = upload_speed if upload_speed is not None else 0
                    download_speed = download_speed if download_speed is not None else 0

                    job_taken_score = (
                        min(avg_job_time * 0.7 / job_taken_time, 1) if job_taken_time > 0 else 0
                    )
                    upload_speed_score = min(upload_speed / MAX_UPLOAD_SPEED, 1)
                    download_speed_score = min(download_speed / MAX_DOWNLOAD_SPEED, 1)

                    score = max_score * gpu_count * UNRENTED_MULTIPLIER * (
                        job_taken_score * JOB_TAKEN_TIME_WEIGHT
                        + upload_speed_score * UPLOAD_SPEED_WEIGHT
                        + download_speed_score * DOWNLOAD_SPEED_WEIGHT
                    )

                    log_status = "info"
                    log_text = _m(
                        "Train task finished",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                "score": score,
                                "job_taken_time": job_taken_time,
                                "upload_speed": upload_speed,
                                "download_speed": download_speed,
                                "gpu_model": gpu_model,
                                "gpu_count": gpu_count,
                            }
                        ),
                    )

                    logger.info(log_text)

                logger.info(
                    _m(
                        "SSH connection closed for executor",
                        extra=get_extra_info(default_extra),
                    ),
                )

                await self.clear_remote_directory(ssh_client, remote_dir)
                return (
                    machine_spec,
                    executor_info,
                    score,
                    score,
                    miner_info.job_batch_id,
                    log_status,
                    log_text,
                )
        except Exception as e:
            log_status = "error"
            log_text = _m(
                "Error creating task for executor",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )

            try:
                key = f"{AVAILABLE_PORT_MAPS_PREFIX}:{miner_info.miner_hotkey}:{executor_info.uuid}"
                await self.redis_service.delete(key)
            except Exception as redis_error:
                log_text = _m(
                    "Error creating task redis_reset_error",
                    extra=get_extra_info({
                        **default_extra,
                        "error": str(e),
                        "redis_reset_error": str(redis_error),
                    }),
                )

            logger.error(
                log_text,
                exc_info=True,
            )

            return (
                None,
                executor_info,
                0,
                0,
                miner_info.job_batch_id,
                log_status,
                log_text,
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
                "command": command[:100] + ('...' if len(command) > 100 else ''),
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
                logger.error(_m("Failed to execute command!", extra=get_extra_info(default_extra)))
                raise Exception("Failed to execute command!")

            return results, None
        except Exception as e:
            logger.error(
                _m("Run task error to executor", extra=get_extra_info(default_extra)),
                exc_info=True,
            )

            return None, str(e)


TaskServiceDep = Annotated[TaskService, Depends(TaskService)]
