import os
import asyncio
import json
import logging
import time
import random
from typing import Annotated

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
)
from services.redis_service import RedisService, RENTED_MACHINE_SET, AVAILABLE_PORTS_PREFIX
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

    def check_digests(self, result, list_digests):
        # Check if each digest exists in list_digests
        digests_in_list = {}
        each_digests = result['all_container_digests']
        for each_digest in each_digests:
            digest = each_digest['digest']
            digests_in_list[digest] = digest in list_digests.values()

        return digests_in_list

    def check_duplidate_digests(self, result):
        # Find duplicate digests in results
        digest_count = {}
        all_container_digests = result.get('all_container_digests', [])
        for container_digest in all_container_digests:
            digest = container_digest['digest']
            if digest in digest_count:
                digest_count[digest] += 1
            else:
                digest_count[digest] = 1

        duplicates = {digest: count for digest, count in digest_count.items() if count > 1}
        return duplicates

    def check_empty_digests(self, result):
        # Find empty digests in results
        all_container_digests = result.get('all_container_digests', [])
        return len(all_container_digests) == 0

    def validate_digests(self, digests_in_list, duplicates, digests_empty):
        # Check if any digest in digests_in_list is False
        if any(not is_in_list for is_in_list in digests_in_list.values()):
            return False

        if duplicates:
            return False

        if digests_empty:
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

    def get_available_port(
        self,
        port_range: str,
        host_ssh_port: int,
    ) -> int:
        if port_range:
            if '-' in port_range:
                min_port, max_port = map(int, (part.strip() for part in port_range.split('-')))
                ports = list(range(min_port, max_port + 1))
            else:
                ports = list(map(int, (part.strip() for part in port_range.split(','))))
        else:
            # Default range if port_range is empty
            ports = list(range(40000, 65536))

        if host_ssh_port in ports:
            ports.remove(host_ssh_port)

        if not ports:
            return 0

        return random.choice(ports)

    async def docker_connection_check(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        job_batch_id: str,
        miner_hotkey: str,
        executor_info: ExecutorSSHInfo,
        private_key: str,
        public_key: str,
    ):
        ssh_port = self.get_available_port(executor_info.port_range, executor_info.ssh_port)
        executor_name = f"{executor_info.uuid}_{executor_info.address}_{executor_info.port}"
        default_extra = {
            "job_batch_id": job_batch_id,
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_info.uuid,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "ssh_username": executor_info.ssh_username,
            "ssh_port": ssh_port,
        }
        context.set(f"[_docker_connection_check][{executor_name}]")

        if ssh_port == 0:
            log_text = _m(
                "No port available for docker container",
                extra=get_extra_info(default_extra),
            )
            log_status = "error"
            logger.error(log_text, exc_info=True)

            return False, log_text, log_status

        try:
            log_text = _m(
                "Creating docker container",
                extra=default_extra,
            )
            log_status = "info"
            logger.info(log_text)

            container_name = f"container_{miner_hotkey}"
            docker_cmd = f"sh -c 'mkdir -p ~/.ssh && echo \"{public_key}\" >> ~/.ssh/authorized_keys && ssh-keygen -A && service ssh start && tail -f /dev/null'"
            command = f"docker run -d --name {container_name} -p {ssh_port}:22 daturaai/compute-subnet-executor:latest {docker_cmd}"

            result = await ssh_client.run(command, timeout=20)
            if result.exit_status != 0:
                log_text = _m(
                    "Error creating docker connection",
                    extra=get_extra_info(default_extra),
                )
                log_status = "error"
                logger.error(log_text, exc_info=True)

                return False, log_text, log_status

            await asyncio.sleep(2)

            pkey = asyncssh.import_private_key(private_key)
            async with asyncssh.connect(
                host=executor_info.address,
                port=ssh_port,
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
                key = f"{AVAILABLE_PORTS_PREFIX}:{miner_hotkey}:{executor_info.uuid}"
                existing_ports = await self.redis_service.get(key)
                if existing_ports:
                    port_set = set(existing_ports.decode().split(','))
                    port_set.add(str(ssh_port))
                    ports = ','.join(port_set)
                else:
                    ports = f'{ssh_port}'
                await self.redis_service.set(key, ports)

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
                command = f"docker container stop {container_name} && docker container prune -f"
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
                if gpu_count > MAX_GPU_COUNT:
                    score = 0
                    log_status = "warning"
                    log_text = _m(
                        f"GPU count({gpu_count}) is greater than the maximum allowed ({MAX_GPU_COUNT}).",
                        extra=get_extra_info(default_extra),
                    )
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
                        "Machine spec scraped",
                        extra=get_extra_info(
                            {**default_extra, "gpu_model": gpu_model, "gpu_count": gpu_count}
                        ),
                    ),
                )

                if max_score == 0 or gpu_count == 0:
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

                logger.info(
                    _m(
                        f"Got GPU specs: {gpu_model} with max score: {max_score}",
                        extra=get_extra_info(default_extra),
                    ),
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
                    # if not rented, check docker digests
                    digests_in_list = self.check_digests(machine_spec, docker_hub_digests)
                    duplicates = self.check_duplidate_digests(machine_spec)
                    digests_empty = self.check_empty_digests(machine_spec)  # True: docker image empty, False: docker image not empty
                    # Validate digests
                    self.is_valid = self.validate_digests(digests_in_list, duplicates, digests_empty)
                    if not self.is_valid:
                        log_text = _m(
                            "Docker digests are not valid",
                            extra=get_extra_info({
                                **default_extra,
                                "docker_digests": machine_spec.get('all_container_digests', [])
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
                avg_job_time = hashcat_config.get("average_time")[gpu_count - 1] if hashcat_config.get("average_time") else 60
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
                key = f"{AVAILABLE_PORTS_PREFIX}:{miner_info.miner_hotkey}:{executor_info.uuid}"
                await self.redis_service.set(key, '')
            except Exception as redis_error:
                log_text = _m(
                    "Error creating task for executor",
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
                "command": command,
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
