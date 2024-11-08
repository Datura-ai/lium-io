import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Annotated
import random
import string

import asyncssh
import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from payload_models.payloads import MinerJobRequestPayload

from core.config import settings
from core.utils import _m, context, get_extra_info
from services.const import (
    DOWNLOAD_SPEED_WEIGHT,
    GPU_MAX_SCORES,
    JOB_TAKEN_TIME_WEIGHT,
    MAX_DOWNLOAD_SPEED,
    MAX_UPLOAD_SPEED,
    MIN_JOB_TAKEN_TIME,
    UPLOAD_SPEED_WEIGHT,
)
from services.redis_service import RENTED_MACHINE_SET, RedisService
from services.ssh_service import SSHService

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
        self.my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()

    def generate_random_string(self, length=30):
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choices(characters, k=length))
        return random_string

    def generate_signature(self):
        random_string = self.generate_random_string()
        return random_string, f"0x{self.my_key.sign(random_string).hex()}"

    def verify_signature(self, s: str, signature: str):
        try:
            return self.my_key.verify(s, signature)
        except:
            return False

    async def create_task(
        self,
        miner_info: MinerJobRequestPayload,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
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
                await ssh_client.run(f"mkdir -p {executor_info.root_dir}/temp")

                async with ssh_client.start_sftp_client() as sftp_client:
                    # get machine specs
                    random_string, signature_value = self.generate_signature()

                    timestamp = int(time.time())
                    machine_scrape_file_path = str(
                        Path(__file__).parent / ".." / "miner_jobs/machine_scrape.py"
                    )
                    with open(machine_scrape_file_path, 'r') as file:
                        content = file.read()
                    modified_content = content.replace('signature_value', signature_value)

                    remote_file_path = f"{executor_info.root_dir}/temp/{timestamp}.py"
                    async with sftp_client.open(remote_file_path, 'w') as remote_file:
                        await remote_file.write(modified_content)

                    logger.info(
                        _m(
                            "Uploaded machine scrape script to {remote_file_path}",
                            extra=get_extra_info(default_extra),
                        ),
                    )

                    machine_specs, _ = await self._run_task(
                        ssh_client, executor_info, remote_file_path, miner_info.miner_hotkey
                    )
                    if not machine_specs:
                        log_status = "warning"
                        log_text = _m("No machine specs found", extra=get_extra_info(default_extra))
                        logger.warning(log_text)
                        return None, executor_info, 0, miner_info.job_batch_id, log_status, log_text

                    machine_spec = json.loads(machine_specs[0].strip())

                    signature = machine_spec.get("signature", "")

                    gpu_model = None
                    if machine_spec.get("gpu", {}).get("count", 0) > 0:
                        details = machine_spec["gpu"].get("details", [])
                        if len(details) > 0:
                            gpu_model = details[0].get("name", None)

                    max_score = 0
                    if gpu_model:
                        max_score = GPU_MAX_SCORES.get(gpu_model, 0)

                    gpu_count = machine_spec.get("gpu", {}).get("count", 0)

                    if not self.verify_signature(random_string, signature):
                        log_text = _m(
                            "Unverified machine spec",
                            extra=get_extra_info({
                                **default_extra,
                                "gpu_model": gpu_model,
                                "gpu_count": gpu_count,
                            }),
                        )
                        log_status = "error"
                        logger.error(log_text)

                        return None, executor_info, 0, miner_info.job_batch_id, log_status, log_text

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
                        return (
                            machine_spec,
                            executor_info,
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
                        return (
                            machine_spec,
                            executor_info,
                            score,
                            miner_info.job_batch_id,
                            log_status,
                            log_text,
                        )

                    random_string, signature_value = self.generate_signature()

                    timestamp = int(time.time())
                    score_script_path = str(Path(__file__).parent / ".." / "miner_jobs/score.py")
                    with open(score_script_path, 'r') as file:
                        content = file.read()
                    modified_content = content.replace('signature_value', signature_value)

                    remote_file_path = f"{executor_info.root_dir}/temp/{timestamp}.py"
                    async with sftp_client.open(remote_file_path, 'w') as remote_file:
                        await remote_file.write(modified_content)

                    start_time = time.time()

                    results, err = await self._run_task(
                        ssh_client, executor_info, remote_file_path, miner_info.miner_hotkey
                    )
                    if not results:
                        log_text = _m(
                            "No result from training job task.",
                            extra=get_extra_info(default_extra),
                        )
                        log_status = "warning"
                        logger.warning(log_text)
                        return (
                            machine_spec,
                            executor_info,
                            0,
                            miner_info.job_batch_id,
                            log_status,
                            log_text,
                        )

                    end_time = time.time()

                    result = json.loads(results[0])
                    signature = result.get("signature", "")

                    if not self.verify_signature(random_string, signature):
                        log_text = _m(
                            "Unverified score result",
                            extra=get_extra_info({
                                **default_extra,
                                **result,
                            }),
                        )
                        log_status = "error"
                        logger.error(log_text)

                        return machine_spec, executor_info, 0, miner_info.job_batch_id, log_status, log_text

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

                    else:
                        job_taken_time = result["time"]
                        try:
                            job_taken_time = float(job_taken_time)
                        except Exception:
                            job_taken_time = end_time - start_time

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
                            min(MIN_JOB_TAKEN_TIME / job_taken_time, 1) if job_taken_time > 0 else 0
                        )
                        upload_speed_score = min(upload_speed / MAX_UPLOAD_SPEED, 1)
                        download_speed_score = min(download_speed / MAX_DOWNLOAD_SPEED, 1)

                        score = max_score * (
                            job_taken_score * gpu_count * JOB_TAKEN_TIME_WEIGHT
                            + upload_speed_score * UPLOAD_SPEED_WEIGHT
                            + download_speed_score * DOWNLOAD_SPEED_WEIGHT
                        )

                        logger.info(
                            _m(
                                "Train task finished",
                                extra=get_extra_info(
                                    {
                                        **default_extra,
                                        "score": score,
                                        "job_taken_time": job_taken_time,
                                        "upload_speed": upload_speed,
                                        "download_speed": download_speed,
                                    }
                                ),
                            ),
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

                    return (
                        machine_spec,
                        executor_info,
                        score,
                        miner_info.job_batch_id,
                        log_status,
                        log_text,
                    )
        except Exception as e:
            logger.error(
                _m(
                    "Error creating task for executor",
                    extra=get_extra_info({**default_extra, "error": str(e)}),
                ),
                exc_info=True,
            )
            log_status = "error"
            log_text = _m(
                "Error creating task for executor",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            return None, executor_info, 0, miner_info.job_batch_id, log_status, log_text

    async def _run_task(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        executor_info: ExecutorSSHInfo,
        remote_file_path: str,
        miner_hotkey: str,
    ) -> tuple[list[str] | None, str | None]:
        try:
            executor_name = f"{executor_info.uuid}_{executor_info.address}_{executor_info.port}"
            default_extra = {
                "executor_uuid": executor_info.uuid,
                "executor_ip_address": executor_info.address,
                "executor_port": executor_info.port,
                "remote_file_path": remote_file_path,
                "miner_hotkey": miner_hotkey,
            }
            context.set(f"[_run_task][{executor_name}]")
            logger.info(
                _m(
                    "Running task for executor",
                    extra=get_extra_info({**default_extra, "remote_file_path": remote_file_path}),
                ),
            )
            result = await ssh_client.run(
                f"export PYTHONPATH={executor_info.root_dir}:$PYTHONPATH && {executor_info.python_path} {remote_file_path}",
                timeout=JOB_LENGTH,
            )
            results = result.stdout.splitlines()
            errors = result.stderr.splitlines()

            actual_errors = [error for error in errors if "warnning" not in error.lower()]

            if len(results) == 0 and len(actual_errors) > 0:
                logger.error(_m("Failed to execute command!", extra=get_extra_info(default_extra)))
                raise Exception("Failed to execute command!")

            #  remove remote_file
            await ssh_client.run(f"rm {remote_file_path}", timeout=30)

            return results, None
        except Exception as e:
            logger.error(
                _m("Run task error to executor", extra=get_extra_info(default_extra)),
                exc_info=True,
            )

            #  remove remote_file
            try:
                await asyncio.wait_for(ssh_client.run(f"rm {remote_file_path}"), timeout=10)
            except Exception:
                logger.error(
                    _m("Failed to remove remote file", extra=get_extra_info(default_extra)),
                    exc_info=True,
                )

            return None, str(e)

    async def get_decrypted_private_key_for_task(self, uuid: str) -> str | None:
        task = await self.task_dao.get_task_by_uuid(uuid)
        if task is None:
            return None
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        return self.ssh_service.decrypt_payload(my_key.ss58_address, task.ssh_private_key)


TaskServiceDep = Annotated[TaskService, Depends(TaskService)]
