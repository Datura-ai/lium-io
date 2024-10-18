import asyncio
import contextvars
import json
import logging
import time
from pathlib import Path
from typing import Annotated

import asyncssh
import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from payload_models.payloads import MinerJobRequestPayload

from core.config import settings
from services.const import (
    DOWNLOAD_SPEED_WEIGHT,
    GPU_MAX_SCORES,
    JOB_TAKEN_TIME_WEIGHT,
    MAX_DOWNLOAD_SPEED,
    MAX_UPLOAD_SPEED,
    MIN_JOB_TAKEN_TIME,
    UPLOAD_SPEED_WEIGHT,
)
from services.ssh_service import SSHService
from services.redis_service import RedisService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the logger for the 'asyncssh' module
asyncssh_logger = logging.getLogger("asyncssh")

# Set the logging level to WARNING
asyncssh_logger.setLevel(logging.INFO)

# Create a ContextVar to hold the context information
context = contextvars.ContextVar("context", default="custom")

context.set("custom")


class ContextFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """

    def filter(self, record):
        record.context = context.get() or "Default"
        return True


# Create a custom formatter that adds the context to the log messages
class CustomFormatter(logging.Formatter):
    def format(self, record):
        try:
            return f"{getattr(record, 'context', 'Default')} {super().format(record)}"
        except Exception:
            return ""


# Add the filter to the logger
asyncssh_logger.addFilter(ContextFilter())

# Create a handler for the logger
handler = logging.StreamHandler()

# Add the handler to the logger
asyncssh_logger.addHandler(handler)

# Set the formatter for the handler
handler.setFormatter(CustomFormatter())

JOB_LENGTH = 300


class TaskService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
    ):
        self.ssh_service = ssh_service
        self.redis_service = redis_service

    async def create_task(
        self,
        miner_info: MinerJobRequestPayload,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        executor_name = f"{miner_info.miner_hotkey}_{executor_info.uuid}_{executor_info.address}_{executor_info.port}"
        try:
            logger.info(
                f"[create_task] ({executor_name}): Connecting ssh with info: {executor_info.address}:{executor_info.ssh_port}"
            )
            private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
            pkey = asyncssh.import_private_key(private_key)

            async with asyncssh.connect(
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=None,
            ) as ssh_client:
                logger.info(
                    f"[create_task][{executor_name}] SSH connection established"
                )
                await ssh_client.run(f"mkdir -p {executor_info.root_dir}/temp")
                logger.debug(
                    f"[create_task][{executor_name}] Created temporary directory"
                )

                async with ssh_client.start_sftp_client() as sftp_client:
                    # run synthetic job
                    logger.debug(f"[create_task][{executor_name}] Opened SFTP client")

                    # get machine specs
                    timestamp = int(time.time())
                    local_file_path = str(
                        Path(__file__).parent / ".." / "miner_jobs/machine_scrape.py"
                    )
                    remote_file_path = f"{executor_info.root_dir}/temp/job_{timestamp}.py"

                    await sftp_client.put(local_file_path, remote_file_path)
                    logger.info(
                        f"[create_task][{executor_name}] Uploaded machine scrape script"
                    )

                    machine_specs, _ = await self._run_task(
                        ssh_client, executor_info, remote_file_path
                    )
                    if not machine_specs:
                        logger.warning(
                            f"[create_task][{executor_name}] No result from machine scrape task."
                        )
                        return None

                    machine_spec = json.loads(machine_specs[0].strip())
                    logger.info(
                        f"[create_task][{executor_name}] Machine spec: {machine_spec}"
                    )

                    gpu_model = None
                    if machine_spec.get("gpu", {}).get("count", 0) > 0:
                        details = machine_spec["gpu"].get("details", [])
                        if len(details) > 0:
                            gpu_model = details[0].get("name", None)

                    max_score = 0
                    if gpu_model:
                        max_score = GPU_MAX_SCORES.get(gpu_model, 0)

                    gpu_count = machine_spec.get("gpu", {}).get("count", 0)

                    if max_score == 0 or gpu_count == 0:
                        logger.warning(
                            f"[create_task] Max Score({max_score}) or GPU count({gpu_count}) is 0 for executor({executor_name}). No need to run job."
                        )
                        return machine_spec, executor_info, 0

                    logger.info(
                        f"[create_task][{executor_name}] gpu model: {gpu_model}, max score: {max_score}"
                    )

                    # if executor.rented:
                    #     score = max_score * gpu_count
                    #     logger.info(
                    #         f"[create_task] Executor({executor_name}) is already rented. Give score: {score}"
                    #     )

                    #     logger.info(
                    #         f"[create_task] Task saved with status Finished for executor({executor_name})"
                    #     )
                    #     return machine_spec, executor_info

                    timestamp = int(time.time())
                    local_file_path = str(Path(__file__).parent / ".." / "miner_jobs/score.py")
                    remote_file_path = f"{executor_info.root_dir}/temp/job_{timestamp}.py"

                    await sftp_client.put(local_file_path, remote_file_path)
                    logger.info(f"[create_task][{executor_name}] Uploaded score script")

                    start_time = time.time()

                    results, err = await self._run_task(ssh_client, executor_info, remote_file_path)
                    if not results:
                        logger.warning(f"[create_task][{executor_name}] No result from task.")
                        return machine_spec, executor_info, 0

                    end_time = time.time()

                    score = 0

                    if err is not None:
                        logger.error(
                            f"[create_task][{executor_name}] Error executing task: {err}"
                        )
                    else:
                        job_taken_time = results[-1]
                        try:
                            job_taken_time = float(job_taken_time.strip())
                        except Exception:
                            job_taken_time = end_time - start_time

                        logger.info(
                            f"[create_task][{executor_name}] Job taken time: {job_taken_time}"
                        )

                        upload_speed = machine_spec.get("network", {}).get("upload_speed", 0)
                        download_speed = machine_spec.get("network", {}).get("download_speed", 0)

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
                            f"[create_task][{executor_name}] Give score {score}",
                        )

                    logger.debug(f"[create_task][{executor_name}] SFTP client closed")
                    logger.info(
                        f"[create_task][{executor_name}] SSH connection closed"
                    )

                    return machine_spec, executor_info, score
        except Exception as e:
            logger.error(f"[create_task][{executor_name}] Error creating task: {e}")
            return None

    async def _run_task(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        executor_info: ExecutorSSHInfo,
        remote_file_path: str,
    ) -> tuple[list[str] | None, str | None]:
        try:
            executor_name = f"{executor_info.uuid}_{executor_info.address}_{executor_info.port}"
            context.set(f"[_run_task][{executor_name}]")
            logger.info(
                f"[_run_task][{executor_name}] Run task -> executor(%s:%d)",
                executor_info.address,
                executor_info.ssh_port,
            )
            result = await ssh_client.run(
                f"export PYTHONPATH={executor_info.root_dir}:$PYTHONPATH && {executor_info.python_path} {remote_file_path}",
                timeout=JOB_LENGTH,
            )
            results = result.stdout.splitlines()
            errors = result.stderr.splitlines()
            logger.info(f"[_run_task][{executor_name}] results ================> {results}")
            logger.warning(f"[_run_task][{executor_name}] errors ===> {errors}")

            actual_errors = [error for error in errors if "warnning" not in error.lower()]

            if len(results) == 0 and len(actual_errors) > 0:
                logger.error(
                    f"[_run_task][{executor_name}] Failed to execute command! {actual_errors}"
                )
                raise Exception("Failed to execute command!")

            #  remove remote_file
            await ssh_client.run(f"rm {remote_file_path}", timeout=30)

            logger.info(
                f"[_run_task][{executor_name}] Run task success -> executor(%s:%d)",
                executor_info.address,
                executor_info.ssh_port,
            )
            return results, None
        except Exception as e:
            logger.error(
                f"[_run_task][{executor_name}] Run task error to executor(%s:%d): %s",
                executor_info.address,
                executor_info.ssh_port,
                str(e),
            )

            #  remove remote_file
            try:
                logger.info(
                    f"[_run_task][{executor_name}] Removing remote file - {remote_file_path}"
                )
                await asyncio.wait_for(ssh_client.run(f"rm {remote_file_path}"), timeout=10)
                logger.info(
                    f"[_run_task][{executor_name}] Removed remote file - {remote_file_path}"
                )
            except Exception as e:
                logger.error(
                    f"[_run_task][{executor_name}] Failed to remove remote file - {remote_file_path}"
                )

            return None, str(e)

    async def get_decrypted_private_key_for_task(self, uuid: str) -> str | None:
        task = await self.task_dao.get_task_by_uuid(uuid)
        if task is None:
            return None
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        return self.ssh_service.decrypt_payload(my_key.ss58_address, task.ssh_private_key)


TaskServiceDep = Annotated[TaskService, Depends(TaskService)]
