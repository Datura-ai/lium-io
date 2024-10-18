import asyncio
import json
import time
from pathlib import Path
from typing import Annotated

import asyncssh
import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from payload_models.payloads import MinerJobRequestPayload

from core.config import settings
from core.utils import configure_logs_of_other_modules, context, get_extra_info, get_logger
from daos.executor import ExecutorDao
from daos.task import TaskDao
from models.executor import Executor
from models.task import Task, TaskStatus
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

logger = get_logger(__name__)
configure_logs_of_other_modules()

JOB_LENGTH = 300


class TaskService:
    def __init__(
        self,
        task_dao: Annotated[TaskDao, Depends(TaskDao)],
        executor_dao: Annotated[ExecutorDao, Depends(ExecutorDao)],
        ssh_service: Annotated[SSHService, Depends(SSHService)],
    ):
        self.task_dao = task_dao
        self.executor_dao = executor_dao
        self.ssh_service = ssh_service

    async def create_task(
        self,
        miner_info: MinerJobRequestPayload,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        executor_name = f"{executor_info.uuid}_{executor_info.address}_{executor_info.port}"
        default_extra = {
            "miner_hotkey": miner_info.miner_hotkey,
            "executor_uuid": executor_info.uuid,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }
        try:
            # logger.info(
            #     f"[create_task] Creating task for executor({executor_name}): Upsert executor uuid: {executor_info.uuid}"
            # )
            logger.info("Update or create an executor", extra=get_extra_info(default_extra))
            await self.executor_dao.upsert(
                Executor(
                    miner_address=miner_info.miner_address,
                    miner_port=miner_info.miner_port,
                    miner_hotkey=miner_info.miner_hotkey,
                    executor_id=executor_info.uuid,
                    executor_ip_address=executor_info.address,
                    executor_ssh_username=executor_info.ssh_username,
                    executor_ssh_port=executor_info.ssh_port,
                )
            )

            private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
            pkey = asyncssh.import_private_key(private_key)

            logger.info(
                f"Connecting with SSH INFO(ssh -p {executor_info.ssh_port} {executor_info.ssh_username}:{executor_info.address})",
                extra=get_extra_info(default_extra),
            )

            async with asyncssh.connect(
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=None,
            ) as ssh_client:
                logger.info(
                    f"SSH Connection Established. Creating temp directory at {executor_info.root_dir}/temp",
                    extra=get_extra_info(default_extra),
                )

                await ssh_client.run(f"mkdir -p {executor_info.root_dir}/temp")
                async with ssh_client.start_sftp_client() as sftp_client:
                    # get machine specs
                    timestamp = int(time.time())
                    local_file_path = str(
                        Path(__file__).parent / ".." / "miner_jobs/machine_scrape.py"
                    )
                    remote_file_path = f"{executor_info.root_dir}/temp/job_{timestamp}.py"
                    await sftp_client.put(local_file_path, remote_file_path)

                    logger.info(
                        f"Uploaded machine scrape script to {remote_file_path}",
                        extra=get_extra_info(default_extra),
                    )

                    machine_specs, _ = await self._run_task(
                        ssh_client, executor_info, remote_file_path
                    )
                    if not machine_specs:
                        logger.warning(
                            "No machine specs found",
                            extra=get_extra_info(default_extra),
                        )
                        return None

                    machine_spec = json.loads(machine_specs[0].strip())
                    logger.info(
                        f"Machine spec scraped: {machine_spec}",
                        extra=get_extra_info(default_extra),
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
                            f"Max Score({max_score}) or GPU count({gpu_count}) is 0. No need to run job.",
                            extra=get_extra_info(default_extra),
                        )
                        return machine_spec, executor_info

                    logger.info(
                        f"Got GPU specs: {gpu_model} with max score: {max_score}",
                        extra=get_extra_info(default_extra),
                    )

                    executor = await self.executor_dao.get_executor(
                        executor_id=executor_info.uuid, miner_hotkey=miner_info.miner_hotkey
                    )
                    if executor.rented:
                        score = max_score * gpu_count
                        logger.info(
                            "Executor is already rented.",
                            extra=get_extra_info({**default_extra, "score": score}),
                        )
                        await self.task_dao.save(
                            Task(
                                task_status=TaskStatus.Finished,
                                miner_hotkey=miner_info.miner_hotkey,
                                executor_id=executor_info.uuid,
                                proceed_time=0,
                                score=score,
                            )
                        )
                        logger.info(
                            f"Task saved with status Finished for executor({executor_name})",
                            extra=get_extra_info(default_extra),
                        )
                        return machine_spec, executor_info

                    logger.info(
                        "Creating task for executor",
                        extra=get_extra_info(default_extra),
                    )
                    task = await self.task_dao.save(
                        Task(
                            task_status=TaskStatus.SSHConnected,
                            miner_hotkey=miner_info.miner_hotkey,
                            executor_id=executor_info.uuid,
                        )
                    )
                    logger.info(
                        "Task saved with status SSHConnected for executor",
                        extra=get_extra_info(default_extra),
                    )

                    timestamp = int(time.time())
                    local_file_path = str(Path(__file__).parent / ".." / "miner_jobs/score.py")
                    remote_file_path = f"{executor_info.root_dir}/temp/job_{timestamp}.py"

                    await sftp_client.put(local_file_path, remote_file_path)
                    logger.info(
                        f"Uploaded score script to {remote_file_path}",
                        extra=get_extra_info(default_extra),
                    )

                    start_time = time.time()

                    results, err = await self._run_task(ssh_client, executor_info, remote_file_path)
                    if not results:
                        logger.warning(
                            "No result from training job task.", extra=get_extra_info(default_extra)
                        )
                        return None

                    end_time = time.time()
                    logger.info(
                        f"Results from training job task: {results}",
                        extra=get_extra_info(default_extra),
                    )

                    if err is not None:
                        logger.error(
                            f"Error executing task on executor: {err}",
                            extra=get_extra_info(default_extra),
                        )

                        # mark task is failed
                        await self.task_dao.update(
                            uuid=task.uuid,
                            task_status=TaskStatus.Failed,
                            score=0,
                        )
                        logger.debug(
                            "Task marked as failed for executor",
                            extra=get_extra_info(default_extra),
                        )
                    else:
                        job_taken_time = results[-1]
                        try:
                            job_taken_time = float(job_taken_time.strip())
                        except Exception:
                            job_taken_time = end_time - start_time

                        logger.info(
                            "Job taken time for executor",
                            extra=get_extra_info(
                                {**default_extra, "job_taken_time": job_taken_time}
                            ),
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

                        # update task with results
                        await self.task_dao.update(
                            uuid=task.uuid,
                            task_status=TaskStatus.Finished,
                            proceed_time=job_taken_time,
                            score=score,
                        )
                    logger.info(
                        "SSH connection closed for executor",
                        extra=get_extra_info(default_extra),
                    )

                    return machine_spec, executor_info
        except Exception as e:
            logger.error(
                "Error creating task for executor",
                extra=get_extra_info({**default_extra, "error": str(e)}),
                exc_info=True,
            )
            return None

    async def _run_task(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        executor_info: ExecutorSSHInfo,
        remote_file_path: str,
    ) -> tuple[list[str] | None, str | None]:
        try:
            executor_name = f"{executor_info.uuid}_{executor_info.address}_{executor_info.port}"
            default_extra = {
                "executor_uuid": executor_info.uuid,
                "executor_ip_address": executor_info.address,
                "executor_port": executor_info.port,
                "remote_file_path": remote_file_path,
            }
            context.set(f"[_run_task][{executor_name}]")
            logger.info(
                "Running task for executor",
                extra=get_extra_info({**default_extra, "remote_file_path": remote_file_path}),
            )
            result = await ssh_client.run(
                f"export PYTHONPATH={executor_info.root_dir}:$PYTHONPATH && {executor_info.python_path} {remote_file_path}",
                timeout=JOB_LENGTH,
            )
            results = result.stdout.splitlines()
            errors = result.stderr.splitlines()
            logger.info(
                "Run task results", extra=get_extra_info({**default_extra, "results": results})
            )
            logger.warning(
                "Run task errors", extra=get_extra_info({**default_extra, "errors": errors})
            )

            actual_errors = [error for error in errors if "warnning" not in error.lower()]

            if len(results) == 0 and len(actual_errors) > 0:
                logger.error("Failed to execute command!", extra=get_extra_info(default_extra))
                raise Exception("Failed to execute command!")

            #  remove remote_file
            await ssh_client.run(f"rm {remote_file_path}", timeout=30)

            logger.info("Run task success", extra=get_extra_info(default_extra))
            return results, None
        except Exception as e:
            logger.error(
                "Run task error to executor",
                extra=get_extra_info(default_extra),
                exc_info=True,
            )

            #  remove remote_file
            try:
                logger.info(
                    "Removing remote file",
                    extra=get_extra_info(default_extra),
                )
                await asyncio.wait_for(ssh_client.run(f"rm {remote_file_path}"), timeout=10)
                logger.info(
                    "Removed remote file",
                    extra=get_extra_info(default_extra),
                )
            except Exception:
                logger.error(
                    "Failed to remove remote file",
                    extra=get_extra_info(default_extra),
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
