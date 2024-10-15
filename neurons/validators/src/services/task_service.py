import asyncio
import io
import json
import logging
import time
from pathlib import Path
from typing import Annotated

import bittensor
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from paramiko import AutoAddPolicy, Ed25519Key, SSHClient
from payload_models.payloads import MinerJobRequestPayload

from core.config import settings
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

logger = logging.getLogger(__name__)

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
        try:
            logger.info(
                f"[create_task] Creating task for executor({executor_name}): Upsert executor uuid: {executor_info.uuid}"
            )
            self.executor_dao.upsert(
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

            logger.info(
                f"[create_task] Creating task for executor({executor_name}): Connecting ssh with info: {executor_info.address}:{executor_info.ssh_port}"
            )
            private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
            logger.debug(f"[create_task] Decrypted private key for executor({executor_name})")
            pkey = Ed25519Key.from_private_key(io.StringIO(private_key))

            ssh_client = SSHClient()
            ssh_client.set_missing_host_key_policy(AutoAddPolicy())
            ssh_client.connect(
                hostname=executor_info.address,
                username=executor_info.ssh_username,
                look_for_keys=False,
                pkey=pkey,
                port=executor_info.ssh_port,
            )
            logger.info(f"[create_task] SSH connection established with executor({executor_name})")
            ssh_client.exec_command(f"mkdir -p {executor_info.root_dir}/temp")
            logger.debug(f"[create_task] Created temporary directory on executor({executor_name})")

            # run synthetic job
            ftp_client = ssh_client.open_sftp()
            logger.debug(f"[create_task] Opened SFTP client for executor({executor_name})")

            # get machine specs
            timestamp = int(time.time())
            local_file_path = str(Path(__file__).parent / ".." / "miner_jobs/machine_scrape.py")
            remote_file_path = f"{executor_info.root_dir}/temp/job_{timestamp}.py"

            ftp_client.put(local_file_path, remote_file_path)
            logger.info(
                f"[create_task] Uploaded machine scrape script to executor({executor_name})"
            )

            machine_specs, _ = await asyncio.to_thread(
                self._run_task, ssh_client, executor_info, remote_file_path
            )
            machine_spec = json.loads(machine_specs[0].strip())
            logger.info(
                f"[create_task] Machine spec -> executor: {executor_info.address}, spec: {machine_spec}"
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

            logger.info(
                f"[create_task] Max Score -> executor: {executor_info.address}, gpu model: {gpu_model}, max score: {max_score}"
            )

            executor = self.executor_dao.get_executor(executor_info.uuid, miner_info.miner_hotkey)
            if executor.rented:
                score = max_score * gpu_count
                logger.info(
                    f"[create_task] Executor({executor_name}) is already rented. Give score: {score}"
                )
                self.task_dao.save(
                    Task(
                        task_status=TaskStatus.Finished,
                        miner_hotkey=miner_info.miner_hotkey,
                        executor_id=executor_info.uuid,
                        proceed_time=0,
                        score=score,
                    )
                )
                return machine_spec, executor_info

            logger.info(
                f"[create_task] Create Task -> executor: {executor_info.address}, executor uuid:{executor_info.uuid}, miner_hotkey: {miner_info.miner_hotkey}"
            )
            task = self.task_dao.save(
                Task(
                    task_status=TaskStatus.SSHConnected,
                    miner_hotkey=miner_info.miner_hotkey,
                    executor_id=executor_info.uuid,
                )
            )
            logger.debug(
                f"[create_task] Task saved with status SSHConnected for executor({executor_name})"
            )

            timestamp = int(time.time())
            local_file_path = str(Path(__file__).parent / ".." / "miner_jobs/score.py")
            remote_file_path = f"{executor_info.root_dir}/temp/job_{timestamp}.py"

            ftp_client.put(local_file_path, remote_file_path)
            logger.info(f"[create_task] Uploaded score script to executor({executor_name})")

            start_time = time.time()
            try:
                results, err = await asyncio.wait_for(
                    asyncio.to_thread(self._run_task, ssh_client, executor_info, remote_file_path),
                    timeout=60 * 5,
                )
            except TimeoutError:
                logger.error(
                    f"[create_task] Task execution on executor({executor_info.address}) timed out."
                )
                return None

            end_time = time.time()
            logger.info(
                f"[create_task] Task results -> executor: {executor_info.address}, result: {results}"
            )

            if err is not None:
                logger.error(
                    f"[create_task] Error executing task on executor({executor_name}): {err}"
                )

                # mark task is failed
                self.task_dao.update(
                    uuid=task.uuid,
                    task_status=TaskStatus.Failed,
                    score=0,
                )
                logger.debug(f"[create_task] Task marked as failed for executor({executor_name})")
            else:
                job_taken_time = results[-1]
                try:
                    job_taken_time = float(job_taken_time.strip())
                except Exception:
                    job_taken_time = end_time - start_time

                logger.info(
                    f"[create_task] Job taken time for executor({executor_name}): {job_taken_time}"
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
                    "[create_task] Give score(%f) to miner(%s) for the task(%s).",
                    score,
                    miner_info.miner_hotkey,
                    str(task.uuid),
                )

                # update task with results
                self.task_dao.update(
                    uuid=task.uuid,
                    task_status=TaskStatus.Finished,
                    proceed_time=job_taken_time,
                    score=score,
                )
                logger.debug(
                    f"[create_task] Task updated with final score for executor({executor_name})"
                )

            ftp_client.close()
            logger.debug(f"[create_task] SFTP client closed for executor({executor_name})")
            ssh_client.close()
            logger.info(f"[create_task] SSH connection closed for executor({executor_name})")

            return machine_spec, executor_info
        except Exception as e:
            logger.error(f"[create_task] Error creating task for executor({executor_name}): {e}")
            return None

    def _run_task(
        self, ssh_client: SSHClient, executor_info: ExecutorSSHInfo, remote_file_path: str
    ) -> tuple[list[str] | None, str | None]:
        try:
            logger.info(
                "run task -> executor(%s:%d)", executor_info.address, executor_info.ssh_port
            )
            _, stdout, stderr = ssh_client.exec_command(
                f"export PYTHONPATH={executor_info.root_dir}:$PYTHONPATH && {executor_info.python_path} {remote_file_path}",
                timeout=JOB_LENGTH,
            )
            results = stdout.readlines()
            errors = stderr.readlines()

            actual_errors = [error for error in errors if "warnning" not in error.lower()]

            if len(results) == 0 and len(actual_errors) > 0:
                logger.error(f"{actual_errors}")
                raise Exception("Failed to execute command!")

            #  remove remote_file
            ssh_client.exec_command(f"rm {remote_file_path}")

            logger.info(
                "run task success -> executor(%s:%d)", executor_info.address, executor_info.ssh_port
            )
            return results, None
        except Exception as e:
            logger.error(
                "run task error to executor(%s:%d): %s",
                executor_info.address,
                executor_info.ssh_port,
                str(e),
            )

            #  remove remote_file
            ssh_client.exec_command(f"rm {remote_file_path}")

            return None, str(e)

    def get_decrypted_private_key_for_task(self, uuid: str) -> str | None:
        task = self.task_dao.get_task_by_uuid(uuid)
        if task is None:
            return None
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        return self.ssh_service.decrypt_payload(my_key.ss58_address, task.ssh_private_key)


TaskServiceDep = Annotated[TaskService, Depends(TaskService)]
