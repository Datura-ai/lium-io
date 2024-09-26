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
from services.ssh_service import SSHService
from services.const import GPU_MAX_SCORES, MIN_JOB_TAKEN_TIME

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
        logger.info(
            f"Upsert executor -> miner_address: {miner_info.miner_address}, executor uuid: {executor_info.uuid}"
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

        logger.info("Connect ssh")
        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
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
        ssh_client.exec_command(f"mkdir -p {executor_info.root_dir}/temp")

        # run synthetic job
        ftp_client = ssh_client.open_sftp()

        # get machine specs
        timestamp = int(time.time())
        local_file_path = str(Path(__file__).parent / ".." / "miner_jobs/machine_scrape.py")
        remote_file_path = f"{executor_info.root_dir}/temp/job_{timestamp}.py"

        ftp_client.put(local_file_path, remote_file_path)

        machine_specs, _ = await asyncio.to_thread(
            self._run_task, ssh_client, executor_info, remote_file_path
        )
        machine_spec = json.loads(machine_specs[0].strip())
        logger.info(f'machine spec: {machine_spec}')
        
        gpu_model = None
        if machine_spec.get("gpu", {}).get("count", 0) > 0:
            details = machine_spec["gpu"].get("details", [])
            if len(details) > 0:
                gpu_model = details[0].get("name", None)
                
        max_score = 0
        if gpu_model:
            max_score = GPU_MAX_SCORES.get(gpu_model, 0)
            
        logger.info(f'gpu model: {gpu_model}, max score: {max_score}')

        executor = self.executor_dao.get_executor(executor_info.uuid, miner_info.miner_hotkey)
        if executor.rented:
            self.task_dao.save(
                Task(
                    task_status=TaskStatus.Finished,
                    miner_hotkey=miner_info.miner_hotkey,
                    executor_id=executor_info.uuid,
                    proceed_time=0,
                    score=max_score
                )
            )
            return machine_spec, executor_info

        logger.info(
            f"Create Task -> miner_address: {miner_info.miner_address}, miner_hotkey: {miner_info.miner_hotkey}"
        )
        task = self.task_dao.save(
            Task(
                task_status=TaskStatus.SSHConnected,
                miner_hotkey=miner_info.miner_hotkey,
                executor_id=executor_info.uuid,
            )
        )

        timestamp = int(time.time())
        local_file_path = str(Path(__file__).parent / ".." / "miner_jobs/score.py")
        remote_file_path = f"{executor_info.root_dir}/temp/job_{timestamp}.py"

        ftp_client.put(local_file_path, remote_file_path)

        start_time = time.time()
        # results, err = await sync_to_async(self._run_task)(ssh_client, msg, remote_file_path)
        results, err = await asyncio.to_thread(
            self._run_task, ssh_client, executor_info, remote_file_path
        )
        end_time = time.time()
        logger.info(f"results: {results}")

        if err is not None:
            logger.error(f"error: {err}")

            # mark task is failed
            self.task_dao.update(
                uuid=task.uuid,
                task_status=TaskStatus.Failed,
                score=0,
            )
        else:
            job_taken_time = results[-1]
            try:
                job_taken_time = float(job_taken_time.strip())
            except Exception:
                job_taken_time = end_time - start_time

            logger.info(f"job_taken_time: {job_taken_time}")

            # update task with results
            self.task_dao.update(
                uuid=task.uuid,
                task_status=TaskStatus.Finished,
                proceed_time=job_taken_time,
                score=max_score * min(MIN_JOB_TAKEN_TIME/job_taken_time, 1) if job_taken_time > 0 else 0,
            )

        ftp_client.close()
        ssh_client.close()

        return machine_spec, executor_info

    def _run_task(
        self, ssh_client: SSHClient, executor_info: ExecutorSSHInfo, remote_file_path: str
    ) -> tuple[list[str] | None, str | None]:
        try:
            _, stdout, stderr = ssh_client.exec_command(
                f"export PYTHONPATH={executor_info.root_dir} && {executor_info.python_path} {remote_file_path}",
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

            return results, None
        except Exception as e:
            logger.error("ssh connection error: %s", str(e))

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
