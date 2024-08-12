from typing import Annotated, Tuple, List, Optional
from pathlib import Path
import io
import logging
import time
from asgiref.sync import sync_to_async

import bittensor
from daos.task import TaskDao
from fastapi import Depends

from core.config import settings
from services.ssh_service import SSHService
from paramiko import SSHClient, AutoAddPolicy, Ed25519Key 

from datura.requests.miner_requests import AcceptSSHKeyRequest

logger = logging.getLogger(__name__)

JOB_LENGTH = 300

class TaskService:
    def __init__(
        self,
        task_dao: Annotated[TaskDao, Depends(TaskDao)],
        ssh_service: Annotated[SSHService, Depends(SSHService)],
    ):
        self.task_dao = task_dao
        self.ssh_service = ssh_service

    async def create_task(
        self,
        miner_address: str,
        msg: AcceptSSHKeyRequest,
        keypair: bittensor.Keypair,
        private_key: str
    ):
        # self.task_dao.save(
        #     Task(
        #         task_status=TaskStatus.SSHConnected,
        #         miner_hotkey=payload.miner_hotkey,
        #         ssh_private_key=private_key.decode(),
        #     )
        # )
        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = Ed25519Key.from_private_key(io.StringIO(private_key))

        ssh_client = SSHClient()
        ssh_client.set_missing_host_key_policy(AutoAddPolicy())
        ssh_client.connect(hostname=miner_address, username=msg.ssh_username, look_for_keys=False, pkey=pkey)
        ssh_client.exec_command(f"mkdir -p {msg.root_dir}/temp")

        timestamp = int(time.time())
        local_file_path = str(Path(__file__).parent / ".." / "job.py")
        remote_file_path = f"{msg.root_dir}/temp/job_{timestamp}.py"

        ftp_client=ssh_client.open_sftp()
        ftp_client.put(local_file_path, remote_file_path)
        ftp_client.close()
        
        start_time = time.time()
        restuls, err = await sync_to_async(self._run_task)(ssh_client, msg, remote_file_path)
        end_time = time.time()
        
        if err is not None:
            # mark task is failed
            
            return

        job_taken_time = restuls[-1]
        try:
            job_taken_time = int(job_taken_time)
        except:
            job_taken_time = end_time - start_time
        
        
        ssh_client.close()
        
        # update task with results
        # self.task_dao.save(
        #     Task(
        #         task_status=TaskStatus.SSHConnected,
        #         miner_hotkey=payload.miner_hotkey,
        #         ssh_private_key=private_key.decode(),
        #     )
        # )
        
    def _run_task(
        self,
        ssh_client: SSHClient,
        msg: AcceptSSHKeyRequest,
        remote_file_path: str
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        try:
            _, stdout, stderr = ssh_client.exec_command(f"export PYTHONPATH={msg.root_dir} && {msg.python_path} {remote_file_path}", timeout=JOB_LENGTH)
            results = stdout.readlines()
            errors = stderr.readlines()
            if (len(errors) > 0):
                print(errors)
                raise Exception("Failed to execute command!")

            #  remove remote_file
            ssh_client.exec_command(f"rm {remote_file_path}")
        
            return results, None
        except Exception as e:
            logger.error('ssh connection error: %s', str(e))
            
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
