from typing import Annotated, Tuple, List, Optional
from pathlib import Path
import io
import logging
import asyncio
from asgiref.sync import sync_to_async

import bittensor
from daos.task import TaskDao
from fastapi import Depends

from core.config import settings
from services.ssh_service import SSHService
from paramiko import SSHClient, AutoAddPolicy, Ed25519Key 

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
        ssh_username: str,
        keypair: bittensor.Keypair,
        private_key: str,
        public_key: str
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
        ssh_client.connect(hostname=miner_address, username=ssh_username, look_for_keys=False, pkey=pkey)

        localFilePath = str(Path(__file__).parent / ".." / "test.py")
        remoteFilePath = f"/home/{ssh_username}/test.py"

        ftp_client=ssh_client.open_sftp()
        ftp_client.put(localFilePath, remoteFilePath)
        ftp_client.close()
        
        restuls, err = await sync_to_async(self._run_task)(ssh_client, remoteFilePath)
        print('restuls ===>', restuls)
        #  remove public_key
        
        
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
        remoteFilePath: str
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        try:
            _, stdout, stderr = ssh_client.exec_command(f"python3 {remoteFilePath}", timeout=JOB_LENGTH)
            results = stdout.readlines()
            errors = stderr.readlines()
            if (len(errors) > 0):
                print(errors)
                raise Exception("Failed to execute command!")

            return results, None
        except Exception as e:
            logger.error('ssh connection error', e)
            return None, str(e)

    def get_decrypted_private_key_for_task(self, uuid: str) -> str | None:
        task = self.task_dao.get_task_by_uuid(uuid)
        if task is None:
            return None
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        return self.ssh_service.decrypt_payload(my_key.ss58_address, task.ssh_private_key)


TaskServiceDep = Annotated[TaskService, Depends(TaskService)]
