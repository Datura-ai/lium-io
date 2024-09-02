import asyncio
import logging
from typing import Annotated

from pydantic import BaseModel

import bittensor
from clients.miner_client import MinerClient
from datura.requests.miner_requests import (
    AcceptSSHKeyRequest,
    FailedRequest,
    DeclineJobRequest,
)
from datura.requests.validator_requests import SSHPubKeySubmitRequest, SSHPubKeyRemoveRequest
from fastapi import Depends

from core.config import settings
from services.ssh_service import SSHService
from services.task_service import TaskService
from services.docker_service import DockerService, ResourceType

logger = logging.getLogger(__name__)


JOB_LENGTH = 300

class MinerJobRequestPayload(BaseModel):
    miner_hotkey: str
    miner_address: str
    miner_port: int

class MinerResourceRequestPayload(BaseModel):
    miner_hotkey: str
    miner_address: str
    miner_port: int
    docker_image: str
    public_key: str
    resources: ResourceType

class MinerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        task_service: Annotated[TaskService, Depends(TaskService)],
    ):
        self.ssh_service = ssh_service
        self.task_service = task_service
        self.docker_service = DockerService(
            ssh_service=ssh_service
        )
        
    async def request_job_to_miner(self, payload: MinerJobRequestPayload):
        loop = asyncio.get_event_loop()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()

        miner_client = MinerClient(
            loop=loop,
            miner_address=payload.miner_address,
            miner_port=payload.miner_port,
            miner_hotkey=payload.miner_hotkey,
            my_hotkey=my_key.ss58_address,
            keypair=my_key,
            miner_url=f"ws://{payload.miner_address}:{payload.miner_port}/jobs/{my_key.ss58_address}"
        )

        async with miner_client:
            # generate ssh key and send it to miner
            private_key, public_key = self.ssh_service.generate_ssh_key(my_key.ss58_address)
            await miner_client.send_model(SSHPubKeySubmitRequest(public_key=public_key))

            logger.info("Sent SSH key to miner %s", miner_client.miner_name)

            try:
                msg = await asyncio.wait_for(
                    miner_client.job_state.miner_accepted_ssh_key_or_failed_future, JOB_LENGTH
                )
            except TimeoutError:
                msg = None

            if isinstance(msg, AcceptSSHKeyRequest):
                logger.info(f"Miner {miner_client.miner_name} accepted SSH key: {msg}")

                machine_specs = await self.task_service.create_task(
                    miner_address=payload.miner_address,
                    miner_hotkey=payload.miner_hotkey,
                    msg=msg,
                    keypair=my_key,
                    private_key=private_key.decode('utf-8')
                )
                logger.info(f"miner {payload.miner_hotkey} machine specs => {machine_specs}")
                
                await miner_client.send_model(SSHPubKeyRemoveRequest(public_key=public_key))
            elif isinstance(msg, FailedRequest):
                logger.info(f"Miner {miner_client.miner_name} failed job: {msg}")
                return
            elif isinstance(msg, DeclineJobRequest):
                logger.info(f"Miner {miner_client.miner_name} job declined: {msg}")
                return
            else:
                raise ValueError(f"Unexpected msg: {msg}")
            
    async def request_resource_to_miner(self, payload: MinerResourceRequestPayload):
        loop = asyncio.get_event_loop()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()

        miner_client = MinerClient(
            loop=loop,
            miner_address=payload.miner_address,
            miner_port=payload.miner_port,
            miner_hotkey=payload.miner_hotkey,
            my_hotkey=my_key.ss58_address,
            keypair=my_key,
            miner_url=f"ws://{payload.miner_address}:{payload.miner_port}/resources/{my_key.ss58_address}"
        )

        async with miner_client:
            # generate ssh key and send it to miner
            private_key, public_key = self.ssh_service.generate_ssh_key(my_key.ss58_address)
            await miner_client.send_model(SSHPubKeySubmitRequest(public_key=public_key))

            logger.info("Sent SSH key to miner %s", miner_client.miner_name)

            try:
                msg = await asyncio.wait_for(
                    miner_client.job_state.miner_accepted_ssh_key_or_failed_future, JOB_LENGTH
                )
            except TimeoutError:
                msg = None

            if isinstance(msg, AcceptSSHKeyRequest):
                logger.info(f"Miner {miner_client.miner_name} accepted SSH key: {msg}")

                await self.docker_service.create_docker(
                    miner_address=payload.miner_address,
                    miner_hotkey=payload.miner_hotkey,
                    msg=msg,
                    keypair=my_key,
                    private_key=private_key.decode('utf-8'),
                    docker_image=payload.docker_image,
                    public_key=payload.public_key,
                    resources=payload.resources,
                )
                await miner_client.send_model(SSHPubKeyRemoveRequest(public_key=public_key))
            elif isinstance(msg, FailedRequest):
                logger.info(f"Miner {miner_client.miner_name} failed job: {msg}")
                return
            else:
                raise ValueError(f"Unexpected msg: {msg}")
    
MinerServiceDep = Annotated[MinerService, Depends(MinerService)]
