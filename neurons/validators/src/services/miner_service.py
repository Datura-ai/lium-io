import asyncio
import json
import logging
from typing import Annotated

import bittensor
import redis.asyncio as aioredis
from clients.miner_client import MinerClient
from datura.requests.miner_requests import (
    AcceptSSHKeyRequest,
    DeclineJobRequest,
    ExecutorSSHInfo,
    FailedRequest,
)
from datura.requests.validator_requests import SSHPubKeyRemoveRequest, SSHPubKeySubmitRequest
from fastapi import Depends
from payload_models.payloads import (
    ContainerBaseRequest,
    ContainerCreated,
    ContainerCreateRequest,
    ContainerDeleted,
    ContainerDeleteRequest,
    ContainerStarted,
    ContainerStartRequest,
    ContainerStopped,
    ContainerStopRequest,
    FailedContainerRequest,
    MinerJobRequestPayload,
)

from core.config import settings
from daos.executor import ExecutorDao
from services.docker_service import DockerService
from services.ssh_service import SSHService
from services.task_service import TaskService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


JOB_LENGTH = 300


class MinerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        task_service: Annotated[TaskService, Depends(TaskService)],
        docker_service: Annotated[DockerService, Depends(DockerService)],
        executor_dao: Annotated[ExecutorDao, Depends(ExecutorDao)],
    ):
        self.ssh_service = ssh_service
        self.task_service = task_service
        self.docker_service = docker_service
        self.executor_dao = executor_dao
        self.redis = aioredis.from_url(f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}")

    async def request_job_to_miner(self, payload: MinerJobRequestPayload):
        loop = asyncio.get_event_loop()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        miner_name = f"{payload.miner_hotkey}_{payload.miner_address}_{payload.miner_port}"

        try:
            logger.info("[_request_job_to_miner] Requesting job to miner(%s)", miner_name)

            miner_client = MinerClient(
                loop=loop,
                miner_address=payload.miner_address,
                miner_port=payload.miner_port,
                miner_hotkey=payload.miner_hotkey,
                my_hotkey=my_key.ss58_address,
                keypair=my_key,
                miner_url=f"ws://{payload.miner_address}:{payload.miner_port}/jobs/{my_key.ss58_address}",
            )

            async with miner_client:
                # generate ssh key and send it to miner
                private_key, public_key = self.ssh_service.generate_ssh_key(my_key.ss58_address)

                logger.info(
                    f"[_request_job_to_miner] Sending SSHPubKeySubmitRequest to miner({miner_client.miner_name})"
                )

                await miner_client.send_model(SSHPubKeySubmitRequest(public_key=public_key))

                logger.info(
                    "[_request_job_to_miner] Sent SSH key to miner %s", miner_client.miner_name
                )

                try:
                    msg = await asyncio.wait_for(
                        miner_client.job_state.miner_accepted_ssh_key_or_failed_future, JOB_LENGTH
                    )
                except TimeoutError:
                    msg = None

                if isinstance(msg, AcceptSSHKeyRequest):
                    logger.info(f"[_request_job_to_miner] Received AcceptSSHKeyRequest: {msg}")

                    tasks = [
                        asyncio.create_task(
                            self.task_service.create_task(
                                miner_info=payload,
                                executor_info=executor_info,
                                keypair=my_key,
                                private_key=private_key.decode("utf-8"),
                            )
                        )
                        for executor_info in msg.executors
                    ]

                    results = [
                        result
                        for result in await asyncio.gather(*tasks, return_exceptions=True)
                        if result
                    ]
                    logger.info(
                        f"[_request_job_to_miner] Miner({miner_name}) machine specs: {results}"
                    )
                    await self.publish_machine_specs(results, miner_client.miner_hotkey)
                    await miner_client.send_model(SSHPubKeyRemoveRequest(public_key=public_key))
                    logger.info(
                        f"[_request_job_to_miner][finished] Requesting job success for miner({miner_name})"
                    )
                elif isinstance(msg, FailedRequest):
                    logger.warning(
                        f"[_request_job_to_miner][finished] Requesting job failed for miner({miner_name}): {msg}"
                    )
                    return
                elif isinstance(msg, DeclineJobRequest):
                    logger.warning(
                        f"[_request_job_to_miner][finished] Requesting job declined for miner({miner_name}): {msg}"
                    )
                    return
                else:
                    raise ValueError(
                        f"Requesting job to miner({miner_name}): Unexpected msg: {msg}"
                    )
        except asyncio.CancelledError:
            logger.info(
                f"[_request_job_to_miner][finished] Requesting job to miner({miner_name}) was cancelled"
            )
            return
        except Exception as e:
            logger.error(
                f"[_request_job_to_miner][finished] Requesting job to miner({miner_name}) resulted in an exception: {e}"
            )
            return

    async def publish_machine_specs(
        self, results: list[tuple[dict, ExecutorSSHInfo]], miner_hotkey: str
    ):
        """Publish machine specs to compute app connector process"""
        logger.info(f"Publishing machine specs to compute app connector process: {results}")
        for specs, ssh_info in results:
            try:
                await self.redis.publish(
                    "channel:1",
                    json.dumps(
                        {
                            "specs": specs,
                            "miner_hotkey": miner_hotkey,
                            "executor_uuid": ssh_info.uuid,
                            "executor_ip": ssh_info.address,
                            "executor_port": ssh_info.port,
                        }
                    ),
                )
            except Exception as e:
                logger.error(
                    f"Error publishing machine specs of {miner_hotkey} to compute app connector process: {e}"
                )

    async def handle_container(self, payload: ContainerBaseRequest):
        loop = asyncio.get_event_loop()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        executor_id = payload.executor_id

        miner_client = MinerClient(
            loop=loop,
            miner_address=payload.miner_address,
            miner_port=payload.miner_port,
            miner_hotkey=payload.miner_hotkey,
            my_hotkey=my_key.ss58_address,
            keypair=my_key,
            miner_url=f"ws://{payload.miner_address}:{payload.miner_port}/resources/{my_key.ss58_address}",
        )

        async with miner_client:
            # generate ssh key and send it to miner
            private_key, public_key = self.ssh_service.generate_ssh_key(my_key.ss58_address)
            await miner_client.send_model(
                SSHPubKeySubmitRequest(public_key=public_key, executor_id=payload.executor_id)
            )

            logger.info(
                f"[handle_container][{executor_id}] Sent SSH key to miner {miner_client.miner_name}"
            )

            try:
                msg = await asyncio.wait_for(
                    miner_client.job_state.miner_accepted_ssh_key_or_failed_future, JOB_LENGTH
                )
            except TimeoutError:
                msg = None

            if isinstance(msg, AcceptSSHKeyRequest):
                logger.info(
                    f"[handle_container][{executor_id}] Miner {miner_client.miner_name} accepted SSH key: {msg}"
                )

                try:
                    executor = msg.executors[0]
                except Exception as e:
                    logger.error(
                        f"[handle_container][{executor_id}] Error: Miner didn't return executor info: {e}"
                    )
                    executor = None

                if executor is None or executor.uuid != payload.executor_id:
                    logger.error(
                        f"[handle_container][{executor_id}] Error: Invalid executor id {payload.executor_id}"
                    )
                    await miner_client.send_model(
                        SSHPubKeyRemoveRequest(
                            public_key=public_key, executor_id=payload.executor_id
                        )
                    )

                    self.executor_dao.unrent(payload.executor_id, payload.miner_hotkey)

                    return FailedContainerRequest(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        msg=f"Invalid executor id {payload.executor_id}",
                    )

                try:
                    if isinstance(payload, ContainerCreateRequest):
                        result = await self.docker_service.create_container(
                            payload,
                            executor,
                            my_key,
                            private_key.decode("utf-8"),
                        )
                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key, executor_id=payload.executor_id
                            )
                        )

                        return ContainerCreated(
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            container_name=result.container_name,
                            volume_name=result.volume_name,
                            port_maps=result.port_maps,
                        )
                    elif isinstance(payload, ContainerStartRequest):
                        await asyncio.to_thread(
                            self.docker_service.start_container,
                            payload,
                            executor,
                            my_key,
                            private_key.decode("utf-8"),
                        )
                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key, executor_id=payload.executor_id
                            )
                        )

                        return ContainerStarted(
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            container_name=payload.container_name,
                        )
                    elif isinstance(payload, ContainerStopRequest):
                        await asyncio.to_thread(
                            self.docker_service.stop_container,
                            payload,
                            executor,
                            my_key,
                            private_key.decode("utf-8"),
                        )
                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key, executor_id=payload.executor_id
                            )
                        )

                        return ContainerStopped(
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            container_name=payload.container_name,
                        )
                    elif isinstance(payload, ContainerDeleteRequest):
                        await asyncio.to_thread(
                            self.docker_service.delete_container,
                            payload,
                            executor,
                            my_key,
                            private_key.decode("utf-8"),
                        )
                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key, executor_id=payload.executor_id
                            )
                        )

                        return ContainerDeleted(
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            container_name=payload.container_name,
                            volume_name=payload.volume_name,
                        )
                    else:
                        logger.info(f"Unexpected request: {payload}")
                        return FailedContainerRequest(
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            msg=f"Unexpected request: {payload}",
                        )

                except Exception as e:
                    logger.error("[handle_container] Error: create container error: %s", str(e))
                    await miner_client.send_model(
                        SSHPubKeyRemoveRequest(
                            public_key=public_key, executor_id=payload.executor_id
                        )
                    )

                    return FailedContainerRequest(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        msg=f"create container error: {str(e)}",
                    )

            elif isinstance(msg, FailedRequest):
                logger.info(
                    f"[handle_container] Error: Miner {miner_client.miner_name} failed job: {msg}"
                )
                return FailedContainerRequest(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    msg=f"create container error: {str(msg)}",
                )
            else:
                logger.info(f"[handle_container] Error: Unexpected msg: {msg}")
                return FailedContainerRequest(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    msg=f"Unexpected msg: {msg}",
                )


MinerServiceDep = Annotated[MinerService, Depends(MinerService)]
