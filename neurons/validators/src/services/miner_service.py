import asyncio
import json
import logging
from typing import Annotated

import bittensor
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
    FailedContainerErrorCodes,
    FailedContainerRequest,
    MinerJobEnryptedFiles,
    MinerJobRequestPayload,
)
from protocol.vc_protocol.compute_requests import RentedMachine

from core.config import settings
from core.utils import _m, get_extra_info
from services.docker_service import DockerService
from services.redis_service import EXECUTOR_COUNT_PREFIX, MACHINE_SPEC_CHANNEL_NAME, RedisService
from services.ssh_service import SSHService
from services.task_service import TaskService

logger = logging.getLogger(__name__)


JOB_LENGTH = 300


class MinerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        task_service: Annotated[TaskService, Depends(TaskService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
    ):
        self.ssh_service = ssh_service
        self.task_service = task_service
        self.redis_service = redis_service

    async def request_job_to_miner(
        self,
        payload: MinerJobRequestPayload,
        encypted_files: MinerJobEnryptedFiles,
        docker_hub_digests: dict[str, str],
    ):
        loop = asyncio.get_event_loop()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "job_batch_id": payload.job_batch_id,
            "miner_hotkey": payload.miner_hotkey,
            "miner_address": payload.miner_address,
            "miner_port": payload.miner_port,
        }

        try:
            logger.info(_m("Requesting job to miner", extra=get_extra_info(default_extra)))
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

                await miner_client.send_model(SSHPubKeySubmitRequest(public_key=public_key))

                try:
                    msg = await asyncio.wait_for(
                        miner_client.job_state.miner_accepted_ssh_key_or_failed_future, JOB_LENGTH
                    )
                except TimeoutError:
                    logger.error(
                        _m(
                            "Waiting accepted ssh key or failed request from miner resulted in TimeoutError",
                            extra=get_extra_info(default_extra),
                        ),
                    )
                    msg = None
                except Exception:
                    logger.error(
                        _m(
                            "Waiting accepted ssh key or failed request from miner resulted in an exception",
                            extra=get_extra_info(default_extra),
                        ),
                    )
                    msg = None

                if isinstance(msg, AcceptSSHKeyRequest):
                    logger.info(
                        _m(
                            "Received AcceptSSHKeyRequest for miner. Running tasks for executors",
                            extra=get_extra_info(
                                {**default_extra, "executors": len(msg.executors)}
                            ),
                        ),
                    )
                    if len(msg.executors) == 0:
                        return None

                    tasks = [
                        asyncio.create_task(
                            self.task_service.create_task(
                                miner_info=payload,
                                executor_info=executor_info,
                                keypair=my_key,
                                private_key=private_key.decode("utf-8"),
                                public_key=public_key.decode("utf-8"),
                                encypted_files=encypted_files,
                                docker_hub_digests=docker_hub_digests,
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
                        _m(
                            "Finished running tasks for executors",
                            extra=get_extra_info({**default_extra, "executors": len(results)}),
                        ),
                    )

                    await miner_client.send_model(SSHPubKeyRemoveRequest(public_key=public_key))

                    await self.publish_machine_specs(results, miner_client.miner_hotkey)
                    await self.store_executor_counts(
                        payload.miner_hotkey, payload.job_batch_id, len(msg.executors), results
                    )

                    total_score = 0
                    for _, _, score, _, _, _, _ in results:
                        total_score += score

                    logger.info(
                        _m(
                            f"total score: {total_score}",
                            extra=get_extra_info(default_extra),
                        )
                    )

                    return {
                        "miner_hotkey": payload.miner_hotkey,
                        "score": total_score,
                    }
                elif isinstance(msg, FailedRequest):
                    logger.warning(
                        _m(
                            "Requesting job failed for miner",
                            extra=get_extra_info({**default_extra, "msg": str(msg)}),
                        ),
                    )
                    return None
                elif isinstance(msg, DeclineJobRequest):
                    logger.warning(
                        _m(
                            "Requesting job declined for miner",
                            extra=get_extra_info({**default_extra, "msg": str(msg)}),
                        ),
                    )
                    return None
                else:
                    logger.error(
                        _m(
                            "Unexpected msg",
                            extra=get_extra_info({**default_extra, "msg": str(msg)}),
                        ),
                    )
                    return None
        except asyncio.CancelledError:
            logger.error(
                _m("Requesting job to miner was cancelled", extra=get_extra_info(default_extra)),
            )
            return None
        except Exception as e:
            logger.error(
                _m(
                    "Requesting job to miner resulted in an exception",
                    extra=get_extra_info({**default_extra, "error": str(e)}),
                ),
                exc_info=True,
            )
            return None

    async def publish_machine_specs(
        self, results: list[tuple[dict, ExecutorSSHInfo]], miner_hotkey: str
    ):
        """Publish machine specs to compute app connector process"""
        default_extra = {
            "miner_hotkey": miner_hotkey,
        }

        logger.info(
            _m(
                "Publishing machine specs to compute app connector process",
                extra=get_extra_info({**default_extra, "results": len(results)}),
            ),
        )
        for (
            specs,
            ssh_info,
            score,
            synthetic_job_score,
            job_batch_id,
            log_status,
            log_text,
        ) in results:
            try:
                await self.redis_service.publish(
                    MACHINE_SPEC_CHANNEL_NAME,
                    {
                        "specs": specs,
                        "miner_hotkey": miner_hotkey,
                        "executor_uuid": ssh_info.uuid,
                        "executor_ip": ssh_info.address,
                        "executor_port": ssh_info.port,
                        "score": score,
                        "synthetic_job_score": synthetic_job_score,
                        "job_batch_id": job_batch_id,
                        "log_status": log_status,
                        "log_text": str(log_text),
                    },
                )
            except Exception as e:
                logger.error(
                    _m(
                        f"Error publishing machine specs of {miner_hotkey} to compute app connector process",
                        extra=get_extra_info({**default_extra, "error": str(e)}),
                    ),
                    exc_info=True,
                )

    async def store_executor_counts(
        self, miner_hotkey: str, job_batch_id: str, total: int, results: list[dict]
    ):
        default_extra = {
            "job_batch_id": job_batch_id,
            "miner_hotkey": miner_hotkey,
        }

        success = 0
        failed = 0

        for _, _, score, _, _, _, _ in results:
            if score > 0:
                success += 1
            else:
                failed += 1

        data = {"total": total, "success": success, "failed": failed}

        key = f"{EXECUTOR_COUNT_PREFIX}:{miner_hotkey}"

        try:
            await self.redis_service.hset(key, job_batch_id, json.dumps(data))

            logger.info(
                _m(
                    "Stored executor counts",
                    extra=get_extra_info({**default_extra, **data}),
                ),
            )
        except Exception as e:
            logger.error(
                _m(
                    "Failed storing executor counts",
                    extra=get_extra_info({**default_extra, **data, "error": str(e)}),
                ),
                exc_info=True,
            )

    async def handle_container(self, payload: ContainerBaseRequest):
        loop = asyncio.get_event_loop()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_id": payload.executor_id,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
            "container_request_type": str(payload.message_type),
        }

        docker_service = DockerService(
            ssh_service=self.ssh_service,
            redis_service=self.redis_service,
        )

        try:
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
                    _m("Sent SSH key to miner.", extra=get_extra_info(default_extra)),
                )

                try:
                    msg = await asyncio.wait_for(
                        miner_client.job_state.miner_accepted_ssh_key_or_failed_future,
                        timeout=JOB_LENGTH,
                    )
                except TimeoutError:
                    logger.error(
                        _m(
                            "Waiting accepted ssh key or failed request from miner resulted in an timeout error",
                            extra=get_extra_info(default_extra),
                        ),
                    )
                    msg = None
                except Exception as e:
                    logger.error(
                        _m(
                            "Waiting accepted ssh key or failed request from miner resulted in an exception",
                            extra=get_extra_info({**default_extra, "error": str(e)}),
                        ),
                    )
                    msg = None

                if isinstance(msg, AcceptSSHKeyRequest):
                    logger.info(
                        _m(
                            "Received AcceptSSHKeyRequest",
                            extra=get_extra_info({**default_extra, "msg": str(msg)}),
                        ),
                    )

                    try:
                        executor = msg.executors[0]
                    except Exception as e:
                        logger.error(
                            _m(
                                "Error: Miner didn't return executor info",
                                extra=get_extra_info({**default_extra, "error": str(e)}),
                            ),
                        )
                        executor = None

                    if executor is None or executor.uuid != payload.executor_id:
                        logger.error(
                            _m("Error: Invalid executor id", extra=get_extra_info(default_extra)),
                        )
                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key, executor_id=payload.executor_id
                            )
                        )

                        await self.redis_service.remove_rented_machine(
                            RentedMachine(
                                miner_hotkey=payload.miner_hotkey,
                                executor_id=payload.executor_id,
                                executor_ip_address=executor.address if executor else "",
                                executor_ip_port=str(executor.port if executor else ""),
                            )
                        )

                        return FailedContainerRequest(
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            msg=f"Invalid executor id {payload.executor_id}",
                            error_code=FailedContainerErrorCodes.InvalidExecutorId,
                        )

                    try:
                        if isinstance(payload, ContainerCreateRequest):
                            logger.info(
                                _m(
                                    "Creating container",
                                    extra=get_extra_info(
                                        {**default_extra, "payload": str(payload)}
                                    ),
                                ),
                            )
                            result = await docker_service.create_container(
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

                            if isinstance(result, FailedContainerRequest):
                                return result

                            return ContainerCreated(
                                miner_hotkey=payload.miner_hotkey,
                                executor_id=payload.executor_id,
                                container_name=result.container_name,
                                volume_name=result.volume_name,
                                port_maps=result.port_maps,
                            )

                        # elif isinstance(payload, ContainerStartRequest):
                        #     logger.info(
                        #         _m(
                        #             "Starting container",
                        #             extra=get_extra_info(
                        #                 {**default_extra, "payload": str(payload)}
                        #             ),
                        #         ),
                        #     )
                        #     await docker_service.start_container(
                        #         payload,
                        #         executor,
                        #         my_key,
                        #         private_key.decode("utf-8"),
                        #     )

                        #     logger.info(
                        #         _m(
                        #             "Started Container",
                        #             extra=get_extra_info(
                        #                 {**default_extra, "payload": str(payload)}
                        #             ),
                        #         ),
                        #     )
                        #     await miner_client.send_model(
                        #         SSHPubKeyRemoveRequest(
                        #             public_key=public_key, executor_id=payload.executor_id
                        #         )
                        #     )

                        #     return ContainerStarted(
                        #         miner_hotkey=payload.miner_hotkey,
                        #         executor_id=payload.executor_id,
                        #         container_name=payload.container_name,
                        #     )
                        # elif isinstance(payload, ContainerStopRequest):
                        #     await docker_service.stop_container(
                        #         payload,
                        #         executor,
                        #         my_key,
                        #         private_key.decode("utf-8"),
                        #     )
                        #     await miner_client.send_model(
                        #         SSHPubKeyRemoveRequest(
                        #             public_key=public_key, executor_id=payload.executor_id
                        #         )
                        #     )

                        #     return ContainerStopped(
                        #         miner_hotkey=payload.miner_hotkey,
                        #         executor_id=payload.executor_id,
                        #         container_name=payload.container_name,
                        #     )
                        elif isinstance(payload, ContainerDeleteRequest):
                            logger.info(
                                _m(
                                    "Deleting container",
                                    extra=get_extra_info(
                                        {**default_extra, "payload": str(payload)}
                                    ),
                                ),
                            )
                            await docker_service.delete_container(
                                payload,
                                executor,
                                my_key,
                                private_key.decode("utf-8"),
                            )

                            logger.info(
                                _m(
                                    "Deleted Container",
                                    extra=get_extra_info(
                                        {**default_extra, "payload": str(payload)}
                                    ),
                                ),
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
                            logger.error(
                                _m(
                                    "Unexpected request",
                                    extra=get_extra_info(
                                        {**default_extra, "payload": str(payload)}
                                    ),
                                ),
                            )
                            return FailedContainerRequest(
                                miner_hotkey=payload.miner_hotkey,
                                executor_id=payload.executor_id,
                                msg=f"Unexpected request: {payload}",
                                error_code=FailedContainerErrorCodes.UnknownError,
                            )

                    except Exception as e:
                        logger.error(
                            _m(
                                "Error: create container error",
                                extra=get_extra_info({**default_extra, "error": str(e)}),
                            ),
                        )
                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key, executor_id=payload.executor_id
                            )
                        )

                        return FailedContainerRequest(
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            msg=f"create container error: {str(e)}",
                            error_code=FailedContainerErrorCodes.ExceptionError,
                        )

                elif isinstance(msg, FailedRequest):
                    logger.info(
                        _m(
                            "Error: Miner failed job",
                            extra=get_extra_info({**default_extra, "msg": str(msg)}),
                        ),
                    )
                    return FailedContainerRequest(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        msg=f"Failed request from miner: {str(msg)}",
                        error_code=FailedContainerErrorCodes.FailedMsgFromMiner,
                    )
                else:
                    logger.error(
                        _m(
                            "Error: Unexpected msg",
                            extra=get_extra_info({**default_extra, "msg": str(msg)}),
                        ),
                    )
                    return FailedContainerRequest(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        msg=f"Unexpected msg: {str(msg)}",
                        error_code=FailedContainerErrorCodes.UnknownError,
                    )
        except Exception as e:
            log_text = _m(
                "[handle_container] resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )

            logger.error(log_text, exc_info=True)

            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                msg=f"Exception: {str(e)}",
                error_code=FailedContainerErrorCodes.ExceptionError,
            )


MinerServiceDep = Annotated[MinerService, Depends(MinerService)]
