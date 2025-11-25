import asyncio
import json
import logging
import os
import time
from typing import Annotated

import aiohttp
from asyncssh import SSHKey
import asyncssh
import bittensor
from clients.miner_client import MinerClient
from daos.port_mapping_dao import PortMappingDao
from datura.requests.miner_requests import (
    AcceptSSHKeyRequest,
    DeclineJobRequest,
    ExecutorSSHInfo,
    FailedRequest,
    PodLogsResponse,
    RequestType,
)
from datura.requests.validator_requests import (
    SSHPubKeyRemoveRequest,
    SSHPubKeySubmitRequest,
    GetPodLogsRequest,
    AuthenticationPayload,
)
from fastapi import Depends
from clients.validator_portal_api import ValidatorPortalAPI
from payload_models.payloads import (
    BackupContainerRequest,
    RestoreContainerRequest,
    ContainerBaseRequest,
    ContainerCreateRequest,
    ContainerDeleteRequest,
    AddSshPublicKeyRequest,
    RemoveSshPublicKeysRequest,
    FailedContainerErrorCodes,
    FailedContainerErrorTypes,
    FailedContainerRequest,
    MinerJobEnryptedFiles,
    MinerJobRequestPayload,
    GetPodLogsRequestFromServer,
    PodLogsResponseToServer,
    FailedGetPodLogs,
    AddDebugSshKeyRequest,
    DebugSshKeyAdded,
    FailedAddDebugSshKey,
    InstallJupyterServerRequest,
    JupyterServerInstalled,
    JupyterInstallationFailed,
)

from core.config import settings
from core.utils import _m, get_extra_info
from services.docker_service import DockerService
from services.redis_service import MACHINE_SPEC_CHANNEL, RedisService
from services.ssh_service import SSHService
from services.task_service import TaskService, JobResult

logger = logging.getLogger(__name__)


def _get_error_details(error: Exception) -> str:
    """Extract exception details. For RetryError unwraps the underlying exception."""
    last_attempt = getattr(error, 'last_attempt', None)
    if last_attempt:
        last_exc = last_attempt.exception()
        if last_exc:
            return f"RetryError, {type(last_exc).__name__}: {str(last_exc)}"
    return f"{type(error).__name__}: {str(error)}"


def _parse_miner_response(response_data: dict) -> AcceptSSHKeyRequest | FailedRequest | PodLogsResponse:
    """Parse miner REST API response based on message_type field.

    Args:
        response_data: JSON response data from miner

    Returns:
        Parsed response model (AcceptSSHKeyRequest, PodLogsResponse, or FailedRequest)

    Raises:
        ValueError: If message_type is missing or unknown
    """
    message_type_str = response_data.get("message_type")
    if not message_type_str:
        logger.error(
            f"Response missing message_type field. Raw payload: {json.dumps(response_data)}"
        )
        # Treat missing message_type as unexpected response
        return FailedRequest(
            message_type=RequestType.FailedRequest,
            details="Missing message_type in response"
        )

    try:
        message_type = RequestType(message_type_str)
    except ValueError:
        logger.error(
            f"Unknown message_type: {message_type_str}. Raw payload: {json.dumps(response_data)}"
        )
        # Treat unknown message_type as unexpected response
        return FailedRequest(
            message_type=RequestType.FailedRequest,
            details=f"Unknown message_type: {message_type_str}"
        )

    # Dispatch to appropriate model based on message_type
    if message_type == RequestType.AcceptSSHKeyRequest:
        return AcceptSSHKeyRequest.model_validate(response_data)
    elif message_type == RequestType.PodLogsResponse:
        return PodLogsResponse.model_validate(response_data)
    elif message_type == RequestType.FailedRequest:
        return FailedRequest.model_validate(response_data)
    elif message_type == RequestType.UnAuthorizedRequest:
        return FailedRequest.model_validate(response_data)
    elif message_type == RequestType.SSHKeyRemoved:
        # SSHKeyRemoved is a success response, but we don't have a use case for it yet
        # Treat it as an unexpected response for now
        logger.error(
            f"Unexpected message_type: {message_type}. Raw payload: {json.dumps(response_data)}"
        )
        return FailedRequest(
            message_type=RequestType.FailedRequest,
            details=f"Unexpected message_type: {message_type}"
        )
    elif message_type == RequestType.GenericError:
        # GenericError should be treated as a FailedRequest
        return FailedRequest.model_validate(response_data)
    elif message_type in (RequestType.AcceptJobRequest, RequestType.DeclineJobRequest):
        # These message types are not expected in REST API responses
        logger.error(
            f"Unexpected message_type: {message_type}. Raw payload: {json.dumps(response_data)}"
        )
        return FailedRequest(
            message_type=RequestType.FailedRequest,
            details=f"Unexpected message_type: {message_type}"
        )
    else:
        # Handle any other unknown message types
        logger.error(
            f"Unexpected message_type: {message_type}. Raw payload: {json.dumps(response_data)}"
        )
        return FailedRequest(
            message_type=RequestType.FailedRequest,
            details=f"Unexpected message_type: {message_type}"
        )


JOB_LENGTH = 30

# HTTP timeout constants for REST API calls
REST_SSH_SUBMIT_TIMEOUT = 30  # Timeout for SSH key submission requests
REST_CONTAINER_OP_TIMEOUT = 30  # Timeout for container operations
REST_POD_LOGS_TIMEOUT = 30  # Timeout for pod logs requests
REST_SSH_REMOVE_TIMEOUT = 10  # Timeout for SSH key removal requests


class MinerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        task_service: Annotated[TaskService, Depends(TaskService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
        port_mapping_dao: Annotated[PortMappingDao, Depends(PortMappingDao)],
    ):
        self.ssh_service = ssh_service
        self.task_service = task_service
        self.redis_service = redis_service
        self.port_mapping_dao = port_mapping_dao

    async def request_job_to_miner(
        self,
        payload: MinerJobRequestPayload,
        encrypted_files: MinerJobEnryptedFiles,
    ):
        """Request job to miner - uses REST API if configured, otherwise WebSocket."""
        if settings.USE_REST_API:
            logger.info(
                _m(
                    "Routing request_job_to_miner to REST API",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                    }),
                ),
            )
            return await self._request_job_to_miner(payload, encrypted_files)
        else:
            logger.info(
                _m(
                    "Routing request_job_to_miner to WebSocket",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                    }),
                ),
            )
        
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
                miner_url=f"ws://{payload.miner_address}:{payload.miner_port}/websocket/{my_key.ss58_address}"
            )

            async with miner_client:
                # generate ssh key and send it to miner
                private_key, public_key = self.ssh_service.generate_ssh_key(my_key.ss58_address)


                await miner_client.send_model(
                    SSHPubKeySubmitRequest(
                        public_key=public_key,
                        miner_hotkey=payload.miner_hotkey, # include miner's hotkey in the request
                    )
                )

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
                            asyncio.wait_for(
                                self.task_service.create_task(
                                    miner_info=payload,
                                    executor_info=executor_info,
                                    keypair=my_key,
                                    private_key=private_key.decode("utf-8"),
                                    public_key=public_key.decode("utf-8"),
                                    encrypted_files=encrypted_files,
                                ),
                                timeout=settings.JOB_TIME_OUT - 120
                            )
                        )
                        for executor_info in msg.executors
                    ]

                    results = [
                        result
                        for result in await asyncio.gather(*tasks, return_exceptions=True)
                        if result and not isinstance(result, Exception) and not isinstance(result, BaseException)
                    ]

                    logger.info(
                        _m(
                            "Finished running tasks for executors",
                            extra=get_extra_info({**default_extra, "executors": len(results)}),
                        ),
                    )

                    try:
                        await miner_client.send_model(SSHPubKeyRemoveRequest(
                            public_key=public_key,
                            miner_hotkey=payload.miner_hotkey
                        ))
                    except Exception as e:
                        logger.warning(
                            _m(
                                "Failed to send SSHPubKeyRemoveRequest (non-critical)",
                                extra=get_extra_info({
                                    **default_extra,
                                    "error": _get_error_details(e),
                                }),
                            ),
                        )

                    return {
                        "miner_hotkey": payload.miner_hotkey,
                        "miner_coldkey": payload.miner_coldkey,
                        "results": results,
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
        except asyncio.TimeoutError:
            logger.error(
                _m("Requesting job to miner was timed out", extra=get_extra_info(default_extra)),
            )
            return None
        except Exception as e:
            logger.error(
                _m(
                    "Requesting job to miner resulted in an exception",
                    extra=get_extra_info({
                        **default_extra,
                        "error": _get_error_details(e),
                    }),
                ),
            )
            return None

    async def publish_machine_specs(
        self, results: list[JobResult], miner_hotkey: str, miner_coldkey: str
    ):
        """Publish machine specs to compute app connector process"""
        default_extra = {
            "miner_hotkey": miner_hotkey,
        }
        if not results:
            return

        logger.info(
            _m(
                "Publishing machine specs to compute app connector process",
                extra=get_extra_info({**default_extra, "job_batch_id": results[0].job_batch_id, "results": len(results)}),
            ),
        )
        for result in results:
            try:
                await self.redis_service.publish(
                    MACHINE_SPEC_CHANNEL,
                    {
                        "specs": result.spec,
                        "miner_hotkey": miner_hotkey,
                        "miner_coldkey": miner_coldkey,
                        "executor_uuid": result.executor_info.uuid,
                        "executor_ip": result.executor_info.address,
                        "executor_port": result.executor_info.port,
                        "executor_ssh_port": result.executor_info.ssh_port,
                        "executor_price": result.executor_info.price,
                        "price_per_gpu": result.executor_info.price_per_gpu,
                        "score": result.score,
                        "synthetic_job_score": result.job_score,
                        "job_batch_id": result.job_batch_id,
                        "log_status": result.log_status,
                        "log_text": result.log_text,
                        "collateral_deposited": result.collateral_deposited,
                        "ssh_pub_keys": result.ssh_pub_keys,
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

    def _handle_container_error(self, payload: ContainerBaseRequest, msg: str, error_code: FailedContainerErrorCodes):
        logger.error(msg)

        if isinstance(payload, ContainerCreateRequest):
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                msg=msg,
                error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                error_code=error_code,
            )

        elif isinstance(payload, ContainerDeleteRequest):
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                msg=msg,
                error_type=FailedContainerErrorTypes.ContainerDeletionFailed,
                error_code=error_code,
            )
        elif isinstance(payload, AddSshPublicKeyRequest):
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                msg=msg,
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=error_code,
            )
        elif isinstance(payload, InstallJupyterServerRequest):
            return JupyterInstallationFailed(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                msg=msg,
            )
        else:
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                msg=msg,
                error_type=FailedContainerErrorTypes.UnknownRequest,
                error_code=error_code,
            )

    async def handle_container(self, payload: ContainerBaseRequest):
        """Handle container request - uses REST API if configured, otherwise WebSocket."""
        if settings.USE_REST_API:
            logger.info(
                _m(
                    "Routing handle_container to REST API",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                        "executor_id": payload.executor_id,
                        "request_type": str(payload.message_type),
                    }),
                ),
            )
            return await self._handle_container(payload)
        else:
            logger.info(
                _m(
                    "Routing handle_container to WebSocket",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                        "executor_id": payload.executor_id,
                        "request_type": str(payload.message_type),
                    }),
                ),
            )
        
        loop = asyncio.get_event_loop()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_id": payload.executor_id,
            "pod_id": payload.pod_id,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
            "container_request_type": str(payload.message_type),
        }

        docker_service = DockerService(
            ssh_service=self.ssh_service,
            redis_service=self.redis_service,
            port_mapping_dao=self.port_mapping_dao
        )

        try:
            miner_client = MinerClient(
                loop=loop,
                miner_address=payload.miner_address,
                miner_port=payload.miner_port,
                miner_hotkey=payload.miner_hotkey,
                my_hotkey=my_key.ss58_address,
                keypair=my_key,
                miner_url=f"ws://{payload.miner_address}:{payload.miner_port}/websocket/{my_key.ss58_address}",
            )

            async with miner_client:
                # generate ssh key and send it to miner
                private_key, public_key = self.ssh_service.generate_ssh_key(my_key.ss58_address)

                await miner_client.send_model(
                    SSHPubKeySubmitRequest(
                        public_key=public_key,
                        executor_id=payload.executor_id,
                        is_rental_request=isinstance(payload, ContainerCreateRequest),
                        miner_hotkey=payload.miner_hotkey
                    )
                )

                logger.info(
                    _m("Sent SSH key to miner.", extra=get_extra_info(default_extra)),
                )

                msg = await asyncio.wait_for(
                    miner_client.job_state.miner_accepted_ssh_key_or_failed_future,
                    timeout=JOB_LENGTH,
                )

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
                        executor = None

                    if executor is None or executor.uuid != payload.executor_id:
                        log_text = _m("Error: Invalid executor id", extra=get_extra_info(default_extra))

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key, 
                                executor_id=payload.executor_id, 
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        if executor:
                            logger.info(
                                _m(
                                    "Remove rented machine from redis",
                                    extra=get_extra_info(default_extra),
                                ),
                            )
                            await self.redis_service.remove_rented_machine(executor)

                        return self._handle_container_error(
                            payload=payload,
                            msg=str(log_text),
                            error_code=FailedContainerErrorCodes.InvalidExecutorId
                        )

                    renting_in_progress = await self.redis_service.renting_in_progress(payload.miner_hotkey, payload.executor_id)
                    if renting_in_progress:
                        log_text = _m(
                            "Decline renting pod request. Renting is still in progress",
                            extra=get_extra_info(default_extra),
                        )

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key, 
                                executor_id=payload.executor_id, 
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return self._handle_container_error(
                            payload=payload,
                            msg=str(log_text),
                            error_code=FailedContainerErrorCodes.RentingInProgress,
                        )

                    # get private key for ssh connection - asyncssh
                    ssh_pkey = asyncssh.import_private_key(
                        self.ssh_service.decrypt_payload(
                            my_key.ss58_address, private_key.decode("utf-8")
                        )
                    )


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
                                public_key=public_key,
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return result

                    elif isinstance(payload, ContainerDeleteRequest):
                        logger.info(
                            _m(
                                "Deleting container",
                                extra=get_extra_info(
                                    {**default_extra, "payload": str(payload)}
                                ),
                            ),
                        )
                        result = await docker_service.delete_container(
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
                                public_key=public_key,
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return result
                    elif isinstance(payload, AddSshPublicKeyRequest):
                        logger.info(
                            _m(
                                "adding ssh key to container",
                                extra=get_extra_info(
                                    {**default_extra, "payload": str(payload)}
                                ),
                            ),
                        )
                        result = await docker_service.add_ssh_key(
                            payload,
                            executor,
                            my_key,
                            private_key.decode("utf-8"),
                        )

                        logger.info(
                            _m(
                                "Added ssh to the container",
                                extra=get_extra_info(
                                    {**default_extra, "payload": str(payload)}
                                ),
                            ),
                        )

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key,
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return result
                    elif isinstance(payload, RemoveSshPublicKeysRequest):
                        result = await docker_service.remove_ssh_keys(payload, executor, my_key, private_key.decode("utf-8"))

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key,
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return result
                    elif isinstance(payload, InstallJupyterServerRequest):
                        result = await docker_service.install_jupyter_server(payload, executor, my_key, private_key.decode("utf-8"))

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key,
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return result
                    elif isinstance(payload, BackupContainerRequest):
                        return await self.handle_backup_container_req(executor, payload, ssh_pkey)
                    elif isinstance(payload, RestoreContainerRequest):
                        return await self.handle_restore_container_req(executor, payload, ssh_pkey)
                    else:
                        log_text = _m(
                            "Unexpected request",
                            extra=get_extra_info(
                                {**default_extra, "payload": str(payload)}
                            ),
                        )

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key,
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return self._handle_container_error(
                            payload=payload,
                            msg=str(log_text),
                            error_code=FailedContainerErrorCodes.UnknownError,
                        )

                elif isinstance(msg, FailedRequest):
                    log_text = _m(
                        "Error: Miner failed job",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    )

                    return self._handle_container_error(
                        payload=payload,
                        msg=str(log_text),
                        error_code=FailedContainerErrorCodes.FailedMsgFromMiner,
                    )
                else:
                    log_text = _m(
                        "Error: Unexpected msg",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    )

                    return self._handle_container_error(
                        payload=payload,
                        msg=str(log_text),
                        error_code=FailedContainerErrorCodes.UnknownError,
                    )
        except Exception as e:
            log_text = _m(
                "Resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            return self._handle_container_error(
                payload=payload,
                msg=str(log_text),
                error_code=FailedContainerErrorCodes.ExceptionError,
            )

    async def get_pod_logs(self, payload: GetPodLogsRequestFromServer) -> PodLogsResponseToServer:
        """Get pod logs - uses REST API if configured, otherwise WebSocket."""
        if settings.USE_REST_API:
            logger.info(
                _m(
                    "Routing get_pod_logs to REST API",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                        "executor_id": payload.executor_id,
                    }),
                ),
            )
            return await self._get_pod_logs(payload)
        else:
            logger.info(
                _m(
                    "Routing get_pod_logs to WebSocket",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                        "executor_id": payload.executor_id,
                    }),
                ),
            )
        
        loop = asyncio.get_event_loop()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "pod_id": payload.pod_id,
            "executor_id": payload.executor_id,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
            "container_name": payload.container_name,
        }

        try:
            miner_client = MinerClient(
                loop=loop,
                miner_address=payload.miner_address,
                miner_port=payload.miner_port,
                miner_hotkey=payload.miner_hotkey,
                my_hotkey=my_key.ss58_address,
                keypair=my_key,
                miner_url=f"ws://{payload.miner_address}:{payload.miner_port}/websocket/{my_key.ss58_address}",
            )

            async with miner_client:
                # generate ssh key and send it to miner
                await miner_client.send_model(
                    GetPodLogsRequest(
                        container_name=payload.container_name,
                        pod_id=payload.pod_id,
                        executor_id=payload.executor_id, 
                        miner_hotkey=payload.miner_hotkey,
                    )
                )

                logger.info(
                    _m("Getting logs from executor", extra=get_extra_info(default_extra)),
                )

                msg = await asyncio.wait_for(
                    miner_client.job_state.miner_accepted_ssh_key_or_failed_future,
                    timeout=JOB_LENGTH,
                )

                if isinstance(msg, PodLogsResponse):
                    logger.info(
                        _m(
                            "Pod Log result",
                            extra=get_extra_info({**default_extra, "logs": len(msg.logs)}),
                        )
                    )
                    return PodLogsResponseToServer(
                        miner_hotkey=payload.miner_hotkey,
                        pod_id=payload.pod_id,
                        executor_id=payload.executor_id,
                        container_name=payload.container_name,
                        logs=msg.logs
                    )

                elif isinstance(msg, FailedRequest):
                    log_text = _m(
                        "Error: FailedRequest",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    )
                    logger.error(log_text)

                    return FailedGetPodLogs(
                        miner_hotkey=payload.miner_hotkey,
                        pod_id=payload.pod_id,
                        executor_id=payload.executor_id,
                        container_name=payload.container_name,
                        msg=str(log_text),
                    )

                else:
                    log_text = _m(
                        "Error: Unexpected msg",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    )
                    logger.error(log_text)

                    return FailedGetPodLogs(
                        miner_hotkey=payload.miner_hotkey,
                        pod_id=payload.pod_id,
                        executor_id=payload.executor_id,
                        container_name=payload.container_name,
                        msg=str(log_text),
                    )

        except Exception as e:
            log_text = _m(
                "Resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text)

            return FailedGetPodLogs(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                container_name=payload.container_name,
                msg=str(log_text),
            )

    async def add_debug_ssh_key(self, payload: AddDebugSshKeyRequest) -> DebugSshKeyAdded:
        """Add debug SSH key - uses REST API if configured, otherwise WebSocket."""
        if settings.USE_REST_API:
            logger.info(
                _m(
                    "Routing add_debug_ssh_key to REST API",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                        "executor_id": payload.executor_id,
                    }),
                ),
            )
            return await self._add_debug_ssh_key(payload)
        else:
            logger.info(
                _m(
                    "Routing add_debug_ssh_key to WebSocket",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                        "executor_id": payload.executor_id,
                    }),
                ),
            )
        
        loop = asyncio.get_event_loop()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_id": payload.executor_id,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
        }

        try:
            miner_client = MinerClient(
                loop=loop,
                miner_address=payload.miner_address,
                miner_port=payload.miner_port,
                miner_hotkey=payload.miner_hotkey,
                my_hotkey=my_key.ss58_address,
                keypair=my_key,
                miner_url=f"ws://{payload.miner_address}:{payload.miner_port}/websocket/{my_key.ss58_address}",
            )

            async with miner_client:

                await miner_client.send_model(
                    SSHPubKeySubmitRequest(
                        public_key=payload.public_key,
                        executor_id=payload.executor_id,
                        is_rental_request=False,
                        miner_hotkey=payload.miner_hotkey,
                    )
                )

                logger.info(
                    _m("Sent SSH key to miner.", extra=get_extra_info(default_extra)),
                )

                msg = await asyncio.wait_for(
                    miner_client.job_state.miner_accepted_ssh_key_or_failed_future,
                    timeout=JOB_LENGTH,
                )

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
                        executor = None

                    if executor is None or executor.uuid != payload.executor_id:
                        log_text = _m("Error: Invalid executor id", extra=get_extra_info(default_extra))
                        logger.error(log_text)

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=payload.public_key, 
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return FailedAddDebugSshKey(
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            msg=str(log_text),
                        )

                    logger.info(
                        _m(
                            "Added debug public key",
                            extra=get_extra_info(default_extra),
                        ),
                    )

                    return DebugSshKeyAdded(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        address=executor.address,
                        port=executor.port,
                        ssh_username=executor.ssh_username,
                        ssh_port=executor.ssh_port,
                    )

                else:
                    log_text = _m(
                        "Error: Failed to add debug public key",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    )
                    logger.error(log_text)

                    return FailedAddDebugSshKey(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        msg=str(log_text),
                    )

        except Exception as e:
            log_text = _m(
                "Resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            return FailedAddDebugSshKey(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                msg=str(log_text),
            )

    async def handle_backup_container_req(self, executor_info: ExecutorSSHInfo, payload: BackupContainerRequest, pkey: SSHKey):
        """Handle backup container request."""
        async with asyncssh.connect(
            host=executor_info.address,
            port=executor_info.ssh_port,
            username=executor_info.ssh_username,
            client_keys=[pkey],
            known_hosts=None,
        ) as ssh_client:

            # Upload the backup_storage.py script to the remote server before running it
            # Assume the local script is at './scripts/backup_storage.py'
            remote_script_path = "/root/app/backup_storage.py"
            local_script_path = os.path.join(
                os.path.dirname(__file__), 
                "..",
                "miner_jobs", 
                "backup_storage.py"
            )

            logger.info(
                _m(
                    "Uploading backup_storage.py script to the remote server", 
                    extra=get_extra_info({ "remote_script_path": remote_script_path, "local_script_path": local_script_path })
                ),
            )

            async with ssh_client.start_sftp_client() as sftp:
                await sftp.put(local_script_path, remote_script_path)

            commands = [
                "nohup",
                executor_info.python_path,
                "/root/app/backup_storage.py",
                "--api-url", settings.COMPUTE_REST_API_URL,
                "--source-volume", payload.source_volume,
                "--backup-path", payload.backup_path,
                "--auth-token", payload.auth_token,
                "--backup-log-id", payload.backup_log_id,
                "--backup-volume-name", payload.backup_volume_info.name,
                "--backup-volume-iam_user_access_key", payload.backup_volume_info.iam_user_access_key,
                "--backup-volume-iam_user_secret_key", payload.backup_volume_info.iam_user_secret_key,
                "--source-volume-path", payload.source_volume_path,
                "--backup-target-path", payload.backup_target_path,
                "> /root/app/backup_storage.log 2>&1 &"
            ]
            await ssh_client.run(" ".join(commands), timeout=50, check=True)

    async def handle_restore_container_req(self, executor_info: ExecutorSSHInfo, payload: RestoreContainerRequest, pkey: SSHKey):
        """Handle restore container request."""
        async with asyncssh.connect(
            host=executor_info.address,
            port=executor_info.ssh_port,
            username=executor_info.ssh_username,
            client_keys=[pkey],
            known_hosts=None,
        ) as ssh_client:

            # Upload the restore_storage.py script to the remote server before running it
            remote_script_path = "/root/app/restore_storage.py"
            local_script_path = os.path.join(
                os.path.dirname(__file__), 
                "..",
                "miner_jobs", 
                "restore_storage.py"
            )

            logger.info(
                _m(
                    "Uploading restore_storage.py script to the remote server for restore operation", 
                    extra=get_extra_info({ "remote_script_path": remote_script_path, "local_script_path": local_script_path })
                ),
            )

            async with ssh_client.start_sftp_client() as sftp:
                await sftp.put(local_script_path, remote_script_path)

            commands = [
                "nohup",
                executor_info.python_path,
                "/root/app/restore_storage.py",
                "--api-url", settings.COMPUTE_REST_API_URL,
                "--target-volume", payload.target_volume,
                "--restore-path", payload.restore_path,
                "--backup-source-path", payload.backup_source_path,
                "--auth-token", payload.auth_token,
                "--restore-log-id", payload.restore_log_id,
                "--backup-volume-name", payload.backup_volume_info.name,
                "--backup-volume-iam_user_access_key", payload.backup_volume_info.iam_user_access_key,
                "--backup-volume-iam_user_secret_key", payload.backup_volume_info.iam_user_secret_key,
                "--target-volume-path", payload.target_volume_path,
                "> /root/app/restore_storage.log 2>&1 &"
            ]
            await ssh_client.run(" ".join(commands), timeout=50, check=True)

    def _generate_auth_headers(self, my_key: bittensor.Keypair, miner_hotkey: str) -> dict:
        """Generate authentication headers for REST API requests.

        IMPORTANT: Uses AuthenticationPayload.blob_for_signing() for canonical serialization.
        This contract is defined in datura/datura/requests/validator_requests.py
        """
        payload = AuthenticationPayload(
            validator_hotkey=my_key.ss58_address,
            miner_hotkey=miner_hotkey,
            timestamp=int(time.time()),
        )
        signature = f"0x{my_key.sign(payload.blob_for_signing()).hex()}"

        return {
            "X-Validator-Hotkey": my_key.ss58_address,
            "X-Miner-Hotkey": miner_hotkey,
            "X-Timestamp": str(payload.timestamp),
            "X-Signature": signature,
        }

    async def _make_rest_request(
        self,
        method: str,
        url: str,
        json_data: dict,
        headers: dict,
        timeout: int,
        log_extra: dict,
        operation_name: str,
    ) -> tuple[int, dict | None]:
        """Make a REST API request to miner with proper error handling and logging.

        Args:
            method: HTTP method (e.g., 'POST', 'GET')
            url: Full URL to request
            json_data: JSON payload to send
            headers: HTTP headers
            timeout: Request timeout in seconds
            log_extra: Additional logging context
            operation_name: Name of operation for logging (e.g., 'SSH key submit')

        Returns:
            Tuple of (status_code, response_json). response_json is None if request failed
            or response is not valid JSON.

        Raises:
            asyncio.TimeoutError: If request times out
            aiohttp.ClientError: For other HTTP client errors
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    status = response.status
                    try:
                        response_data = await response.json()
                    except Exception:
                        # If response is not JSON, read as text
                        response_text = await response.text()
                        if status != 200:
                            logger.error(
                                _m(
                                    f"REST API {operation_name} failed with non-JSON response",
                                    extra=get_extra_info({
                                        **log_extra,
                                        "status": status,
                                        "response_text": response_text[:500],  # Limit length
                                        "url": url,
                                    }),
                                ),
                            )
                        return status, None

                    if status != 200:
                        logger.error(
                            _m(
                                f"REST API {operation_name} failed",
                                extra=get_extra_info({
                                    **log_extra,
                                    "status": status,
                                    "response": response_data,
                                    "url": url,
                                }),
                            ),
                        )
                    return status, response_data

        except asyncio.TimeoutError:
            logger.error(
                _m(
                    f"REST API {operation_name} timed out after {timeout}s",
                    extra=get_extra_info({
                        **log_extra,
                        "timeout": timeout,
                        "url": url,
                    }),
                ),
            )
            raise
        except aiohttp.ClientError as e:
            logger.error(
                _m(
                    f"REST API {operation_name} client error",
                    extra=get_extra_info({
                        **log_extra,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "url": url,
                    }),
                ),
            )
            raise
        except Exception as e:
            logger.error(
                _m(
                    f"REST API {operation_name} unexpected error",
                    extra=get_extra_info({
                        **log_extra,
                        "error": _get_error_details(e),
                        "url": url,
                    }),
                ),
                exc_info=True,
            )
            raise

    async def _remove_ssh_key_via_rest(
        self,
        base_url: str,
        my_key: bittensor.Keypair,
        public_key: bytes,
        miner_hotkey: str,
        executor_id: str | None,
        log_extra: dict,
    ) -> bool:
        """Remove SSH key from miner via REST API.

        Args:
            base_url: Base URL of miner (e.g., 'http://192.168.1.1:8000')
            my_key: Validator's keypair for authentication
            public_key: SSH public key to remove
            miner_hotkey: Miner's hotkey
            executor_id: Optional executor ID
            log_extra: Additional logging context

        Returns:
            True if removal was successful (status 200), False otherwise.
            Logs warnings for failures but does not raise exceptions.
        """
        try:
            remove_request = SSHPubKeyRemoveRequest(
                public_key=public_key,
                executor_id=executor_id,
                miner_hotkey=miner_hotkey,
            )

            status, _ = await self._make_rest_request(
                method="POST",
                url=f"{base_url}/api/validator/ssh-pubkey-remove",
                json_data=self._serialize_request(remove_request),
                headers=self._generate_auth_headers(my_key, miner_hotkey),
                timeout=REST_SSH_REMOVE_TIMEOUT,
                log_extra=log_extra,
                operation_name="SSH key removal",
            )

            if status != 200:
                logger.warning(
                    _m(
                        "Failed to remove SSH key via REST API. Validator key may still be present on miner",
                        extra=get_extra_info({
                            **log_extra,
                            "status": status,
                            "miner_hotkey": miner_hotkey,
                            "executor_id": executor_id,
                        }),
                    ),
                )
                return False

            return True

        except Exception as e:
            logger.warning(
                _m(
                    "Failed to remove SSH key via REST API. Validator key may still be present on miner",
                    extra=get_extra_info({
                        **log_extra,
                        "error": _get_error_details(e),
                        "miner_hotkey": miner_hotkey,
                        "executor_id": executor_id,
                    }),
                ),
            )
            return False

    def _serialize_request(self, request) -> dict:
        """Serialize a Pydantic request model to dict for JSON serialization.
        
        Handles bytes fields by ensuring they're properly encoded.
        """
        # Use model_dump_json and parse back to ensure proper serialization
        # This handles bytes fields correctly (base64 encoding)
        return json.loads(request.model_dump_json())

    async def _request_job_to_miner(
        self,
        payload: MinerJobRequestPayload,
        encrypted_files: MinerJobEnryptedFiles,
    ):
        """REST API version of request_job_to_miner."""
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "job_batch_id": payload.job_batch_id,
            "miner_hotkey": payload.miner_hotkey,
            "miner_address": payload.miner_address,
            "miner_port": payload.miner_port,
        }

        try:
            logger.info(_m("Requesting job to miner via REST API", extra=get_extra_info(default_extra)))
            
            # Generate SSH key
            private_key, public_key = self.ssh_service.generate_ssh_key(my_key.ss58_address)
            
            # Prepare request
            ssh_request = SSHPubKeySubmitRequest(
                public_key=public_key,
                miner_hotkey=payload.miner_hotkey,
            )
            
            # Make REST API call
            base_url = f"http://{payload.miner_address}:{payload.miner_port}"
            headers = self._generate_auth_headers(my_key, payload.miner_hotkey)
            headers["Content-Type"] = "application/json"
            
            status, response_data = await self._make_rest_request(
                method="POST",
                url=f"{base_url}/api/validator/ssh-pubkey-submit",
                json_data=self._serialize_request(ssh_request),
                headers=headers,
                timeout=REST_SSH_SUBMIT_TIMEOUT,
                log_extra=default_extra,
                operation_name="SSH key submit",
            )

            if status != 200 or response_data is None:
                return None

            msg = _parse_miner_response(response_data)
            
            # Track whether SSH key was successfully accepted
            ssh_key_accepted = False
            
            if isinstance(msg, AcceptSSHKeyRequest):
                ssh_key_accepted = True
                logger.info(
                    _m(
                        "Received AcceptSSHKeyRequest for miner via REST API. Running tasks for executors",
                        extra=get_extra_info(
                            {**default_extra, "executors": len(msg.executors)}
                        ),
                    ),
                )
                if len(msg.executors) == 0:
                    return None

                tasks = [
                    asyncio.create_task(
                        asyncio.wait_for(
                            self.task_service.create_task(
                                miner_info=payload,
                                executor_info=executor_info,
                                keypair=my_key,
                                private_key=private_key.decode("utf-8"),
                                public_key=public_key.decode("utf-8"),
                                encrypted_files=encrypted_files,
                            ),
                            timeout=settings.JOB_TIME_OUT - 120
                        )
                    )
                    for executor_info in msg.executors
                ]

                results = [
                    result
                    for result in await asyncio.gather(*tasks, return_exceptions=True)
                    if result and not isinstance(result, Exception) and not isinstance(result, BaseException)
                ]

                logger.info(
                    _m(
                        "Finished running tasks for executors",
                        extra=get_extra_info({**default_extra, "executors": len(results)}),
                    ),
                )

                # Remove SSH key only if it was successfully accepted
                if ssh_key_accepted:
                    await self._remove_ssh_key_via_rest(
                        base_url=base_url,
                        my_key=my_key,
                        public_key=public_key,
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=None,
                        log_extra=default_extra,
                    )

                return {
                    "miner_hotkey": payload.miner_hotkey,
                    "miner_coldkey": payload.miner_coldkey,
                    "results": results,
                }
            elif isinstance(msg, FailedRequest):
                logger.warning(
                    _m(
                        "Requesting job failed for miner via REST API",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    ),
                )
                return None
            else:
                logger.error(
                    _m(
                        "Unexpected response from miner via REST API",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    ),
                )
                return None
        except asyncio.CancelledError:
            logger.error(
                _m("Requesting job to miner via REST API was cancelled", extra=get_extra_info(default_extra)),
            )
            return None
        except asyncio.TimeoutError:
            logger.error(
                _m("Requesting job to miner via REST API was timed out", extra=get_extra_info(default_extra)),
            )
            return None
        except Exception as e:
            logger.error(
                _m(
                    "Requesting job to miner via REST API resulted in an exception",
                    extra=get_extra_info({
                        **default_extra,
                        "error": _get_error_details(e),
                    }),
                ),
            )
            return None

    async def _handle_container(self, payload: ContainerBaseRequest):
        """REST API version of handle_container."""
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_id": payload.executor_id,
            "pod_id": payload.pod_id,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
            "container_request_type": str(payload.message_type),
        }

        docker_service = DockerService(
            ssh_service=self.ssh_service,
            redis_service=self.redis_service,
            port_mapping_dao=self.port_mapping_dao
        )

        try:
            base_url = f"http://{payload.miner_address}:{payload.miner_port}"
            headers = self._generate_auth_headers(my_key, payload.miner_hotkey)
            headers["Content-Type"] = "application/json"
            
            # Generate SSH key and send it to miner
            private_key, public_key = self.ssh_service.generate_ssh_key(my_key.ss58_address)

            ssh_request = SSHPubKeySubmitRequest(
                public_key=public_key,
                executor_id=payload.executor_id,
                is_rental_request=isinstance(payload, ContainerCreateRequest),
                miner_hotkey=payload.miner_hotkey
            )

            logger.info(
                _m("Sent SSH key to miner via REST API.", extra=get_extra_info(default_extra)),
            )

            status, response_data = await self._make_rest_request(
                method="POST",
                url=f"{base_url}/api/validator/ssh-pubkey-submit",
                json_data=self._serialize_request(ssh_request),
                headers=headers,
                timeout=REST_CONTAINER_OP_TIMEOUT,
                log_extra=default_extra,
                operation_name="SSH key submit",
            )

            if status != 200 or response_data is None:
                error_msg = "Failed to submit SSH key"
                if response_data:
                    error_msg = f"{error_msg}: {response_data}"
                return self._handle_container_error(
                    payload=payload,
                    msg=error_msg,
                    error_code=FailedContainerErrorCodes.FailedMsgFromMiner,
                )

            msg = _parse_miner_response(response_data)

            # Track whether SSH key was successfully accepted
            ssh_key_accepted = False

            if isinstance(msg, AcceptSSHKeyRequest):
                ssh_key_accepted = True
                logger.info(
                    _m(
                        "Received AcceptSSHKeyRequest via REST API",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    ),
                )

                try:
                    executor = msg.executors[0]
                except Exception as e:
                    executor = None

                if executor is None or executor.uuid != payload.executor_id:
                    log_text = _m("Error: Invalid executor id", extra=get_extra_info(default_extra))

                    # Remove SSH key only if it was accepted
                    if ssh_key_accepted:
                        await self._remove_ssh_key_via_rest(
                            base_url=base_url,
                            my_key=my_key,
                            public_key=public_key,
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            log_extra=default_extra,
                        )

                    if executor:
                        logger.info(
                            _m(
                                "Remove rented machine from redis",
                                extra=get_extra_info(default_extra),
                            ),
                        )
                        await self.redis_service.remove_rented_machine(executor)

                    return self._handle_container_error(
                        payload=payload,
                        msg=str(log_text),
                        error_code=FailedContainerErrorCodes.InvalidExecutorId
                    )

                renting_in_progress = await self.redis_service.renting_in_progress(payload.miner_hotkey, payload.executor_id)
                if renting_in_progress:
                    log_text = _m(
                        "Decline renting pod request. Renting is still in progress",
                        extra=get_extra_info(default_extra),
                    )

                    # Remove SSH key only if it was accepted
                    if ssh_key_accepted:
                        await self._remove_ssh_key_via_rest(
                            base_url=base_url,
                            my_key=my_key,
                            public_key=public_key,
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            log_extra=default_extra,
                        )

                    return self._handle_container_error(
                        payload=payload,
                        msg=str(log_text),
                        error_code=FailedContainerErrorCodes.RentingInProgress,
                    )

                # Get private key for ssh connection - asyncssh
                ssh_pkey = asyncssh.import_private_key(
                    self.ssh_service.decrypt_payload(
                        my_key.ss58_address, private_key.decode("utf-8")
                    )
                )

                # Handle different container request types
                result = None
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
                elif isinstance(payload, ContainerDeleteRequest):
                    logger.info(
                        _m(
                            "Deleting container",
                            extra=get_extra_info(
                                {**default_extra, "payload": str(payload)}
                            ),
                        ),
                    )
                    result = await docker_service.delete_container(
                        payload,
                        executor,
                        my_key,
                        private_key.decode("utf-8"),
                    )
                elif isinstance(payload, AddSshPublicKeyRequest):
                    logger.info(
                        _m(
                            "adding ssh key to container",
                            extra=get_extra_info(
                                {**default_extra, "payload": str(payload)}
                            ),
                        ),
                    )
                    result = await docker_service.add_ssh_key(
                        payload,
                        executor,
                        my_key,
                        private_key.decode("utf-8"),
                    )
                elif isinstance(payload, RemoveSshPublicKeysRequest):
                    result = await docker_service.remove_ssh_keys(payload, executor, my_key, private_key.decode("utf-8"))
                elif isinstance(payload, InstallJupyterServerRequest):
                    result = await docker_service.install_jupyter_server(payload, executor, my_key, private_key.decode("utf-8"))
                elif isinstance(payload, BackupContainerRequest):
                    result = await self.handle_backup_container_req(executor, payload, ssh_pkey)
                elif isinstance(payload, RestoreContainerRequest):
                    result = await self.handle_restore_container_req(executor, payload, ssh_pkey)
                else:
                    log_text = _m(
                        "Unexpected request",
                        extra=get_extra_info(
                            {**default_extra, "payload": str(payload)}
                        ),
                    )
                    return self._handle_container_error(
                        payload=payload,
                        msg=str(log_text),
                        error_code=FailedContainerErrorCodes.UnknownError,
                    )

                # Remove SSH key after operation only if it was accepted
                if ssh_key_accepted:
                    await self._remove_ssh_key_via_rest(
                        base_url=base_url,
                        my_key=my_key,
                        public_key=public_key,
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        log_extra=default_extra,
                    )

                return result

            elif isinstance(msg, FailedRequest):
                log_text = _m(
                    "Error: Miner failed job",
                    extra=get_extra_info({**default_extra, "msg": str(msg)}),
                )

                return self._handle_container_error(
                    payload=payload,
                    msg=str(log_text),
                    error_code=FailedContainerErrorCodes.FailedMsgFromMiner,
                )
            else:
                log_text = _m(
                    "Error: Unexpected msg",
                    extra=get_extra_info({**default_extra, "msg": str(msg)}),
                )

                return self._handle_container_error(
                    payload=payload,
                    msg=str(log_text),
                    error_code=FailedContainerErrorCodes.UnknownError,
                )
        except Exception as e:
            log_text = _m(
                "Resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            return self._handle_container_error(
                payload=payload,
                msg=str(log_text),
                error_code=FailedContainerErrorCodes.ExceptionError,
            )

    async def _get_pod_logs(self, payload: GetPodLogsRequestFromServer) -> PodLogsResponseToServer:
        """REST API version of get_pod_logs."""
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "pod_id": payload.pod_id,
            "executor_id": payload.executor_id,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
            "container_name": payload.container_name,
        }

        try:
            base_url = f"http://{payload.miner_address}:{payload.miner_port}"
            headers = self._generate_auth_headers(my_key, payload.miner_hotkey)
            headers["Content-Type"] = "application/json"
            
            logs_request = GetPodLogsRequest(
                container_name=payload.container_name,
                pod_id=payload.pod_id,
                executor_id=payload.executor_id, 
                miner_hotkey=payload.miner_hotkey,
            )

            logger.info(
                _m("Getting logs from executor via REST API", extra=get_extra_info(default_extra)),
            )

            status, response_data = await self._make_rest_request(
                method="POST",
                url=f"{base_url}/api/validator/pod-logs",
                json_data=self._serialize_request(logs_request),
                headers=headers,
                timeout=REST_POD_LOGS_TIMEOUT,
                log_extra=default_extra,
                operation_name="pod logs",
            )

            if status != 200 or response_data is None:
                error_msg = "Failed to get pod logs"
                if response_data:
                    error_msg = f"{error_msg}: {response_data}"
                log_text = _m(
                    "Error: FailedRequest",
                    extra=get_extra_info({**default_extra, "error": error_msg}),
                )
                logger.error(log_text)
                return FailedGetPodLogs(
                    miner_hotkey=payload.miner_hotkey,
                    pod_id=payload.pod_id,
                    executor_id=payload.executor_id,
                    container_name=payload.container_name,
                    msg=str(log_text),
                )

            msg = _parse_miner_response(response_data)

            if isinstance(msg, PodLogsResponse):
                logger.info(
                    _m(
                        "Pod Log result via REST API",
                        extra=get_extra_info({**default_extra, "logs": len(msg.logs)}),
                    )
                )
                return PodLogsResponseToServer(
                    miner_hotkey=payload.miner_hotkey,
                    pod_id=payload.pod_id,
                    executor_id=payload.executor_id,
                    container_name=payload.container_name,
                    logs=msg.logs
                )

            elif isinstance(msg, FailedRequest):
                log_text = _m(
                    "Error: FailedRequest",
                    extra=get_extra_info({**default_extra, "msg": str(msg)}),
                )
                logger.error(log_text)

                return FailedGetPodLogs(
                    miner_hotkey=payload.miner_hotkey,
                    pod_id=payload.pod_id,
                    executor_id=payload.executor_id,
                    container_name=payload.container_name,
                    msg=str(log_text),
                )

            else:
                log_text = _m(
                    "Error: Unexpected msg",
                    extra=get_extra_info({**default_extra, "msg": str(msg)}),
                )
                logger.error(log_text)

                return FailedGetPodLogs(
                    miner_hotkey=payload.miner_hotkey,
                    pod_id=payload.pod_id,
                    executor_id=payload.executor_id,
                    container_name=payload.container_name,
                    msg=str(log_text),
                )

        except Exception as e:
            log_text = _m(
                "Resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text)

            return FailedGetPodLogs(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                container_name=payload.container_name,
                msg=str(log_text),
            )

    async def _add_debug_ssh_key(self, payload: AddDebugSshKeyRequest) -> DebugSshKeyAdded:
        """REST API version of add_debug_ssh_key."""
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_id": payload.executor_id,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
        }

        try:
            base_url = f"http://{payload.miner_address}:{payload.miner_port}"
            headers = self._generate_auth_headers(my_key, payload.miner_hotkey)
            headers["Content-Type"] = "application/json"
            
            ssh_request = SSHPubKeySubmitRequest(
                public_key=payload.public_key,
                executor_id=payload.executor_id,
                is_rental_request=False,
                miner_hotkey=payload.miner_hotkey,
            )

            logger.info(
                _m("Sent SSH key to miner via REST API.", extra=get_extra_info(default_extra)),
            )

            status, response_data = await self._make_rest_request(
                method="POST",
                url=f"{base_url}/api/validator/ssh-pubkey-submit",
                json_data=self._serialize_request(ssh_request),
                headers=headers,
                timeout=REST_SSH_SUBMIT_TIMEOUT,
                log_extra=default_extra,
                operation_name="SSH key submit (debug)",
            )

            if status != 200 or response_data is None:
                error_msg = "Failed to add debug public key"
                if response_data:
                    error_msg = f"{error_msg}: {response_data}"
                log_text = _m(
                    "Error: Failed to add debug public key",
                    extra=get_extra_info({**default_extra, "error": error_msg}),
                )
                logger.error(log_text)
                return FailedAddDebugSshKey(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    msg=str(log_text),
                )

            msg = _parse_miner_response(response_data)

            # Track whether SSH key was successfully accepted
            ssh_key_accepted = False

            if isinstance(msg, AcceptSSHKeyRequest):
                ssh_key_accepted = True
                logger.info(
                    _m(
                        "Received AcceptSSHKeyRequest via REST API",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    ),
                )

                try:
                    executor = msg.executors[0]
                except Exception as e:
                    executor = None

                if executor is None or executor.uuid != payload.executor_id:
                    log_text = _m("Error: Invalid executor id", extra=get_extra_info(default_extra))
                    logger.error(log_text)

                    # Remove SSH key only if it was accepted
                    if ssh_key_accepted:
                        await self._remove_ssh_key_via_rest(
                            base_url=base_url,
                            my_key=my_key,
                            public_key=payload.public_key,
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            log_extra=default_extra,
                        )

                    return FailedAddDebugSshKey(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        msg=str(log_text),
                    )

                logger.info(
                    _m(
                        "Added debug public key",
                        extra=get_extra_info(default_extra),
                    ),
                )

                return DebugSshKeyAdded(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    address=executor.address,
                    port=executor.port,
                    ssh_username=executor.ssh_username,
                    ssh_port=executor.ssh_port,
                )

            else:
                log_text = _m(
                    "Error: Failed to add debug public key",
                    extra=get_extra_info({**default_extra, "msg": str(msg)}),
                )
                logger.error(log_text)

                return FailedAddDebugSshKey(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    msg=str(log_text),
                )

        except Exception as e:
            log_text = _m(
                "Resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            return FailedAddDebugSshKey(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                msg=str(log_text),
            )

MinerServiceDep = Annotated[MinerService, Depends(MinerService)]
