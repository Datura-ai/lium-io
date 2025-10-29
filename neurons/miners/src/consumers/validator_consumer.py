import asyncio
import logging
import time
from typing import Annotated

import bittensor
from datura.consumers.base import BaseConsumer
from datura.requests.miner_requests import (
    AcceptJobRequest,
    AcceptSSHKeyRequest,
    DeclineJobRequest,
    Executor,
    ExecutorSSHInfo,
    FailedRequest,
    UnAuthorizedRequest,
    PodLog,
    PodLogsResponse,
)
from datura.requests.validator_requests import (
    AuthenticateRequest,
    BaseValidatorRequest,
    SSHPubKeyRemoveRequest,
    SSHPubKeySubmitRequest,
    GetPodLogsRequest,
)
from fastapi import Depends, WebSocket

from core.config import settings
from services.executor_service import ExecutorService
from services.ssh_service import MinerSSHService
from services.validator_service import ValidatorService

AUTH_MESSAGE_MAX_AGE = 10
MAX_MESSAGE_COUNT = 10

logger = logging.getLogger(__name__)


class ValidatorConsumer(BaseConsumer):
    def __init__(
        self,
        websocket: WebSocket,
        validator_key: str,
        ssh_service: Annotated[MinerSSHService, Depends(MinerSSHService)],
        validator_service: Annotated[ValidatorService, Depends(ValidatorService)],
        executor_service: Annotated[ExecutorService, Depends(ExecutorService)],
    ):
        super().__init__(websocket)
        self.ssh_service = ssh_service
        self.validator_service = validator_service
        self.executor_service = executor_service
        self.validator_key = validator_key
        self.my_hotkey = settings.get_bittensor_wallet().get_hotkey().ss58_address
        self.validator_authenticated = False
        self.msg_queue = []

    def accepted_request_type(self):
        return BaseValidatorRequest

    def verify_auth_msg(self, msg: AuthenticateRequest) -> tuple[bool, str]:
        if msg.payload.timestamp < time.time() - AUTH_MESSAGE_MAX_AGE:
            return False, "msg too old"
        if msg.payload.miner_hotkey != self.my_hotkey:
            return False, f"wrong miner hotkey ({self.my_hotkey}!={msg.payload.miner_hotkey})"
        if msg.payload.validator_hotkey != self.validator_key:
            return (
                False,
                f"wrong validator hotkey ({self.validator_key}!={msg.payload.validator_hotkey})",
            )

        keypair = bittensor.Keypair(ss58_address=self.validator_key)
        if keypair.verify(msg.blob_for_signing(), msg.signature):
            return True, ""

    async def handle_authentication(self, msg: AuthenticateRequest):
        # check if validator is registered
        if not self.validator_service.is_valid_validator(self.validator_key):
            await self.send_message(UnAuthorizedRequest(details="Validator is not registered"))
            await self.disconnect()
            return

        authenticated, error_msg = self.verify_auth_msg(msg)
        if not authenticated:
            response_msg = f"Validator {self.validator_key} not authenticated due to: {error_msg}"
            logger.info(response_msg)
            await self.send_message(UnAuthorizedRequest(details=response_msg))
            await self.disconnect()
            return

        self.validator_authenticated = True
        for msg in self.msg_queue:
            await self.handle_message(msg)

    async def check_validator_allowance(self):
        """Check if there's any executors opened for current validator.

        If there are any executors, send accept job request to validator w/ executors list
        available for that validator.

        If no executors, decline job request
        """
        executors = self.executor_service.get_executors_for_validator(self.validator_key)
        if len(executors):
            logger.info("Found %d executors for validator(%s)", len(executors), self.validator_key)
            await self.send_message(
                AcceptJobRequest(
                    executors=[
                        Executor(uuid=str(executor.uuid), address=executor.address, port=executor.port)
                        for executor in executors
                    ]
                )
            )
        else:
            logger.debug("Not found any executors for validator(%s)", self.validator_key)
            await self.send_message(DeclineJobRequest())
            await self.disconnect()

    async def invoke_rental_request_hook(self, payload: dict):
       if settings.RENTAL_REQUEST_HOOK:
           import aiohttp
           try:
               async with aiohttp.ClientSession() as session:
                   resp = await asyncio.wait_for(
                       session.post(settings.RENTAL_REQUEST_HOOK, json=payload),
                       timeout=5
                   )
                   if resp.status != 200:
                       logger.error(f"RENTAL_REQUEST_HOOK returned status {resp.status}: {await resp.text()}")
           except asyncio.TimeoutError:
               logger.error("RENTAL_REQUEST_HOOK timed out after 5 seconds")
           except Exception as e:
               logger.error(f"Exception in RENTAL_REQUEST_HOOK: {e}")

    async def handle_message(self, msg: BaseValidatorRequest):
        if isinstance(msg, AuthenticateRequest):
            await self.handle_authentication(msg)
            if self.validator_authenticated:
                await self.check_validator_allowance()
            return

        # TODO: update logic here, fow now, it sends AcceptJobRequest regardless
        # if self.validator_authenticated:
        #     await self.send_message(AcceptJobRequest())

        if not self.validator_authenticated:
            if len(self.msg_queue) <= MAX_MESSAGE_COUNT:
                self.msg_queue.append(msg)
            return

        if isinstance(msg, SSHPubKeySubmitRequest):
            logger.info("Validator %s sent SSH Pubkey.", self.validator_key)

            try:
                msg: SSHPubKeySubmitRequest
                executors: list[ExecutorSSHInfo] = await self.executor_service.register_pubkey(
                    self.validator_key, msg.public_key, msg.executor_id, msg.validator_signature
                )
                if msg.is_rental_request and len(executors) == 1:
                    await self.invoke_rental_request_hook(
                        {
                            "executor_address": executors[0].address,
                            "executor_port": executors[0].port,
                            "uuid": str(executors[0].uuid),
                        }
                    )
    
                await self.send_message(AcceptSSHKeyRequest(executors=executors))
                logger.info("Sent AcceptSSHKeyRequest to validator %s", self.validator_key)
            except Exception as e:
                logger.error("Storing SSH key or Sending AcceptSSHKeyRequest failed: %s", str(e))
                self.ssh_service.remove_pubkey_from_host(msg.public_key)
                await self.send_message(FailedRequest(details=str(e)))
            return

        if isinstance(msg, SSHPubKeyRemoveRequest):
            logger.info("Validator %s sent remove SSH Pubkey.", self.validator_key)
            try:
                await self.executor_service.deregister_pubkey(self.validator_key, msg.public_key, msg.executor_id)
                logger.info("Sent SSHKeyRemoved to validator %s", self.validator_key)
            except Exception as e:
                logger.error("Failed SSHKeyRemoved request: %s", str(e))
                await self.send_message(FailedRequest(details=str(e)))
            return

        if isinstance(msg, GetPodLogsRequest):
            logger.info("Validator %s get pod logs for container %s.", self.validator_key, msg.container_name)
            try:
                logs: list[PodLog] = await self.executor_service.get_pod_logs(self.validator_key, msg.executor_id, msg.container_name)
                await self.send_message(PodLogsResponse(logs=logs))
                logger.info("Sent GetPodLogs to validator %s", self.validator_key)
            except Exception as e:
                logger.error("Failed GetPodLogs request: %s", str(e))
                await self.send_message(FailedRequest(details=str(e)))
            return


class ValidatorConsumerManger:
    def __init__(
        self,
    ):
        self.active_consumer: ValidatorConsumer | None = None
        self.lock = asyncio.Lock()

    async def addConsumer(
        self,
        websocket: WebSocket,
        validator_key: str,
        ssh_service: Annotated[MinerSSHService, Depends(MinerSSHService)],
        validator_service: Annotated[ValidatorService, Depends(ValidatorService)],
        executor_service: Annotated[ExecutorService, Depends(ExecutorService)],
    ):
        consumer = ValidatorConsumer(
            websocket=websocket,
            validator_key=validator_key,
            ssh_service=ssh_service,
            validator_service=validator_service,
            executor_service=executor_service,
        )
        await consumer.connect()

        if self.active_consumer is not None:
            await consumer.send_message(DeclineJobRequest())
            await consumer.disconnect()
            return

        async with self.lock:
            self.active_consumer = consumer

            await self.active_consumer.handle()

            self.active_consumer = None


validatorConsumerManager = ValidatorConsumerManger()
