import asyncio
import json
import logging
from typing import NoReturn

import bittensor
import pydantic
import redis.asyncio as aioredis
import tenacity
import websockets
from datura.requests.base import BaseRequest
from payload_models.payloads import (
    ContainerBaseRequest,
    ContainerCreated,
    ContainerCreateRequest,
    ContainerDeleteRequest,
    ContainerStartRequest,
    ContainerStopRequest,
    DuplicateExecutorsResponse,
    FailedContainerRequest,
)
from protocol.vc_protocol.compute_requests import Error, RentedMachineResponse, Response
from protocol.vc_protocol.validator_requests import (
    AuthenticateRequest,
    DuplicateExecutorsRequest,
    ExecutorSpecRequest,
    LogStreamRequest,
    RentedMachineRequest,
    ResetVerifiedJobRequest,
)
from pydantic import BaseModel
from websockets.asyncio.client import ClientConnection

from core.validator import Validator
from core.utils import _m, get_extra_info
from services.miner_service import MinerService
from services.redis_service import (
    DUPLICATED_MACHINE_SET,
    MACHINE_SPEC_CHANNEL,
    RENTED_MACHINE_PREFIX,
    RESET_VERIFIED_JOB_CHANNEL,
    STREAMING_LOG_CHANNEL,
)

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    def __init__(self, reason: str, errors: list[Error]):
        self.reason = reason
        self.errors = errors


class ComputeClient:
    HEARTBEAT_PERIOD = 60

    def __init__(
        self, keypair: bittensor.Keypair, compute_app_uri: str, miner_service: MinerService
    ):
        self.keypair = keypair
        self.ws: ClientConnection | None = None
        self.compute_app_uri = compute_app_uri
        self.miner_drivers = asyncio.Queue()
        self.miner_driver_awaiter_task = asyncio.create_task(self.miner_driver_awaiter())
        # self.heartbeat_task = asyncio.create_task(self.heartbeat())
        self.miner_service = miner_service
        self.validator = Validator()

        self.logging_extra = get_extra_info(
            {
                "validator_hotkey": self.my_hotkey(),
                "compute_app_uri": compute_app_uri,
            }
        )

    def accepted_request_type(self) -> type[BaseRequest]:
        return ContainerBaseRequest

    def connect(self):
        """Create an awaitable/async-iterable websockets.connect() object"""
        logger.info(
            _m(
                "Connecting to backend app",
                extra=self.logging_extra,
            )
        )
        return websockets.connect(self.compute_app_uri)

    async def miner_driver_awaiter(self):
        """avoid memory leak by awaiting miner driver tasks"""
        while True:
            task = await self.miner_drivers.get()
            if task is None:
                return

            try:
                await task
            except Exception as exc:
                logger.error(
                    _m(
                        "Error occurred during driving a miner client",
                        extra={**self.logging_extra, "error": str(exc)},
                    ),
                    exc_info=True,
                )

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.miner_drivers.put(None)
        await self.miner_driver_awaiter_task

    def my_hotkey(self) -> str:
        return self.keypair.ss58_address

    async def run_forever(self) -> NoReturn:
        asyncio.create_task(self.validator.warm_up_subtensor())
        asyncio.create_task(self.subscribe_mesages_from_redis())
        asyncio.create_task(self.poll_rented_machines())

        try:
            while True:
                async for ws in self.connect():
                    try:
                        logger.info(
                            _m(
                                "Connected to backend app",
                                extra=self.logging_extra,
                            )
                        )
                        await self.handle_connection(ws)
                    except websockets.ConnectionClosed as exc:
                        self.ws = None
                        logger.warning(
                            _m(
                                f"validator connection to backend app closed with code {exc.code} and reason {exc.reason}, reconnecting...",
                                extra=self.logging_extra,
                            )
                        )
                    except asyncio.exceptions.CancelledError:
                        self.ws = None
                        logger.warning(
                            _m(
                                "Facilitator client received cancel, stopping",
                                extra=self.logging_extra,
                            )
                        )
                    except Exception:
                        self.ws = None
                        logger.error(
                            _m(
                                "Error in connecting to compute app",
                                extra=self.logging_extra,
                            )
                        )

        except Exception as exc:
            self.ws = None
            logger.error(
                _m(
                    "Connecting to compute app failed",
                    extra={**self.logging_extra, "error": str(exc)},
                ),
                exc_info=True,
            )

    async def handle_connection(self, ws: ClientConnection):
        """handle a single websocket connection"""
        await ws.send(AuthenticateRequest.from_keypair(self.keypair).model_dump_json())

        raw_msg = await ws.recv()
        try:
            response = Response.model_validate_json(raw_msg)
        except pydantic.ValidationError as exc:
            raise AuthenticationError(
                "did not receive Response for AuthenticationRequest", []
            ) from exc
        if response.status != "success":
            raise AuthenticationError("auth request received failed response", response.errors)

        self.ws = ws

        async for raw_msg in ws:
            await self.handle_message(raw_msg)

    async def subscribe_mesages_from_redis(self):
        validator_hotkey = self.my_hotkey()
        queue = []
        
        while True:
            try:
                pubsub = await self.miner_service.redis_service.subscribe(
                    MACHINE_SPEC_CHANNEL,
                    STREAMING_LOG_CHANNEL,
                    RESET_VERIFIED_JOB_CHANNEL,
                )
                async for message in pubsub.listen():
                    try:
                        channel = message['channel'].decode('utf-8')
                        data = json.loads(message['data'])
                    except Exception as exc:
                        continue
                    
                    logger.info(
                        _m(
                            'Received message from redis',
                            extra={
                                **self.logging_extra,
                                "channel": channel,
                            }
                        )
                    )
                    
                    if channel == MACHINE_SPEC_CHANNEL:
                        specs = ExecutorSpecRequest(
                            specs=data["specs"],
                            score=data["score"],
                            synthetic_job_score=data["synthetic_job_score"],
                            log_status=data["log_status"],
                            job_batch_id=data['job_batch_id'],
                            log_text=data["log_text"],
                            miner_hotkey=data["miner_hotkey"],
                            validator_hotkey=validator_hotkey,
                            executor_uuid=data["executor_uuid"],
                            executor_ip=data["executor_ip"],
                            executor_port=data["executor_port"],
                        )
                        queue.append(specs)
                    elif channel == STREAMING_LOG_CHANNEL:
                        log_stream = LogStreamRequest(
                            logs=data["logs"],
                            miner_hotkey=data["miner_hotkey"],
                            validator_hotkey=validator_hotkey,
                            executor_uuid=data["executor_uuid"],
                        )
                        
                        queue.append(log_stream)
                    elif channel == RESET_VERIFIED_JOB_CHANNEL:
                        reset_request = ResetVerifiedJobRequest(
                            miner_hotkey=data["miner_hotkey"],
                            validator_hotkey=validator_hotkey,
                            executor_uuid=data["executor_uuid"],
                        )
                        
                        queue.append(reset_request)
                        
                    if self.ws is not None:
                        while len(queue) > 0:
                            log_to_send = queue.pop(0)
                            try:
                                await self.send_model(log_to_send)
                            except Exception as exc:
                                queue.insert(0, log_to_send)
                                logger.error(
                                    _m(
                                        "Error: message sent error",
                                        extra={
                                            **self.logging_extra,
                                            "error": str(exc),
                                        },
                                    )
                                )
                                break
                        
            except Exception as exc:
                logger.error(
                    _m(
                        "Error: unknown",
                        extra={
                            **self.logging_extra,
                            "error": str(exc),
                        },
                    )
                )
            
            await asyncio.sleep(1)
            
    async def heartbeat(self):
        pass
        # while True:
        #     if self.ws is not None:
        #         try:
        #             await self.send_model(Heartbeat())
        #         except Exception as exc:
        #             msg = f"Error occurred while sending heartbeat: {exc}"
        #             logger.warning(msg)
        #     await asyncio.sleep(self.HEARTBEAT_PERIOD)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(7),
        wait=tenacity.wait_exponential(multiplier=1, exp_base=2, min=1, max=10),
        retry=tenacity.retry_if_exception_type(websockets.ConnectionClosed),
    )
    async def send_model(self, msg: BaseModel):
        if self.ws is None:
            raise websockets.ConnectionClosed(rcvd=None, sent=None)
        await self.ws.send(msg.model_dump_json())
        # Summary: https://github.com/python-websockets/websockets/issues/867
        # Longer discussion: https://github.com/python-websockets/websockets/issues/865
        await asyncio.sleep(0)

    async def poll_rented_machines(self):
        while True:
            if self.ws is not None:
                logger.info(
                    _m(
                        "Request rented machines",
                        extra=self.logging_extra,
                    )
                )
                await self.send_model(RentedMachineRequest())

                logger.info(
                    _m(
                        "Request duplicated machines",
                        extra=self.logging_extra,
                    )
                )
                await self.send_model(DuplicateExecutorsRequest())

                await asyncio.sleep(10 * 60)
            else:
                await asyncio.sleep(10)

    async def handle_message(self, raw_msg: str | bytes):
        """handle message received from facilitator"""
        try:
            response = Response.model_validate_json(raw_msg)
        except pydantic.ValidationError:
            pass
        else:
            if response.status != "success":
                logger.error(
                    _m(
                        "received error response from facilitator",
                        extra={**self.logging_extra, "response": str(response)},
                    )
                )
            return

        try:
            response = pydantic.TypeAdapter(RentedMachineResponse).validate_json(raw_msg)
        except pydantic.ValidationError:
            pass
        else:
            logger.info(
                _m(
                    "Rented machines",
                    extra={**self.logging_extra, "machines": len(response.machines)},
                )
            )

            redis_service = self.miner_service.redis_service
            await redis_service.delete(RENTED_MACHINE_PREFIX)

            for machine in response.machines:
                await redis_service.add_rented_machine(machine)

            return

        try:
            response = pydantic.TypeAdapter(DuplicateExecutorsResponse).validate_json(raw_msg)
        except pydantic.ValidationError:
            pass
        else:
            logger.info(
                _m(
                    "Duplicated executors",
                    extra={**self.logging_extra, "executors": len(response.executors)},
                )
            )

            redis_service = self.miner_service.redis_service
            await redis_service.delete(DUPLICATED_MACHINE_SET)

            for _, details_list in response.executors.items():
                for detail in details_list:
                    executor_id = detail.get("executor_id")
                    miner_hotkey = detail.get("miner_hotkey")
                    await redis_service.sadd(
                        DUPLICATED_MACHINE_SET, f"{miner_hotkey}:{executor_id}"
                    )

            return

        try:
            job_request = self.accepted_request_type().parse(raw_msg)
        except Exception as ex:
            error_msg = f"Invalid message received from celium backend: {str(ex)}"
            logger.error(
                _m(
                    error_msg,
                    extra={**self.logging_extra, "error": str(ex), "raw_msg": raw_msg},
                )
            )
        else:
            task = asyncio.create_task(self.miner_driver(job_request))
            await self.miner_drivers.put(task)
            return
        # logger.error("unsupported message received from facilitator: %s", raw_msg)

    async def get_miner_axon_info(self, hotkey: str) -> bittensor.AxonInfo:
        miners = self.validator.fetch_miners()
        neurons = [n for n in miners if n.hotkey == hotkey]
        if not neurons:
            raise ValueError(f"Miner with {hotkey=} not present in this subnetwork")
        return neurons[0].axon_info

    async def miner_driver(
        self,
        job_request: ContainerCreateRequest
        | ContainerDeleteRequest
        | ContainerStopRequest
        | ContainerStartRequest,
    ):
        """drive a miner client from job start to completion, then close miner connection"""
        logger.info(
            _m(f"Getting miner axon info for {job_request.miner_hotkey}", extra=self.logging_extra)
        )
        miner_axon_info = await self.get_miner_axon_info(job_request.miner_hotkey)
        logging_extra = {
            **self.logging_extra,
            "miner_hotkey": job_request.miner_hotkey,
            "miner_ip": miner_axon_info.ip,
            "miner_port": miner_axon_info.port,
            "job_request": str(job_request),
            "executor_id": str(job_request.executor_id),
        }
        logger.info(
            _m(
                "Miner driver to miner",
                extra=logging_extra,
            )
        )

        if isinstance(job_request, ContainerCreateRequest):
            logger.info(
                _m(
                    "Creating container for executor.",
                    extra={**logging_extra, "job_request": str(job_request)},
                )
            )
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            container_created: (
                ContainerCreated | FailedContainerRequest
            ) = await self.miner_service.handle_container(job_request)

            logger.info(
                _m(
                    "Sending back created container info to compute app",
                    extra={**logging_extra, "container_created": str(container_created)},
                )
            )
            await self.send_model(container_created)
        elif isinstance(job_request, ContainerDeleteRequest):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                ContainerDeleteRequest | FailedContainerRequest
            ) = await self.miner_service.handle_container(job_request)

            logger.info(
                _m(
                    "Sending back deleted container info to compute app",
                    extra={**logging_extra, "response": str(response)},
                )
            )
            await self.send_model(response)
        elif isinstance(job_request, ContainerStopRequest):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                ContainerStopRequest | FailedContainerRequest
            ) = await self.miner_service.handle_container(job_request)

            logger.info(
                _m(
                    "Sending back stopped container info to compute app",
                    extra={**logging_extra, "response": str(response)},
                )
            )
            await self.send_model(response)
        elif isinstance(job_request, ContainerStartRequest):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                ContainerStartRequest | FailedContainerRequest
            ) = await self.miner_service.handle_container(job_request)

            logger.info(
                _m(
                    "Sending back started container info to compute app",
                    extra={**logging_extra, "response": str(response)},
                )
            )
            await self.send_model(response)
