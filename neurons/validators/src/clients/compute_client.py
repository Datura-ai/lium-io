import asyncio
import json
import logging
from typing import NoReturn

import bittensor
import pydantic
import redis.asyncio as aioredis
import tenacity
import websockets
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    ContainerDeleteRequest,
    FailedContainerRequest,
)
from protocol.vc_protocol.compute_requests import Error, Response
from protocol.vc_protocol.validator_requests import AuthenticateRequest, ExecutorSpecRequest
from pydantic import BaseModel

from clients.metagraph_client import create_metagraph_refresh_task, get_miner_axon_info
from core.config import settings
from core.utils import get_extra_info
from services.miner_service import MinerService

logger = logging.getLogger(__name__)

MACHINE_SPEC_CHANNEL_NAME = "channel:1"


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
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.compute_app_uri = compute_app_uri
        self.redis = aioredis.from_url(f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}")
        self.miner_drivers = asyncio.Queue()
        self.miner_driver_awaiter_task = asyncio.create_task(self.miner_driver_awaiter())
        # self.heartbeat_task = asyncio.create_task(self.heartbeat())
        self.refresh_metagraph_task = self.create_metagraph_refresh_task()
        self.miner_service = miner_service

        self.logging_extra = {
            "compute_app_uri": compute_app_uri,
        }

    def connect(self):
        """Create an awaitable/async-iterable websockets.connect() object"""
        logger.info(
            f"Connecting to {self.compute_app_uri}", extra=get_extra_info(self.logging_extra)
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
                    "Error occurred during driving a miner client",
                    extra=get_extra_info({**self.logging_extra, "error": str(exc)}),
                )

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.miner_drivers.put(None)
        await self.miner_driver_awaiter_task

    def my_hotkey(self) -> str:
        return self.keypair.ss58_address

    async def run_forever(self) -> NoReturn:
        """connect (and re-connect) to facilitator and keep reading messages ... forever"""
        # subscribe to channel to get machine specs
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(MACHINE_SPEC_CHANNEL_NAME)

        # send machine specs to facilitator
        self.specs_task = asyncio.create_task(self.wait_for_specs(pubsub))
        try:
            while True:
                async for ws in self.connect():
                    try:
                        logger.info(
                            f"Connected to {self.compute_app_uri}",
                            extra=get_extra_info(self.logging_extra),
                        )
                        await self.handle_connection(ws)
                    except websockets.ConnectionClosed as exc:
                        self.ws = None
                        logger.warning(
                            f"validator connection closed with code {exc.code} and reason {exc.reason}, reconnecting...",
                            extra=get_extra_info(self.logging_extra),
                        )
                    except asyncio.exceptions.CancelledError:
                        self.ws = None
                        logger.warning(
                            "Facilitator client received cancel, stopping",
                            extra=get_extra_info(self.logging_extra),
                        )
                    except Exception:
                        self.ws = None
                        logger.error(
                            "Error in connecting to compute app",
                            extra=get_extra_info(self.logging_extra),
                        )

        except asyncio.exceptions.CancelledError:
            self.ws = None
            logger.error(
                "Facilitator client received cancel, stopping",
                extra=get_extra_info(self.logging_extra),
            )

    async def handle_connection(self, ws: websockets.WebSocketClientProtocol):
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

    async def wait_for_specs(self, channel: aioredis.client.PubSub):
        specs_queue = []
        while True:
            validator_hotkey = settings.get_bittensor_wallet().hotkey.ss58_address
            default_extra = {
                **self.logging_extra,
                "validator_hotkey": validator_hotkey,
            }
            logger.info(
                f"Waiting for machine specs from validator app: {validator_hotkey}",
                extra=get_extra_info(default_extra),
            )
            try:
                msg = await channel.get_message(ignore_subscribe_messages=True, timeout=100 * 60)
                logger.info(
                    "Received machine specs from validator app.",
                    extra=get_extra_info({**default_extra, "msg": str(msg)}),
                )

                if msg is None:
                    logger.warning(
                        "No message received from validator app.",
                        extra=get_extra_info(default_extra),
                    )
                    continue

                msg = json.loads(msg["data"])
                specs = None
                executor_logging_extra = {}
                try:
                    specs = ExecutorSpecRequest(
                        specs=msg["specs"],
                        miner_hotkey=msg["miner_hotkey"],
                        validator_hotkey=validator_hotkey,
                        executor_uuid=msg["executor_uuid"],
                        executor_ip=msg["executor_ip"],
                        executor_port=msg["executor_port"],
                    )
                    executor_logging_extra = {
                        "executor_uuid": msg["executor_uuid"],
                        "executor_ip": msg["executor_ip"],
                        "executor_port": msg["executor_port"],
                    }
                except Exception as exc:
                    msg = "Error occurred while parsing msg"
                    logger.error(
                        msg,
                        extra=get_extra_info(
                            {**default_extra, **executor_logging_extra, "error": str(exc)}
                        ),
                    )
                    continue

                logger.info(
                    "Sending machine specs update of executor to compute app",
                    extra=get_extra_info(
                        {**default_extra, **executor_logging_extra, "specs": str(specs)}
                    ),
                )

                specs_queue.append(specs)
                if self.ws is not None:
                    while len(specs_queue) > 0:
                        spec_to_send = specs_queue.pop(0)
                        try:
                            await self.send_model(spec_to_send)
                        except Exception as exc:
                            specs_queue.insert(0, spec_to_send)
                            msg = "Error occurred while sending specs of executor"
                            logger.error(
                                msg,
                                extra=get_extra_info(
                                    {**default_extra, **executor_logging_extra, "error": str(exc)}
                                ),
                            )
                            break
            except TimeoutError:
                logger.error("wait_for_specs still running", extra=get_extra_info(default_extra))

    def create_metagraph_refresh_task(self, period=None):
        return create_metagraph_refresh_task(period=period)

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

    async def handle_message(self, raw_msg: str | bytes):
        """handle message received from facilitator"""
        try:
            response = Response.model_validate_json(raw_msg)
        except pydantic.ValidationError:
            logger.info(
                "could not parse raw message as Response",
                extra=get_extra_info({**self.logging_extra, "raw_msg": raw_msg}),
            )
        else:
            if response.status != "success":
                logger.error(
                    "received error response from facilitator",
                    extra=get_extra_info({**self.logging_extra, "response": str(response)}),
                )
            return

        try:
            job_request = pydantic.TypeAdapter(ContainerCreateRequest).validate_json(raw_msg)
        except pydantic.ValidationError as exc:
            logger.error(
                "could not parse raw message as ContainerCreateRequest",
                extra=get_extra_info({**self.logging_extra, "error": str(exc), "raw_msg": raw_msg}),
            )
        else:
            task = asyncio.create_task(self.miner_driver(job_request))
            await self.miner_drivers.put(task)
            return

        try:
            job_request = pydantic.TypeAdapter(ContainerDeleteRequest).validate_json(raw_msg)
        except pydantic.ValidationError as exc:
            logger.error(
                "could not parse raw message as ContainerDeleteRequest",
                extra=get_extra_info({**self.logging_extra, "error": str(exc), "raw_msg": raw_msg}),
            )
        else:
            task = asyncio.create_task(self.miner_driver(job_request))
            await self.miner_drivers.put(task)
            return

        # logger.error("unsupported message received from facilitator: %s", raw_msg)

    async def get_miner_axon_info(self, hotkey: str) -> bittensor.AxonInfo:
        return await get_miner_axon_info(hotkey)

    async def miner_driver(self, job_request: ContainerCreateRequest | ContainerDeleteRequest):
        """drive a miner client from job start to completion, then close miner connection"""
        miner_axon_info = await self.get_miner_axon_info(job_request.miner_hotkey)
        logging_extra = {
            "miner_hotkey": job_request.miner_hotkey,
            "miner_ip": miner_axon_info.ip,
            "miner_port": miner_axon_info.port,
            "job_request": job_request,
            "executor_id": job_request.executor_id,
        }
        logger.info("Miner driver to miner", extra=get_extra_info(logging_extra))

        if isinstance(job_request, ContainerCreateRequest):
            logger.info(
                "Creating container for executor.",
                extra=get_extra_info({**logging_extra, "job_request": str(job_request)}),
            )
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            container_created: ContainerCreated = await self.miner_service.handle_container(
                job_request
            )

            logger.info(
                "Sending back created container info to compute app",
                extra=get_extra_info(
                    {**logging_extra, "container_created": str(container_created)}
                ),
            )
            await self.send_model(container_created)
        elif isinstance(job_request, ContainerDeleteRequest):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                ContainerDeleteRequest | FailedContainerRequest
            ) = await self.miner_service.handle_container(job_request)

            logger.info(
                "Sending back deleted container info to compute app",
                extra=get_extra_info({**logging_extra, "response": str(response)}),
            )
            await self.send_model(response)
