import asyncio
import logging
from typing import NoReturn, Union

import tenacity
import websockets

from pydantic import BaseModel
from websockets.asyncio.client import ClientConnection

from core.config import settings
from core.utils import _m, get_extra_info

from services.ioc import ioc
from services.executor_service import ExecutorService

from models.executor import Executor

from protocol.miner_request import (
    AuthenticateRequest,
)

from protocol.miner_portal_request import (
    BaseMinerPortalRequest,
    AddExecutorRequest,
    ExecutorAdded,
    AddExecutorFailed,
    SyncExecutorMinerPortalRequest,
    SyncExecutorMinerPortalSuccess,
    SyncExecutorMinerPortalFailed,
    SyncExecutorCentralMinerRequest,
    SyncExecutorCentralMinerSuccess,
    SyncExecutorCentralMinerFailed,
)

logger = logging.getLogger(__name__)

CONNECT_INTERVAL = 1


class MinerPortalClient:
    HEARTBEAT_PERIOD = 60

    def __init__(
        self,
    ):
        self.keypair = settings.get_bittensor_wallet().get_hotkey()
        self.hotkey = self.keypair.ss58_address

        self.ws: ClientConnection | None = None
        self.miner_portal_uri = f"{settings.MINER_PORTAL_URI}/api/miners/{self.hotkey}"
        self.message_queue = []
        self.lock = asyncio.Lock()

        self.executor_service: ExecutorService = ioc["ExecutorService"]

        self.logging_extra = {
            "miner_hotkey": self.hotkey,
            "miner_portal_uri": self.miner_portal_uri,
        }

    def connect(self):
        logger.info(
            _m(
                "Connecting to miner portal backend",
                extra=get_extra_info(self.logging_extra)
            )
        )
        authRequest = AuthenticateRequest.from_keypair(self.keypair)
        return websockets.connect(
            self.miner_portal_uri,
            additional_headers={
                "hotkey": authRequest.payload.miner_hotkey,
                "timestamp": authRequest.payload.timestamp,
                "signature": authRequest.signature,
            }
        )

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def run_forever(self) -> NoReturn:
        asyncio.create_task(self.handle_send_messages())

        while True:
            try:
                async for ws in self.connect():
                    try:
                        logger.info(
                            _m(
                                "Connected to miner portal",
                                extra=get_extra_info(self.logging_extra),
                            )
                        )

                        self.ws = ws
                        async for raw_msg in ws:
                            await self.handle_message(raw_msg)
                    except Exception as e:
                        self.ws = None
                        logger.error(
                            _m(
                                "connection closed",
                                extra=get_extra_info({
                                    **self.logging_extra,
                                    "error": str(e),
                                }),
                            )
                        )
                    finally:
                        await asyncio.sleep(CONNECT_INTERVAL)
            except Exception as exc:
                self.ws = None
                logger.error(
                    _m(
                        "connection failed",
                        extra=get_extra_info({**self.logging_extra, "error": str(exc)}),
                    ),
                    exc_info=True,
                )
                await asyncio.sleep(CONNECT_INTERVAL)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, exp_base=2, min=1, max=10),
        retry=tenacity.retry_if_exception_type(websockets.ConnectionClosed),
    )
    async def send_model(self, msg: BaseModel):
        await self.ws.send(msg.model_dump_json())

    async def handle_send_messages(self):
        while True:
            if self.ws is None:
                await asyncio.sleep(CONNECT_INTERVAL)
                continue

            if len(self.message_queue) > 0:
                async with self.lock:
                    log_to_send = self.message_queue.pop(0)

                if log_to_send:
                    try:
                        await self.send_model(log_to_send)
                    except Exception as exc:
                        async with self.lock:
                            self.message_queue.insert(0, log_to_send)
                        logger.error(
                            _m(
                                "Error: message sent error",
                                extra=get_extra_info({
                                    **self.logging_extra,
                                    "error": str(exc),
                                }),
                            )
                        )
            else:
                await asyncio.sleep(1)

    def accepted_request_type(self) -> type[BaseMinerPortalRequest]:
        return BaseMinerPortalRequest

    async def handle_message(self, raw_msg: str | bytes):
        """handle message received from miner portal"""
        try:
            request = self.accepted_request_type().parse(raw_msg)
        except Exception as ex:
            error_msg = f"Invalid message received from miner portal: {str(ex)}"
            logger.error(
                _m(
                    error_msg,
                    extra=get_extra_info({**self.logging_extra, "error": str(ex), "raw_msg": raw_msg}),
                )
            )
            return

        if isinstance(request, AddExecutorRequest):
            result: Union[ExecutorAdded, AddExecutorFailed] = self.executor_service.create(
                Executor(
                    uuid=request.executor_id,
                    address=request.payload.ip_address,
                    port=request.payload.port,
                    validator=request.payload.validator_hotkey,
                )
            )
            self.message_queue.append(result)

        if isinstance(request, SyncExecutorMinerPortalRequest):
            logger.info("Sync executor miner portal request received")
            result: Union[SyncExecutorMinerPortalSuccess, SyncExecutorMinerPortalFailed] = self.executor_service.sync_executor_miner_portal(request)
            self.message_queue.append(result)

        if isinstance(request, SyncExecutorCentralMinerRequest):
            result: Union[SyncExecutorCentralMinerSuccess, SyncExecutorCentralMinerFailed] = self.executor_service.sync_executor_central_miner(self.hotkey, request)
            self.message_queue.append(result)
            logger.info("Sync executor central miner response sent")