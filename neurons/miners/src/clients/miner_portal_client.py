import asyncio
import uuid
import logging
from typing import NoReturn

import tenacity
import websockets

from pydantic import BaseModel
from websockets.asyncio.client import ClientConnection

from core.config import settings
from core.utils import _m, get_extra_info
from services.ioc import ioc

from models.executor import Executor

from daos.executor import ExecutorDao

from protocol.miner_request import (
    AuthenticateRequest,
)

from protocol.miner_portal_request import (
    BaseMinerPortalRequest,
    AddExecutorRequest,
    ExecutorAdded,
    AddExecutorFailed,
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

        self.executor_dao: ExecutorDao = ioc["ExecutorDao"]

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
            try:
                executor = self.executor_dao.save(Executor(
                    uuid=uuid.uuid4(),
                    address=request.payload.ip_address,
                    port=request.payload.port,
                    validator=request.payload.validator_hotkey,
                ))
                logger.info("Added executor (id=%s)", str(executor.uuid))
                self.message_queue.append(ExecutorAdded(
                    executor_id=executor.uuid,
                    ip_address=executor.address,
                    port=executor.port,
                ))
            except Exception as e:
                logger.error(_m(
                    "❌ Failed to add executor",
                    extra={
                        **self.logging_extra,
                        "address": request.payload.ip_address,
                        "port": request.payload.port,
                        "validator": request.payload.validator_hotkey,
                        "error": str(e),
                    }
                ))
                self.message_queue.append((
                    AddExecutorFailed(
                        ip_address=executor.address,
                        port=executor.port,
                        error=str(e)
                    )
                ))
