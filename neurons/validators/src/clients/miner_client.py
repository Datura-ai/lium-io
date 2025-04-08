import abc
import asyncio
import logging
import random
import time
from tenacity import retry, stop_after_attempt, wait_fixed

import bittensor
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.protocol import State as WebSocketClientState
from datura.errors.protocol import UnsupportedMessageReceived
from datura.requests.base import BaseRequest
from datura.requests.miner_requests import (
    AcceptJobRequest,
    AcceptSSHKeyRequest,
    BaseMinerRequest,
    DeclineJobRequest,
    FailedRequest,
    GenericError,
    SSHKeyRemoved,
    UnAuthorizedRequest,
)
from datura.requests.validator_requests import AuthenticateRequest, AuthenticationPayload

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)


class JobState:
    def __init__(self):
        self.miner_ready_or_declining_future = asyncio.Future()
        self.miner_ready_or_declining_timestamp: int = 0
        self.miner_accepted_ssh_key_or_failed_future = asyncio.Future()
        self.miner_accepted_ssh_key_or_failed_timestamp: int = 0
        self.miner_removed_ssh_key_future = asyncio.Future()


class MinerClient(abc.ABC):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        miner_address: str,
        my_hotkey: str,
        miner_hotkey: str,
        miner_port: int,
        keypair: bittensor.Keypair,
        miner_url: str,
    ):
        self.debounce_counter = 0
        self.max_debounce_count: int | None = 5  # set to None for unlimited debounce
        self.loop = loop
        self.miner_name = f"{miner_hotkey}({miner_address}:{miner_port})"
        self.ws: ClientConnection | None = None
        self.ws_task: asyncio.Task | None = None

        self.miner_hotkey = miner_hotkey
        self.my_hotkey = my_hotkey
        self.miner_address = miner_address
        self.miner_port = miner_port
        self.keypair = keypair

        self.miner_url = miner_url

        self.job_state = JobState()

        self.logging_extra = {
            "miner_hotkey": miner_hotkey,
            "miner_address": miner_address,
            "miner_port": miner_port,
        }

    def accepted_request_type(self) -> type[BaseRequest]:
        return BaseMinerRequest

    async def handle_message(self, msg: BaseRequest):
        """
        Handle the message based on its type or raise UnsupportedMessageReceived
        """
        if isinstance(msg, AcceptJobRequest):
            if not self.job_state.miner_ready_or_declining_future.done():
                self.job_state.miner_ready_or_declining_timestamp = time.time()
                self.job_state.miner_ready_or_declining_future.set_result(msg)
        elif isinstance(
            msg, AcceptSSHKeyRequest | FailedRequest | UnAuthorizedRequest | DeclineJobRequest
        ):
            if not self.job_state.miner_accepted_ssh_key_or_failed_future.done():
                self.job_state.miner_accepted_ssh_key_or_failed_timestamp = time.time()
                self.job_state.miner_accepted_ssh_key_or_failed_future.set_result(msg)
        elif isinstance(msg, SSHKeyRemoved):
            if not self.job_state.miner_removed_ssh_key_future.done():
                self.job_state.miner_removed_ssh_key_future.set_result(msg)

    async def __aenter__(self):
        self.ws_task = asyncio.create_task(self.await_connect())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.ws_task is not None and not self.ws_task.done():
            self.ws_task.cancel()

        if self.ws is not None and self.ws.state is WebSocketClientState.OPEN:
            try:
                await self.ws.close()
            except Exception:
                pass

    def generate_authentication_message(self) -> AuthenticateRequest:
        """Generate authentication request/message for miner."""
        payload = AuthenticationPayload(
            validator_hotkey=self.my_hotkey,
            miner_hotkey=self.miner_hotkey,
            timestamp=int(time.time()),
        )
        return AuthenticateRequest(
            payload=payload, signature=f"0x{self.keypair.sign(payload.blob_for_signing()).hex()}"
        )

    async def _connect(self):
        ws = await websockets.connect(self.miner_url, max_size=50 * (2**20))  # 50MB
        await ws.send(self.generate_authentication_message().json())
        return ws

    async def await_connect(self):
        start_time = time.time()
        while True:
            try:
                async with websockets.connect(self.miner_url, max_size=50 * (2**20)) as ws:
                    if self.debounce_counter:
                        logger.info(
                            _m(
                                f"Connected to miner after {self.debounce_counter + 1} attempts",
                                extra=get_extra_info(self.logging_extra),
                            )
                        )

                    self.ws = ws
                    await ws.send(self.generate_authentication_message().json())

                    async for raw_msg in self.ws:
                        await self.read_messages(raw_msg)
            except (websockets.WebSocketException, OSError) as ex:
                self.debounce_counter += 1

                if (
                    self.max_debounce_count is not None
                    and self.debounce_counter > self.max_debounce_count
                ):
                    time_took = time.time() - start_time
                    raise Exception(
                        f"Could not connect to miner {self.miner_name} after {self.max_debounce_count} tries"
                        f" in {time_took:0.2f} seconds"
                    )
                if self.debounce_counter:
                    sleep_time = self.sleep_time()
                    logger.info(
                        _m(
                            f"Retrying connection to miner in {sleep_time:0.2f}",
                            extra=get_extra_info(self.logging_extra)
                        )
                    )
                    await asyncio.sleep(sleep_time)

    def sleep_time(self):
        return 2 + random.random()

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
    async def send_model(self, model: BaseRequest):
        await self.ws.send(model.json())

    async def read_messages(self, msg):
        try:
            msg = self.accepted_request_type().parse(msg)
        except Exception as ex:
            error_msg = f"Malformed message from miner: {str(ex)}"
            logger.error(
                _m(
                    error_msg,
                    extra=get_extra_info({**self.logging_extra, "error": str(ex)}),
                )
            )
            return

        if isinstance(msg, GenericError):
            await self.ws.close()
            raise RuntimeError(f"Received error message from miner: {msg.json()}")

        try:
            await self.handle_message(msg)
        except UnsupportedMessageReceived:
            error_msg = "Unsupported message from miner"
            logger.error(_m(error_msg, extra=get_extra_info(self.logging_extra)))
        else:
            if self.debounce_counter:
                logger.info(
                    _m(
                        f"Receviced valid message from miner after {self.debounce_counter + 1} connection attempts",
                        extra=get_extra_info(self.logging_extra),
                    )

                )
                self.debounce_counter = 0
