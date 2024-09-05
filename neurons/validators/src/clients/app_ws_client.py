import abc
import asyncio
import logging
import logging
import random
import websockets
from datura.requests.base import BaseRequest

from core.config import settings

logger = logging.getLogger(__name__)


class AppWsClient(abc.ABC):
    ws: websockets.WebSocketClientProtocol | None = None
    read_messages_task: asyncio.Task | None = None

    def __init__(self):
        self.ss58_address: str = settings.get_bittensor_wallet().get_hotkey().ss58_address
        self.ws_uri = f"{settings.APP_WS_URI}/{self.ss58_address}"
        self.should_exit = False

    async def connect(self):
        while not self.should_exit:
            try:
                if self.ws is not None and not self.ws.closed:
                    continue

                self.ws = websockets.connect(self.ws_uri)
                self.read_messages_task = asyncio.create_task(
                    self.read_messages()
                )

                await asyncio.sleep(5)
            except (websockets.ConnectionClosedError, ConnectionRefusedError):
                print("Connection lost, retrying in 5 seconds...")
                await asyncio.sleep(5)

    async def disconnect(self):
        self.should_exit = True

        if self.read_messages_task is not None and not self.read_messages_task.done():
            self.read_messages_task.cancel()

        if self.ws is not None and not self.ws.closed:
            try:
                await self.ws.close()
            except Exception:
                pass

    def accepted_request_type(self) -> type[BaseRequest]:
        return BaseRequest

    async def send_message(self, model: BaseRequest):
        while True:
            try:
                await self.ws.send(model.model_dump_json())
            except websockets.WebSocketException as ex:
                logger.error(
                    f"Could not send to miner {self.miner_name}: {str(ex)}")
                await asyncio.sleep(1 + random.random())
                continue
            return

    async def read_messages(self):
        while True:
            if self.ws is not None and not self.ws.closed:
                return

            msg = await self.ws.recv()

            try:
                msg = self.accepted_request_type().parse(msg)
            except Exception as ex:
                error_msg = f"Malformed message from app: {str(ex)}"
                logger.info(error_msg)
                continue


app_ws_client = AppWsClient()
