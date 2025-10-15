import abc
import logging

from fastapi import WebSocket, WebSocketDisconnect

from ..requests.base import BaseRequest

logger = logging.getLogger(__name__)


class BaseConsumer(abc.ABC):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    @abc.abstractmethod
    def accepted_request_type(self) -> type[BaseRequest]:
        pass

    async def connect(self):
        await self.websocket.accept()

    async def receive_message(self) -> BaseRequest:
        data = await self.websocket.receive_text()
        return self.accepted_request_type().parse(data)

    async def send_message(self, msg: BaseRequest):
        await self.websocket.send_text(msg.json())

    async def disconnect(self):
        try:
            await self.websocket.close()
        except Exception:
            pass

    @abc.abstractmethod
    async def handle_message(self, data: BaseRequest):
        raise NotImplementedError

    async def handle(self):
        # await self.connect()
        try:
            while True:
                data: BaseRequest = await self.receive_message()
                await self.handle_message(data)
        except WebSocketDisconnect as ex:
            logger.debug("Websocket connection closed, ex: %s", str(ex))
            await self.disconnect()
        except Exception as ex:
            logger.debug("Handling message error: %s", str(ex))
            await self.disconnect()
