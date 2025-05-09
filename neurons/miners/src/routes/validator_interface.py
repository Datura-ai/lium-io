from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket

from consumers.validator_consumer import ValidatorConsumer
validator_router = APIRouter()


@validator_router.websocket("/websocket/{validator_key}")
async def validator_interface(consumer: Annotated[ValidatorConsumer, Depends(ValidatorConsumer)]):
    await consumer.connect()
    await consumer.handle()
