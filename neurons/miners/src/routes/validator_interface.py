from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket

from consumers.validator_consumer import ValidatorConsumer
validator_router = APIRouter()


@validator_router.websocket("/jobs/{validator_key}")
async def validator_interface(consumer: Annotated[ValidatorConsumer, Depends(ValidatorConsumer)]):
    await consumer.connect()
    await consumer.handle()


@validator_router.websocket("/resources/{validator_key}")
async def validator_interface(consumer: Annotated[ValidatorConsumer, Depends(ValidatorConsumer)]):
    await consumer.connect()
    await consumer.handle()
