from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket

from consumers.validator_consumer import validatorConsumerManager, ValidatorConsumer
from services.ssh_service import MinerSSHService
from services.validator_service import ValidatorService

validator_router = APIRouter()


@validator_router.websocket("/jobs/{validator_key}")
async def validator_interface(
    websocket: WebSocket,
    validator_key: str,
    ssh_service: Annotated[MinerSSHService, Depends(MinerSSHService)],
    validator_service: Annotated[ValidatorService, Depends(ValidatorService)],
):
    await validatorConsumerManager.addConsumer(
        websocket=websocket,
        validator_key=validator_key,
        ssh_service=ssh_service,
        validator_service=validator_service,
    )

@validator_router.websocket("/resources/{validator_key}")
async def validator_interface(consumer: Annotated[ValidatorConsumer, Depends(ValidatorConsumer)]):
    await consumer.handle()