from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket

from consumers.validator_consumer import validatorConsumerManager, ValidatorConsumer
from services.ssh_service import MinerSSHService
from services.validator_service import ValidatorService
from services.executor_service import ExecutorService

validator_router = APIRouter()


@validator_router.websocket("/jobs/{validator_key}")
async def validator_interface(
    websocket: WebSocket,
    validator_key: str,
    ssh_service: Annotated[MinerSSHService, Depends(MinerSSHService)],
    validator_service: Annotated[ValidatorService, Depends(ValidatorService)],
    executor_service: Annotated[ExecutorService, Depends(ExecutorService)],
):
    await validatorConsumerManager.addConsumer(
        websocket=websocket,
        validator_key=validator_key,
        ssh_service=ssh_service,
        validator_service=validator_service,
        executor_service=executor_service,
    )

@validator_router.websocket("/resources/{validator_key}")
async def validator_interface(consumer: Annotated[ValidatorConsumer, Depends(ValidatorConsumer)]):
    await consumer.connect()
    await consumer.handle()