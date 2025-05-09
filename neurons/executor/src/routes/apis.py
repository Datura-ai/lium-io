from typing import Annotated

from fastapi import APIRouter, Depends
from services.miner_service import MinerService

from payloads.miner import MinerAuthPayload

apis_router = APIRouter()


@apis_router.post("/upload_ssh_key")
async def upload_ssh_key(
    payload: MinerAuthPayload, miner_service: Annotated[MinerService, Depends(MinerService)]
):
    return await miner_service.upload_ssh_key(payload)


@apis_router.post("/remove_ssh_key")
async def remove_ssh_key(
    payload: MinerAuthPayload, miner_service: Annotated[MinerService, Depends(MinerService)]
):
    return await miner_service.remove_ssh_key(payload)
