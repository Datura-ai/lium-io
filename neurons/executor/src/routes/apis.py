from typing import Annotated

from fastapi import APIRouter, Depends
from services.miner_service import MinerService
from services.pod_log_service import PodLogService

from payloads.miner import UploadSShKeyPayload, GetPodLogsPaylod

apis_router = APIRouter()


@apis_router.post("/upload_ssh_key")
async def upload_ssh_key(
    payload: UploadSShKeyPayload, miner_service: Annotated[MinerService, Depends(MinerService)]
):
    return await miner_service.upload_ssh_key(payload)


@apis_router.post("/remove_ssh_key")
async def remove_ssh_key(
    payload: UploadSShKeyPayload, miner_service: Annotated[MinerService, Depends(MinerService)]
):
    return await miner_service.remove_ssh_key(payload)


@apis_router.post("/pod_logs")
async def get_pod_logs(
    payload: GetPodLogsPaylod, pod_log_service: Annotated[PodLogService, Depends(PodLogService)]
):
    return await pod_log_service.find_by_continer_name(payload.container_name)
