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


@apis_router.post("/hardware_metrics")
async def get_hardware_metrics():
    import pynvml
    import psutil
    
    metrics = {
        "cpu": psutil.cpu_percent(interval=1),
        "memory": psutil.virtual_memory().percent,
        "gpu": []
    }
    
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            metrics["gpu"].append({
                "gpu": util.gpu,
                "memory": util.memory,
            })
        pynvml.nvmlShutdown()
    except:
        pass
    
    return metrics
