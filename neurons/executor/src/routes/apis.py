from typing import Annotated, Optional

import docker
from fastapi import APIRouter, Depends, Query, Header, HTTPException
from fastapi.responses import StreamingResponse
from services.miner_service import MinerService
from services.pod_log_service import PodLogService
from services.hardware_service import get_system_metrics, get_container_metrics

from payloads.miner import UploadSShKeyPayload, GetPodLogsPaylod
from payloads.backend import ContainerUtilizationPayload
from dependencies.auth import verify_allowed_hotkey_signature, verify_ping_signature, verify_container_signature, verify_container_logs_signature

apis_router = APIRouter()


@apis_router.post("/upload_ssh_key")
async def upload_ssh_key(
    payload: UploadSShKeyPayload, miner_service: Annotated[MinerService, Depends(MinerService)]
):
    if payload.public_key != payload.data_to_sign:
        raise HTTPException(status_code=400, detail="Public key mismatch")

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


@apis_router.post("/hardware_utilization")
async def hardware_utilization(
    _: None = Depends(verify_allowed_hotkey_signature)
):
    """
    Endpoint for hardware utilization that requires signature from allowed Bittensor hotkey.
    This endpoint bypasses the MinerMiddleware authentication.

    Returns:
        dict: Hardware utilization metrics including CPU, memory, storage, and GPU
    """
    return get_system_metrics()


@apis_router.post("/containers/{container_name}")
async def container_hardware_utilization(
    container_name: str,
    payload: ContainerUtilizationPayload,
    _: None = Depends(verify_container_signature)
):
    """
    Endpoint for container-specific hardware utilization.
    Returns CPU, memory, and GPU metrics for the specified container only.

    Args:
        container_name: Name of the Docker container
        payload: Contains gpu_uuids list and signature for verification

    Returns:
        dict: Container-specific hardware utilization metrics
    """
    return get_container_metrics(container_name, payload.gpu_uuids)


@apis_router.post("/ping")
async def ping(_: None = Depends(verify_ping_signature)):
    """
    Simple ping-pong endpoint for checking executor availability with signature verification.
    Requires signature from allowed Bittensor hotkey.

    Returns:
        dict: {"status": "pong"}
    """
    return {"status": "pong"}


@apis_router.get("/containers/{container_name}/logs")
async def stream_container_logs(
    container_name: str,
    x_signature: str = Header(..., description="Signature for authentication"),
    x_timestamp: int = Header(..., description="Unix timestamp used in signature"),
    follow: bool = Query(False, description="Follow log output"),
    tail: Optional[int] = Query(None, description="Number of lines to show from the end"),
    since: Optional[int] = Query(None, description="Unix timestamp to show logs since"),
    stdout: bool = Query(True, description="Include stdout"),
    stderr: bool = Query(True, description="Include stderr"),
):
    """
    Stream logs from a Docker container.

    Args:
        container_name: Name of the Docker container
        x_signature: Signature header for authentication
        x_timestamp: Timestamp header used in signature
        follow: Keep streaming new logs (like docker logs -f)
        tail: Number of lines from end (like docker logs --tail)
        since: Unix timestamp to filter logs
        stdout: Include stdout logs
        stderr: Include stderr logs

    Returns:
        StreamingResponse: Plain text stream of container logs
    """
    await verify_container_logs_signature(container_name, x_timestamp, x_signature)

    client = docker.from_env()
    container = client.containers.get(container_name)

    def generate():
        for log in container.logs(
            stream=True,
            follow=follow,
            tail=tail if tail else "all",
            since=since,
            stdout=stdout,
            stderr=stderr,
        ):
            yield log

    return StreamingResponse(generate(), media_type="text/plain")
