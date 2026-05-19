import logging
from typing import Annotated, Optional
import tomllib
from pathlib import Path

import docker
import bittensor
from fastapi import APIRouter, Depends, Query, Header, HTTPException
from fastapi.responses import StreamingResponse
from core.logger import _m
from services.miner_service import MinerService
from services.pod_log_service import PodLogService
from services.chutes_relay_service import (
    ChutesBridgeConfigError,
    ChutesExecutorError,
    ChutesMalformedResponseError,
    ChutesRelayDisabledError,
    ChutesRelayService,
    ChutesTransportError,
)
from services.hardware_service import get_system_metrics, get_container_metrics
from core.config import VALIDATOR_HOTKEY_SS58

from payloads.miner import UploadSShKeyPayload, GetPodLogsPaylod
from payloads.backend import ContainerUtilizationPayload
from payloads.chutes import ChutesCommandPayload, ChutesInstallPayload
from dependencies.auth import (
    verify_allowed_hotkey_signature,
    verify_ping_signature,
    verify_container_signature,
    verify_container_logs_signature,
)

logger = logging.getLogger(__name__)
_KNOWN_CHUTES_ERRORS = (
    ChutesRelayDisabledError,
    ChutesBridgeConfigError,
    ChutesTransportError,
    ChutesMalformedResponseError,
    ChutesExecutorError,
)

apis_router = APIRouter()


def _get_version() -> str:
    """Read version from pyproject.toml."""
    try:
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
            return pyproject_data.get("project", {}).get("version", "unknown")
    except Exception as e:
        logger.error(f"Failed to read version from pyproject.toml: {e}")
        return "unknown"


def _validate_ssh_key_consistency(payload: UploadSShKeyPayload) -> None:
    """
    Validate that public_key matches data_to_sign to prevent SSH key substitution attacks.
    
    Security: Issue #744 - Without this check, an attacker could intercept a request
    and substitute a different public_key while keeping the valid signature for data_to_sign.
    
    Raises:
        HTTPException: 400 if keys don't match after normalization
    """
    pk_normalized = payload.public_key.strip()
    dts_normalized = payload.data_to_sign.strip()
    
    if pk_normalized != dts_normalized:
        logger.warning(
            "SSH key substitution attack detected or key mismatch: "
            f"public_key length={len(pk_normalized)}, data_to_sign length={len(dts_normalized)}"
        )
        raise HTTPException(status_code=400, detail="Public key mismatch")


def _validate_validator_signature(payload: UploadSShKeyPayload) -> None:
    """Require a valid validator signature over the SSH public key."""
    try:
        keypair = bittensor.Keypair(ss58_address=VALIDATOR_HOTKEY_SS58)
        if not keypair.verify(payload.public_key, payload.validator_signature):
            raise HTTPException(status_code=401, detail="Invalid validator signature")
        logger.info("Validator signature verification successful")
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Validator signature verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid validator signature")


def _translate_chutes_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (ChutesRelayDisabledError, ChutesTransportError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ChutesMalformedResponseError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, (ChutesExecutorError, ChutesBridgeConfigError)):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail="Unexpected Chutes relay error")


def _log_chutes_error(verb: str, exc: Exception) -> None:
    logger.error(
        _m(
            "Chutes relay request failed",
            extra={
                "verb": verb,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ),
    )


def _validate_chutes_signature(message: str, signature: str) -> None:
    try:
        keypair = bittensor.Keypair(ss58_address=VALIDATOR_HOTKEY_SS58)
        normalized = signature if signature.startswith("0x") else f"0x{signature}"
        if not keypair.verify(message, normalized):
            raise HTTPException(status_code=401, detail="Invalid validator signature")
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Chutes signature verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid validator signature")


@apis_router.post("/upload_ssh_key")
async def upload_ssh_key(
    payload: UploadSShKeyPayload, miner_service: Annotated[MinerService, Depends(MinerService)]
):
    _validate_ssh_key_consistency(payload)
    _validate_validator_signature(payload)
    return await miner_service.upload_ssh_key(payload)


@apis_router.post("/remove_ssh_key")
async def remove_ssh_key(
    payload: UploadSShKeyPayload, miner_service: Annotated[MinerService, Depends(MinerService)]
):
    _validate_ssh_key_consistency(payload)
    _validate_validator_signature(payload)
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


@apis_router.post("/chutes/install")
def chutes_install(
    payload: ChutesInstallPayload,
    relay_service: Annotated[ChutesRelayService, Depends(ChutesRelayService)],
):
    _validate_chutes_signature(payload.node_name, payload.validator_signature)
    try:
        return relay_service.install(
            validator_hotkey=payload.validator_hotkey,
            hotkey_ss58=payload.hotkey_ss58,
            hotkey_seed=payload.hotkey_seed,
            node_name=payload.node_name,
        )
    except _KNOWN_CHUTES_ERRORS as exc:
        _log_chutes_error("install", exc)
        raise _translate_chutes_error(exc)


@apis_router.post("/chutes/start")
def chutes_start(
    payload: ChutesCommandPayload,
    relay_service: Annotated[ChutesRelayService, Depends(ChutesRelayService)],
):
    _validate_chutes_signature("chutes_start", payload.validator_signature)
    try:
        return relay_service.start()
    except _KNOWN_CHUTES_ERRORS as exc:
        _log_chutes_error("start", exc)
        raise _translate_chutes_error(exc)


@apis_router.post("/chutes/stop")
def chutes_stop(
    payload: ChutesCommandPayload,
    relay_service: Annotated[ChutesRelayService, Depends(ChutesRelayService)],
):
    _validate_chutes_signature("chutes_stop", payload.validator_signature)
    try:
        return relay_service.stop()
    except _KNOWN_CHUTES_ERRORS as exc:
        _log_chutes_error("stop", exc)
        raise _translate_chutes_error(exc)


@apis_router.get("/chutes/status")
def chutes_status(
    relay_service: Annotated[ChutesRelayService, Depends(ChutesRelayService)],
):
    return relay_service.get_status_summary()


@apis_router.get("/version")
async def get_version():
    """
    Get the executor version information.

    Returns:
        dict: {"version": "x.y.z"}
    """
    return {"version": _get_version()}


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
