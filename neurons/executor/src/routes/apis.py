import asyncio
import logging
import os
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Annotated, Optional

import bittensor
import docker
from datura.requests.validator_requests import ssh_pubkey_signing_blob
from fastapi import APIRouter, Depends, Query, Header, HTTPException
from fastapi.responses import StreamingResponse
from services.miner_service import MinerService
from services.pod_log_service import PodLogService
from services.hardware_service import get_system_metrics, get_container_metrics
from core.config import VALIDATOR_HOTKEY_SS58, settings

from payloads.miner import UploadSShKeyPayload, GetPodLogsPaylod
from payloads.backend import ContainerUtilizationPayload
from dependencies.auth import verify_allowed_hotkey_signature, verify_ping_signature, verify_container_signature, verify_container_logs_signature

logger = logging.getLogger(__name__)

apis_router = APIRouter()

MAX_CONTAINER_LOG_TAIL_LINES = int(os.getenv("EXECUTOR_CONTAINER_LOGS_MAX_TAIL", "500"))
MAX_FOLLOW_LOG_STREAMS = int(os.getenv("EXECUTOR_CONTAINER_LOGS_MAX_FOLLOW_STREAMS", "10"))
FOLLOW_LOG_STREAM_MAX_SECONDS = float(os.getenv("EXECUTOR_CONTAINER_LOGS_FOLLOW_MAX_SECONDS", "300"))
FOLLOW_LOG_STREAM_QUEUE_MAX_SIZE = max(1, int(os.getenv("EXECUTOR_CONTAINER_LOGS_QUEUE_MAX_SIZE", "100")))
FOLLOW_LOG_STREAM_QUEUE_PUT_TIMEOUT_SECONDS = 0.1

_CONTAINER_LOG_STREAM_END = object()
_container_log_stream_executor = ThreadPoolExecutor(
    max_workers=max(1, MAX_FOLLOW_LOG_STREAMS),
    thread_name_prefix="container-log-stream",
)
_active_follow_log_streams = 0
_active_follow_log_streams_lock = asyncio.Lock()


def _normalize_log_tail(tail: Optional[int]) -> int:
    if tail is None:
        return MAX_CONTAINER_LOG_TAIL_LINES
    return max(1, min(tail, MAX_CONTAINER_LOG_TAIL_LINES))


def _queue_log_stream_item(loop, queue: asyncio.Queue, item, stop_event: threading.Event) -> bool:
    if stop_event.is_set():
        return False

    future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
    while True:
        try:
            future.result(timeout=FOLLOW_LOG_STREAM_QUEUE_PUT_TIMEOUT_SECONDS)
            return True
        except FutureTimeoutError:
            if stop_event.is_set():
                future.cancel()
                return False


def _release_follow_log_stream_on_loop(loop) -> None:
    loop.call_soon_threadsafe(lambda: asyncio.create_task(_release_follow_log_stream()))


def _produce_follow_container_logs(
    *,
    container,
    tail: int,
    since: Optional[int],
    stdout: bool,
    stderr: bool,
    loop,
    queue: asyncio.Queue,
    stop_event: threading.Event,
    stream_ref: dict,
) -> None:
    log_iterator = None
    try:
        log_iterator = iter(
            container.logs(
                stream=True,
                follow=True,
                tail=tail,
                since=since,
                stdout=stdout,
                stderr=stderr,
            )
        )
        stream_ref["iterator"] = log_iterator

        for log in log_iterator:
            if stop_event.is_set():
                break
            if not _queue_log_stream_item(loop, queue, log, stop_event):
                break
    except Exception as exc:
        _queue_log_stream_item(loop, queue, exc, stop_event)
    finally:
        close = getattr(log_iterator, "close", None)
        if close:
            try:
                close()
            except Exception:
                logger.debug("Failed to close container logs iterator", exc_info=True)
        _queue_log_stream_item(loop, queue, _CONTAINER_LOG_STREAM_END, stop_event)
        _release_follow_log_stream_on_loop(loop)


async def _reserve_follow_log_stream() -> bool:
    global _active_follow_log_streams
    async with _active_follow_log_streams_lock:
        if _active_follow_log_streams >= MAX_FOLLOW_LOG_STREAMS:
            return False
        _active_follow_log_streams += 1
        return True


async def _release_follow_log_stream() -> None:
    global _active_follow_log_streams
    async with _active_follow_log_streams_lock:
        _active_follow_log_streams = max(0, _active_follow_log_streams - 1)


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


def _validate_validator_signature(payload: UploadSShKeyPayload, require_nonce: bool = False) -> None:
    """Require a valid validator signature over the SSH public key.

    When the request carries an attestation nonce the signature must cover
    public_key AND nonce (datura.ssh_pubkey_signing_blob) — stripping or swapping
    the nonce invalidates the signature. Legacy requests without a nonce keep
    the bare-public-key format. `require_nonce` is the G3 enforcement phase:
    attestation-relevant requests without a nonce are rejected outright.
    """
    if require_nonce and not payload.nonce:
        logger.warning("Rejecting SSH-key upload without an attestation nonce (enforcement on)")
        raise HTTPException(status_code=401, detail="Attestation nonce required")
    try:
        keypair = bittensor.Keypair(ss58_address=VALIDATOR_HOTKEY_SS58)
        signed_blob = ssh_pubkey_signing_blob(payload.public_key, payload.nonce)
        if not keypair.verify(signed_blob, payload.validator_signature):
            raise HTTPException(status_code=401, detail="Invalid validator signature")
        logger.info("Validator signature verification successful")
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Validator signature verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid validator signature")


@apis_router.post("/upload_ssh_key")
async def upload_ssh_key(
    payload: UploadSShKeyPayload, miner_service: Annotated[MinerService, Depends(MinerService)]
):
    logger.info("upload_ssh_key route entered")
    _validate_ssh_key_consistency(payload)
    logger.info("upload_ssh_key SSH key consistency validated")
    # The nonce requirement applies only to uploads (the attestation-bearing
    # request); removals never carry a nonce.
    _validate_validator_signature(payload, require_nonce=settings.REQUIRE_ATTESTATION_NONCE)
    logger.info("upload_ssh_key validator signature validated")
    logger.info("upload_ssh_key service call started")
    response = await miner_service.upload_ssh_key(payload)
    logger.info("upload_ssh_key service call completed")
    return response


@apis_router.post("/remove_ssh_key")
async def remove_ssh_key(
    payload: UploadSShKeyPayload, miner_service: Annotated[MinerService, Depends(MinerService)]
):
    logger.info("remove_ssh_key route entered")
    _validate_ssh_key_consistency(payload)
    logger.info("remove_ssh_key SSH key consistency validated")
    _validate_validator_signature(payload)
    logger.info("remove_ssh_key validator signature validated")
    logger.info("remove_ssh_key service call started")
    response = await miner_service.remove_ssh_key(payload)
    logger.info("remove_ssh_key service call completed")
    return response


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

    reserved_follow_stream = False
    if follow:
        reserved_follow_stream = await _reserve_follow_log_stream()
        if not reserved_follow_stream:
            raise HTTPException(status_code=429, detail="Too many active live log streams")

    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
    except Exception:
        if reserved_follow_stream:
            await _release_follow_log_stream()
        raise

    if not follow:
        def generate():
            for log in container.logs(
                stream=True,
                follow=False,
                tail=tail if tail else "all",
                since=since,
                stdout=stdout,
                stderr=stderr,
            ):
                yield log

        return StreamingResponse(generate(), media_type="text/plain")

    tail = _normalize_log_tail(tail)

    async def generate():
        loop = asyncio.get_running_loop()
        started = time.monotonic()
        queue = asyncio.Queue(maxsize=FOLLOW_LOG_STREAM_QUEUE_MAX_SIZE)
        stop_event = threading.Event()
        stream_ref = {}

        try:
            _container_log_stream_executor.submit(
                _produce_follow_container_logs,
                container=container,
                tail=tail,
                since=since,
                stdout=stdout,
                stderr=stderr,
                loop=loop,
                queue=queue,
                stop_event=stop_event,
                stream_ref=stream_ref,
            )
        except Exception:
            await _release_follow_log_stream()
            raise

        try:
            while True:
                remaining_seconds = FOLLOW_LOG_STREAM_MAX_SECONDS - (time.monotonic() - started)
                if remaining_seconds <= 0:
                    logger.info(
                        "Stopping live container log stream after %.1f seconds for %s",
                        FOLLOW_LOG_STREAM_MAX_SECONDS,
                        container_name,
                    )
                    break

                try:
                    log = await asyncio.wait_for(queue.get(), timeout=remaining_seconds)
                except asyncio.TimeoutError:
                    logger.info(
                        "Stopping live container log stream after %.1f seconds for %s",
                        FOLLOW_LOG_STREAM_MAX_SECONDS,
                        container_name,
                    )
                    break

                if log is _CONTAINER_LOG_STREAM_END:
                    break
                if isinstance(log, Exception):
                    raise log
                yield log
        finally:
            stop_event.set()
            close = getattr(stream_ref.get("iterator"), "close", None)
            if close:
                try:
                    close()
                except Exception:
                    logger.debug("Failed to close container logs iterator", exc_info=True)

    return StreamingResponse(generate(), media_type="text/plain")
