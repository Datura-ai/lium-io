import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from core.utils import _m, get_extra_info
from services.rental_docker_sdk import (
    ContainerExecResult,
    ContainerExecSpec,
    ContainerRunSpec,
    RentalDockerSdkClient,
)

logger = logging.getLogger(__name__)

_DOCKER_SDK_LOG_OUTPUT_LIMIT = 2000
_T = TypeVar("_T")


def _truncate_log_text(value: object, limit: int = _DOCKER_SDK_LOG_OUTPUT_LIMIT) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _stdin_size(value: str | bytes | None) -> int:
    if value is None:
        return 0
    if isinstance(value, bytes):
        return len(value)
    return len(value.encode())


def _exec_result_log_fields(result: ContainerExecResult) -> dict:
    return {
        "exit_status": result.exit_status,
        "stdout_len": len(result.stdout),
        "stderr_len": len(result.stderr),
        "stdout": _truncate_log_text(result.stdout),
        "stderr": _truncate_log_text(result.stderr),
    }


def log_rental_docker_sdk_operation(
    *,
    operation: str,
    status: str,
    log_extra: dict,
    level: int = logging.INFO,
    **fields,
) -> None:
    clean_fields = {
        key: value
        for key, value in fields.items()
        if value is not None
    }
    logger.log(
        level,
        _m(
            f"Rental Docker SDK operation {status}",
            extra=get_extra_info(
                {
                    **log_extra,
                    "docker_access": "sdk",
                    "docker_operation": operation,
                    "operation_status": status,
                    "host_shell_command": False,
                    **clean_fields,
                }
            ),
        ),
    )


async def run_logged_rental_docker_sdk_operation(
    *,
    operation: str,
    log_extra: dict,
    call: Callable[[], Awaitable[_T]],
    **fields,
) -> _T:
    start = time.monotonic()
    log_rental_docker_sdk_operation(
        operation=operation,
        status="started",
        log_extra=log_extra,
        **fields,
    )
    try:
        result = await call()
    except Exception as exc:
        log_rental_docker_sdk_operation(
            operation=operation,
            status="failed",
            log_extra=log_extra,
            level=logging.ERROR,
            duration_ms=int((time.monotonic() - start) * 1000),
            error=str(exc),
            **fields,
        )
        raise

    log_rental_docker_sdk_operation(
        operation=operation,
        status="succeeded",
        log_extra=log_extra,
        duration_ms=int((time.monotonic() - start) * 1000),
        **fields,
    )
    return result


async def exec_logged_rental_docker_sdk_operation(
    *,
    docker_client: RentalDockerSdkClient,
    operation: str,
    exec_spec: ContainerExecSpec,
    log_extra: dict,
) -> ContainerExecResult:
    start = time.monotonic()
    fields = rental_exec_spec_log_fields(exec_spec)
    log_rental_docker_sdk_operation(
        operation=operation,
        status="started",
        log_extra=log_extra,
        **fields,
    )
    try:
        result = await docker_client.exec_in_container(exec_spec)
    except Exception as exc:
        log_rental_docker_sdk_operation(
            operation=operation,
            status="failed",
            log_extra=log_extra,
            level=logging.ERROR,
            duration_ms=int((time.monotonic() - start) * 1000),
            error=str(exc),
            **fields,
        )
        raise

    result_fields = {
        **fields,
        "duration_ms": int((time.monotonic() - start) * 1000),
        **_exec_result_log_fields(result),
    }
    if result.exit_status == 0:
        log_rental_docker_sdk_operation(
            operation=operation,
            status="succeeded",
            log_extra=log_extra,
            **result_fields,
        )
    else:
        log_rental_docker_sdk_operation(
            operation=operation,
            status="failed",
            log_extra=log_extra,
            level=logging.WARNING,
            **result_fields,
        )

    return result


def rental_run_spec_log_fields(run_spec: ContainerRunSpec) -> dict:
    return {
        "container_name": run_spec.name,
        "image": run_spec.image,
        "command_argv": list(run_spec.command),
        "environment_keys": sorted(run_spec.environment),
        "ports": [
            {
                "container_port": port.container_port,
                "host_port": port.host_port,
                "protocol": port.protocol,
            }
            for port in run_spec.ports
        ],
        "volumes": [
            {
                "source": volume.source,
                "target": volume.target,
                "read_only": volume.read_only,
            }
            for volume in run_spec.volumes
        ],
        "restart_policy": run_spec.restart_policy,
        "runtime": run_spec.runtime,
        "cap_add": list(run_spec.cap_add),
        "sysctls": run_spec.sysctls,
        "device_mounts": [
            {
                "path_on_host": device.path_on_host,
                "path_in_container": device.path_in_container,
                "permissions": device.permissions,
            }
            for device in run_spec.devices
        ],
        "gpu_device_requests": [
            {
                "count": device_request.count,
                "device_ids": list(device_request.device_ids),
                "capabilities": [
                    list(capability)
                    for capability in device_request.capabilities
                ],
            }
            for device_request in run_spec.device_requests
        ],
        "cpu_count": run_spec.cpu_count,
        "memory_gb": run_spec.memory_gb,
        "storage_limit_gb": run_spec.storage_limit_gb,
        "shm_size": run_spec.shm_size,
        "entrypoint": run_spec.entrypoint,
    }


def rental_exec_spec_log_fields(exec_spec: ContainerExecSpec) -> dict:
    return {
        "container_name": exec_spec.container_name,
        "exec_argv": [_truncate_log_text(part) for part in exec_spec.argv],
        "stdin_bytes": _stdin_size(exec_spec.stdin),
        "environment_keys": sorted(exec_spec.environment),
    }
