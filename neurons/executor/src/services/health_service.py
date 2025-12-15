"""Health check service for executor monitoring and diagnostics.

This module provides comprehensive health status information for the executor,
including system resources, Docker daemon status, GPU availability, and uptime.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import docker
import psutil

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# Track executor start time for uptime calculation
_EXECUTOR_START_TIME = time.time()


class HealthStatus(str, Enum):
    """Health status levels for the executor."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    status: HealthStatus
    message: str
    details: dict[str, Any] | None = None


def _check_docker_health() -> ComponentHealth:
    """Check Docker daemon health and connectivity.
    
    Returns:
        ComponentHealth: Docker daemon health status with container count
    """
    try:
        client = docker.from_env()
        # Ping the Docker daemon
        client.ping()
        
        # Get container counts
        containers = client.containers.list(all=True)
        running = sum(1 for c in containers if c.status == "running")
        total = len(containers)
        
        return ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="Docker daemon is responsive",
            details={
                "containers_running": running,
                "containers_total": total,
                "docker_version": client.version().get("Version", "unknown")
            }
        )
    except docker.errors.DockerException as e:
        logger.warning(f"Docker health check failed: {e}")
        return ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=f"Docker daemon error: {str(e)}",
            details=None
        )
    except Exception as e:
        logger.error(f"Unexpected error checking Docker health: {e}")
        return ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=f"Unexpected error: {str(e)}",
            details=None
        )


def _check_gpu_health() -> ComponentHealth:
    """Check GPU availability and health.
    
    Returns:
        ComponentHealth: GPU health status with device information
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        
        gpu_count = pynvml.nvmlDeviceGetCount()
        if gpu_count == 0:
            pynvml.nvmlShutdown()
            return ComponentHealth(
                status=HealthStatus.DEGRADED,
                message="No GPUs detected",
                details={"gpu_count": 0}
            )
        
        gpus = []
        unhealthy_gpus = 0
        
        for i in range(gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                
                # Get memory info
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError:
                    temp = None
                
                # Get driver version (only once)
                driver_version = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver_version, bytes):
                    driver_version = driver_version.decode("utf-8")
                
                gpu_info = {
                    "index": i,
                    "name": name,
                    "memory_total_mb": round(mem.total / (1024 * 1024)),
                    "memory_free_mb": round(mem.free / (1024 * 1024)),
                    "temperature_c": temp,
                    "driver_version": driver_version
                }
                gpus.append(gpu_info)
                
            except pynvml.NVMLError as e:
                logger.warning(f"Error getting info for GPU {i}: {e}")
                unhealthy_gpus += 1
                gpus.append({
                    "index": i,
                    "error": str(e)
                })
        
        pynvml.nvmlShutdown()
        
        if unhealthy_gpus > 0:
            return ComponentHealth(
                status=HealthStatus.DEGRADED,
                message=f"{unhealthy_gpus}/{gpu_count} GPUs have issues",
                details={"gpu_count": gpu_count, "gpus": gpus}
            )
        
        return ComponentHealth(
            status=HealthStatus.HEALTHY,
            message=f"{gpu_count} GPU(s) available",
            details={"gpu_count": gpu_count, "gpus": gpus}
        )
        
    except ImportError:
        return ComponentHealth(
            status=HealthStatus.DEGRADED,
            message="pynvml not installed",
            details=None
        )
    except Exception as e:
        # No NVIDIA driver or GPUs - this is acceptable for some executors
        logger.debug(f"GPU health check: {e}")
        return ComponentHealth(
            status=HealthStatus.DEGRADED,
            message="No NVIDIA GPUs available or driver not loaded",
            details={"error": str(e)}
        )


def _check_system_resources() -> ComponentHealth:
    """Check system resource availability (CPU, memory, disk).
    
    Returns:
        ComponentHealth: System resource health status
    """
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = round(memory.available / (1024 ** 3), 2)
        memory_total_gb = round(memory.total / (1024 ** 3), 2)
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free_gb = round(disk.free / (1024 ** 3), 2)
        disk_total_gb = round(disk.total / (1024 ** 3), 2)
        
        # Determine health status based on thresholds
        status = HealthStatus.HEALTHY
        warnings = []
        
        if memory_percent > 90:
            status = HealthStatus.DEGRADED
            warnings.append(f"High memory usage: {memory_percent}%")
        
        if disk_percent > 90:
            status = HealthStatus.DEGRADED
            warnings.append(f"High disk usage: {disk_percent}%")
        
        if cpu_percent > 95:
            status = HealthStatus.DEGRADED
            warnings.append(f"High CPU usage: {cpu_percent}%")
        
        message = "System resources OK" if not warnings else "; ".join(warnings)
        
        return ComponentHealth(
            status=status,
            message=message,
            details={
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "usage_percent": memory_percent,
                    "available_gb": memory_available_gb,
                    "total_gb": memory_total_gb
                },
                "disk": {
                    "usage_percent": disk_percent,
                    "free_gb": disk_free_gb,
                    "total_gb": disk_total_gb
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message=f"Failed to check system resources: {str(e)}",
            details=None
        )


def get_health_status() -> dict[str, Any]:
    """Get comprehensive health status for the executor.
    
    This function checks all major components and returns a consolidated
    health report suitable for monitoring and diagnostics.
    
    Returns:
        dict: Health status report with the following structure:
            {
                "status": str,           # Overall health status
                "timestamp": str,        # ISO format timestamp
                "uptime_seconds": int,   # Executor uptime
                "version": str,          # Executor version/project name
                "components": {
                    "docker": {...},     # Docker daemon health
                    "gpu": {...},        # GPU health
                    "system": {...}      # System resources health
                }
            }
    """
    # Check all components
    docker_health = _check_docker_health()
    gpu_health = _check_gpu_health()
    system_health = _check_system_resources()
    
    # Determine overall status (worst of all components, but GPU degraded is OK)
    component_statuses = [docker_health.status, system_health.status]
    
    # GPU being degraded (no GPU) shouldn't make the whole system unhealthy
    if gpu_health.status == HealthStatus.UNHEALTHY:
        component_statuses.append(HealthStatus.DEGRADED)
    
    if HealthStatus.UNHEALTHY in component_statuses:
        overall_status = HealthStatus.UNHEALTHY
    elif HealthStatus.DEGRADED in component_statuses:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.HEALTHY
    
    # Calculate uptime
    uptime_seconds = int(time.time() - _EXECUTOR_START_TIME)
    
    return {
        "status": overall_status.value,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": uptime_seconds,
        "version": settings.PROJECT_NAME,
        "components": {
            "docker": {
                "status": docker_health.status.value,
                "message": docker_health.message,
                "details": docker_health.details
            },
            "gpu": {
                "status": gpu_health.status.value,
                "message": gpu_health.message,
                "details": gpu_health.details
            },
            "system": {
                "status": system_health.status.value,
                "message": system_health.message,
                "details": system_health.details
            }
        }
    }


def get_liveness_status() -> dict[str, Any]:
    """Get simple liveness probe status.
    
    This is a lightweight check suitable for Kubernetes liveness probes
    or simple availability monitoring. It only checks if the service
    is running and can respond.
    
    Returns:
        dict: Simple liveness status
            {
                "alive": bool,
                "timestamp": str
            }
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def get_readiness_status() -> dict[str, Any]:
    """Get readiness probe status.
    
    This checks if the executor is ready to accept work. It verifies
    that critical components (Docker) are available.
    
    Returns:
        dict: Readiness status
            {
                "ready": bool,
                "timestamp": str,
                "checks": {
                    "docker": bool
                }
            }
    """
    docker_health = _check_docker_health()
    docker_ready = docker_health.status != HealthStatus.UNHEALTHY
    
    return {
        "ready": docker_ready,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checks": {
            "docker": docker_ready
        }
    }
