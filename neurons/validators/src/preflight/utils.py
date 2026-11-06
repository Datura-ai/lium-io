"""Shared utilities for preflight checks."""

import os
import sys
import time
import logging
from contextlib import contextmanager
from typing import Optional

import psutil

from services.gpu_spec_table import UNIFIED_MEMORY_GPU_MODELS, normalize_gpu_model

logger = logging.getLogger(__name__)


@contextmanager
def suppress_library_output():
    """Suppress stdout/stderr from C library calls (unless in debug mode)."""
    # Check if we're in debug mode
    if logger.isEnabledFor(logging.DEBUG):
        # In debug mode, don't suppress anything
        yield
        return

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # Save original stdout/stderr
    stdout_dup = os.dup(stdout_fd)
    stderr_dup = os.dup(stderr_fd)

    # Redirect to /dev/null
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)

    try:
        yield
    finally:
        # Restore original stdout/stderr
        os.dup2(stdout_dup, stdout_fd)
        os.dup2(stderr_dup, stderr_fd)
        os.close(stdout_dup)
        os.close(stderr_dup)


def _memory_total_mb(pynvml, handle, name) -> int:
    """A card's memory in MB, 0 when NVML cannot read it; a unified-memory card (GB10) uses the host RAM."""
    try:
        return pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 * 1024)
    except Exception:
        if normalize_gpu_model(name) in UNIFIED_MEMORY_GPU_MODELS:
            return psutil.virtual_memory().total // (1024 * 1024)
        return 0


UTILIZATION_SAMPLES = 5
UTILIZATION_SAMPLE_INTERVAL_S = 1.0


def _sample_utilization(pynvml, gpu_details: list, samples: int = UTILIZATION_SAMPLES,
                        interval_s: float = UTILIZATION_SAMPLE_INTERVAL_S) -> None:
    """Store several paired (gpu, memory) NVML utilization reads per GPU in utilization_samples.

    The first read right after nvmlInit in a fresh container can report a spike on an idle
    GPU (driver/context init when persistence mode is off), so a single sample refused idle
    hosts (ticket-0996, RTX 3090). Reads stay paired: the check needs one whole idle sample.
    A failed utilization read counts as 0, as before; a failed process-list read
    keeps the strict memory check for that sample.
    """
    for detail in gpu_details:
        detail["utilization_samples"] = []
        detail["compute_process_samples"] = []
    for n in range(samples):
        if n:
            time.sleep(interval_s)
        for detail in gpu_details:
            handle = pynvml.nvmlDeviceGetHandleByIndex(detail["index"])
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                detail["utilization_samples"].append((rates.gpu, rates.memory))
            except Exception:
                detail["utilization_samples"].append((0, 0))
            # Read after the utilization read: the process list changes over time, so a compute
            # process starting mid-window must void the display exemption for that sample.
            try:
                detail["compute_process_samples"].append(len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle)))
            except Exception:
                detail["compute_process_samples"].append(None)
    for detail in gpu_details:
        logger.debug(f"GPU {detail['index']} utilization samples (gpu, memory): {detail['utilization_samples']}")


def get_gpu_info(include_utilization: bool = False, include_memory: bool = False) -> Optional[dict]:
    """
    Get GPU information from the local system.

    Args:
        include_utilization: If True, include utilization and memory details for each GPU
        include_memory: If True, include GPU memory info from first GPU

    Returns:
        Dict with gpu_count, gpu_model, gpu_uuids, and optionally gpu_details, gpu_memory_mb
        or None if no GPUs detected
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count == 0:
            pynvml.nvmlShutdown()
            return None

        # Get info from first GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)

        # Get memory from first GPU if requested
        gpu_memory_mb = None
        if include_memory:
            gpu_memory_mb = _memory_total_mb(pynvml, handle, gpu_name)

        # Get UUIDs from all GPUs
        uuids = []
        gpu_details = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            uuids.append(uuid)

            if include_utilization:
                name = pynvml.nvmlDeviceGetName(handle)
                memory_total = _memory_total_mb(pynvml, handle, name)

                # A card driving a monitor shows display memory traffic while idle (ticket-0996);
                # unreadable values keep the strict check.
                try:
                    display_active = pynvml.nvmlDeviceGetDisplayActive(handle) == 1
                except Exception:
                    display_active = False

                gpu_details.append({
                    "index": i,
                    "name": name,
                    "uuid": uuid,
                    "utilization": 0,
                    "memory_utilization": 0,
                    "memory_total_mb": memory_total,
                    "display_active": display_active,
                })

        if include_utilization:
            _sample_utilization(pynvml, gpu_details)

        pynvml.nvmlShutdown()

        uuids_str = ",".join(uuids)
        result = {
            "gpu_count": device_count,
            "gpu_model": gpu_name,
            "gpu_uuids": uuids_str,
            "uuids": uuids_str,  # Alias for compatibility with different libraries
        }

        if include_utilization:
            result["gpu_details"] = gpu_details

        if include_memory and gpu_memory_mb is not None:
            result["gpu_memory_mb"] = gpu_memory_mb

        return result

    except Exception as e:
        logger.debug(f"Failed to get GPU info: {e}")
        return None
