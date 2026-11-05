"""Shared utilities for preflight checks."""

import os
import sys
import time
import logging
from contextlib import contextmanager
from typing import Optional

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


UTILIZATION_SAMPLES = 5
UTILIZATION_SAMPLE_INTERVAL_S = 1.0


def _sample_utilization(pynvml, gpu_details: list, samples: int = UTILIZATION_SAMPLES,
                        interval_s: float = UTILIZATION_SAMPLE_INTERVAL_S) -> None:
    """Set each GPU's utilization/memory_utilization to the lowest of several NVML reads.

    The first read right after nvmlInit in a fresh container can report a spike on an idle
    GPU (driver/context init when persistence mode is off), so a single sample refused idle
    hosts (ticket-0996, RTX 3090). A GPU that is really busy stays high on every sample.
    A failed read counts as 0, as before.
    """
    per_gpu = [([], []) for _ in gpu_details]
    for n in range(samples):
        if n:
            time.sleep(interval_s)
        for detail, (gpu_vals, mem_vals) in zip(gpu_details, per_gpu):
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(detail["index"]))
                gpu_vals.append(rates.gpu)
                mem_vals.append(rates.memory)
            except Exception:
                gpu_vals.append(0)
                mem_vals.append(0)
    for detail, (gpu_vals, mem_vals) in zip(gpu_details, per_gpu):
        detail["utilization"] = min(gpu_vals)
        detail["memory_utilization"] = min(mem_vals)
        logger.debug(f"GPU {detail['index']} utilization samples: gpu={gpu_vals} memory={mem_vals}")


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
            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_mb = memory_info.total // (1024 * 1024)  # MB
            except:
                gpu_memory_mb = 0

        # Get UUIDs from all GPUs
        uuids = []
        gpu_details = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            uuids.append(uuid)

            if include_utilization:
                # Get memory info
                try:
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_total = memory_info.total // (1024 * 1024)  # MB
                except:
                    memory_total = 0

                # A card driving a monitor shows display memory traffic while idle (ticket-0996);
                # unreadable values keep the strict check.
                try:
                    display_active = pynvml.nvmlDeviceGetDisplayActive(handle) == 1
                except Exception:
                    display_active = False
                try:
                    compute_processes = len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
                except Exception:
                    compute_processes = None

                gpu_details.append({
                    "index": i,
                    "name": pynvml.nvmlDeviceGetName(handle),
                    "uuid": uuid,
                    "utilization": 0,
                    "memory_utilization": 0,
                    "memory_total_mb": memory_total,
                    "display_active": display_active,
                    "compute_processes": compute_processes,
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
