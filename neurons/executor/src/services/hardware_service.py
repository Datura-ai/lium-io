import psutil
import pynvml
import docker
from core.logger import get_logger

logger = get_logger(__name__)


def get_system_metrics():
    """
    Collect system hardware utilization metrics including CPU, memory, storage, and GPU.
    
    Returns:
        dict: Hardware utilization metrics with the following structure:
            {
                "cpu": float,       # CPU utilization percentage
                "memory": float,    # Memory utilization percentage  
                "storage": float,   # Storage utilization percentage
                "gpu": [            # Array of GPU metrics
                    {
                        "utilization": float,  # GPU utilization percentage
                        "memory": float        # GPU memory utilization percentage
                    }
                ]
            }
    """
    # CPU and memory
    cpu = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory().percent
    storage = psutil.disk_usage('/').percent

    # GPU
    gpus = []
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                "utilization": util.gpu,                  # %
                "memory": mem.used / mem.total * 100.0    # %
            })
        pynvml.nvmlShutdown()
    except (pynvml.NVMLError, pynvml.NVMLError_NotSupported, pynvml.NVMLError_DriverNotLoaded) as e:
        # This is expected on systems without NVIDIA GPUs or drivers
        logger.debug(f"No GPU available: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error collecting GPU metrics: {e}")

    return {
        "cpu": cpu,
        "memory": memory,
        "storage": storage,
        "gpu": gpus
    }


def get_container_metrics(container_name: str, gpu_uuids: list[str]):
    """
    Collect hardware utilization metrics for a specific container.

    Args:
        container_name: Name of the Docker container
        gpu_uuids: List of GPU UUIDs assigned to this container

    Returns:
        dict: Container-specific hardware utilization metrics with the following structure:
            {
                "cpu": {
                    "usage": float,           # CPU usage percentage
                    "limit": float            # CPU limit (number of cores)
                },
                "memory": {
                    "usage": int,             # Memory usage in bytes
                    "limit": int,             # Memory limit in bytes
                    "usage_percent": float    # Memory usage percentage
                },
                "storage": float,             # Storage utilization percentage (system-wide)
                "gpu": [                      # Array of GPU metrics for assigned GPUs only
                    {
                        "uuid": str,
                        "utilization": float,  # GPU utilization percentage
                        "memory": float        # GPU memory utilization percentage
                    }
                ]
            }
    """
    try:
        # Get Docker client
        client = docker.from_env()
        container = client.containers.get(container_name)

        # Get container stats (non-streaming, single sample)
        stats = container.stats(stream=False)

        # Calculate CPU usage percentage
        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
        cpu_count = stats["cpu_stats"]["online_cpus"]

        cpu_usage_percent = 0.0
        if system_delta > 0 and cpu_delta > 0:
            cpu_usage_percent = (cpu_delta / system_delta) * cpu_count * 100.0

        # Get CPU limit (from container spec)
        cpu_limit = cpu_count  # Default to all CPUs
        if "NanoCpus" in container.attrs["HostConfig"] and container.attrs["HostConfig"]["NanoCpus"]:
            cpu_limit = container.attrs["HostConfig"]["NanoCpus"] / 1e9  # Convert from nano CPUs to number of cores

        # Calculate memory usage
        memory_usage = stats["memory_stats"].get("usage", 0)
        memory_limit = stats["memory_stats"].get("limit", 0)
        memory_usage_percent = 0.0
        if memory_limit > 0:
            memory_usage_percent = (memory_usage / memory_limit) * 100.0

        # Get storage usage (system-wide, same as get_system_metrics)
        storage = psutil.disk_usage('/').percent

        # Get GPU metrics for specified UUIDs only
        gpus = []
        if gpu_uuids:
            try:
                pynvml.nvmlInit()

                for gpu_uuid in gpu_uuids:
                    try:
                        # Get GPU handle by UUID
                        handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)

                        # Get utilization rates
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                        # Get memory info
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                        gpus.append({
                            "uuid": gpu_uuid,
                            "utilization": util.gpu,                  # %
                            "memory": mem.used / mem.total * 100.0    # %
                        })
                    except pynvml.NVMLError as e:
                        logger.warning(f"Error getting metrics for GPU {gpu_uuid}: {e}")
                        # Add entry with zero values if GPU not accessible
                        gpus.append({
                            "uuid": gpu_uuid,
                            "utilization": 0.0,
                            "memory": 0.0
                        })

                pynvml.nvmlShutdown()
            except (pynvml.NVMLError, pynvml.NVMLError_NotSupported, pynvml.NVMLError_DriverNotLoaded) as e:
                logger.debug(f"No GPU available: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error collecting GPU metrics: {e}")

        return {
            "cpu": {
                "usage": round(cpu_usage_percent, 2),
                "limit": cpu_limit
            },
            "memory": {
                "usage": memory_usage,
                "limit": memory_limit,
                "usage_percent": round(memory_usage_percent, 2)
            },
            "storage": storage,
            "gpu": gpus
        }

    except docker.errors.NotFound:
        logger.error(f"Container '{container_name}' not found")
        raise ValueError(f"Container '{container_name}' not found")
    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting container metrics: {e}")
        raise