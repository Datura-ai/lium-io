import psutil
import pynvml
import docker
from core.logger import get_logger

logger = get_logger(__name__)


def _parse_df_output(df_output: str) -> dict:
    """
    Parse `df -k` output to extract filesystem metrics.
    
    Args:
        df_output: Output from `df -k` command
    
    Returns:
        dict: {
            "total": int,  # Total size in KB
            "used": int,   # Used size in KB
            "available": int  # Available size in KB,
            "utilization": float  # Utilization percentage
        }
    """
    lines = df_output.strip().split('\n')
    if len(lines) < 2:
        return {"total": 0, "used": 0, "available": 0, "utilization": 0.0}
    
    # Skip header line, get first data line
    data_line = lines[1].split()
    if len(data_line) < 4:
        return {"total": 0, "used": 0, "available": 0, "utilization": 0.0}
    
    try:
        # df -k output format: Filesystem 1K-blocks Used Available Use% Mounted on
        total = int(data_line[1])
        used = int(data_line[2])
        available = int(data_line[3])
        utilization = 0.0
        if total > 0:
            utilization = round((used / total) * 100.0, 2)
        return {"total": total, "used": used, "available": available, "utilization": utilization}
    except (ValueError, IndexError) as e:
        logger.warning(f"Error parsing df output: {e}, output: {df_output}")
        return {"total": 0, "used": 0, "available": 0, "utilization": 0.0}


def _get_filesystem_usage(container, mount_point: str) -> dict:
    """
    Get filesystem usage for a specific mount point inside the container.
    
    Args:
        container: Docker container object
        mount_point: Mount point path (e.g., "/", "/root")
    
    Returns:
        dict: {
            "total": int,  # Total size in KB
            "used": int,   # Used size in KB
            "available": int,  # Available size in KB
            "utilization": float  # Utilization percentage
            "mount_point": str    # Mount point path
        }
    """
    try:
        # Execute df -k inside the container to get metrics in KB
        exec_result = container.exec_run(f"df -k {mount_point}", user="root")
        if exec_result.exit_code != 0:
            logger.warning(f"Failed to get filesystem usage for {mount_point}: {exec_result.output.decode()}")
            return {"total": 0, "used": 0, "available": 0, "utilization": 0.0, "mount_point": mount_point}
        
        output = exec_result.output.decode('utf-8')
        result = _parse_df_output(output)
        result["mount_point"] = mount_point
        return result
    except Exception as e:
        logger.warning(f"Error getting filesystem usage for {mount_point}: {e}")
        return {"total": 0, "used": 0, "available": 0, "utilization": 0.0, "mount_point": mount_point}


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
                    "utilization": float,     # CPU usage percentage
                    "limit": float            # CPU limit (number of cores)
                },
                "memory": {
                    "used": int,              # Memory usage in bytes
                    "limit": int,             # Memory limit in bytes
                    "utilization": float      # Memory usage percentage
                },
                "storage": {
                    "total": int,             # Total storage in KB
                    "used": int,              # Used storage in KB
                    "available": int,         # Available storage in KB
                    "utilization": float,     # Storage usage percentage
                    "mount_point": str        # Mount point path
                },
                "volume": {                   # Volume metrics (None if no vloopback volume)
                    "total": int,             # Total volume size in KB
                    "used": int,              # Used volume size in KB
                    "available": int,         # Available volume size in KB
                    "utilization": float,     # Volume usage percentage
                    "mount_point": str        # Mount point path
                } | None,
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
        memory_utilization = 0.0
        if memory_limit > 0:
            memory_utilization = round((memory_usage / memory_limit) * 100.0, 2)

        # Get actual filesystem usage from inside the container
        storage_metrics = _get_filesystem_usage(container, "/")
        
        # Get volume metrics if there's a vloopback volume
        volume_metrics = None
        mounts = container.attrs.get("Mounts", [])
        for mount in mounts:
            # Check if it's a vloopback volume
            if mount.get("Driver", "").startswith("vloopback"):
                mount_point = mount.get("Destination", "")
                if mount_point:
                    volume_metrics = _get_filesystem_usage(container, mount_point)
                    break

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
                "utilization": round(cpu_usage_percent, 2),
                "limit": cpu_limit
            },
            "memory": {
                "used": memory_usage,
                "limit": memory_limit,
                "utilization": memory_utilization
            },
            "storage": storage_metrics,
            "volume": volume_metrics,
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