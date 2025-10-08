import psutil
import pynvml
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