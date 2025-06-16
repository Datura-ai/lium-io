import asyncio
import logging
import time
import subprocess
import aiohttp
import click
import pynvml
import psutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GPUMetricsTracker:
    def __init__(self, threshold_percent: float = 10.0):
        self.previous_metrics: dict[int, dict] = {}
        self.threshold = threshold_percent

    def has_significant_change(self, gpu_id: int, util: float, mem_used: float) -> bool:
        if gpu_id not in self.previous_metrics:
            self.previous_metrics[gpu_id] = {"util": util, "mem_used": mem_used}
            return True

        prev = self.previous_metrics[gpu_id]
        util_diff = abs(util - prev["util"])
        mem_diff_percent = abs(mem_used - prev["mem_used"]) / prev["mem_used"] * 100

        if util_diff >= self.threshold or mem_diff_percent >= self.threshold:
            self.previous_metrics[gpu_id] = {"util": util, "mem_used": mem_used}
            return True
        return False


async def scrape_gpu_metrics(
    interval: int,
    program_id: str,
    signature: str,
    executor_id: str,
    validator_hotkey: str,
    compute_rest_app_url: str,
):
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            logger.warning("No NVIDIA GPUs found in the system")
            return
    except pynvml.NVMLError as e:
        logger.error(f"Failed to initialize NVIDIA Management Library: {e}")
        logger.error(
            "This might be because no NVIDIA GPU is present or drivers are not properly installed"
        )
        return

    http_url = f"{compute_rest_app_url}/validator/{validator_hotkey}/update-gpu-metrics"
    logger.info(f"Will send metrics to: {http_url}")

    # Initialize the tracker
    tracker = GPUMetricsTracker(threshold_percent=10.0)

    async with aiohttp.ClientSession() as session:
        logger.info(f"Scraping metrics for {device_count} GPUs...")
        try:
            while True:
                try:
                    gpu_utilization = []
                    should_send = False

                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                        name = pynvml.nvmlDeviceGetName(handle)
                        if isinstance(name, bytes):
                            name = name.decode("utf-8")

                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

                        gpu_util = utilization.gpu
                        mem_used = memory.used
                        mem_total = memory.total
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                        # Check if there's a significant change for this GPU
                        if tracker.has_significant_change(i, gpu_util, mem_used):
                            should_send = True
                            logger.info(f"Significant change detected for GPU {i}")

                        gpu_utilization.append(
                            {
                                "utilization_in_percent": gpu_util,
                                "memory_utilization_in_bytes": mem_used,
                                "memory_utilization_in_percent": round(mem_used / mem_total * 100, 1)
                            }
                        )

                    # Get CPU, RAM, and Disk metrics using psutil
                    cpu_percent = psutil.cpu_percent(interval=1)
                    ram = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    cpu_ram_utilization = {
                        "cpu_utilization_in_percent": cpu_percent,
                        "ram_utilization_in_bytes": ram.used,
                        "ram_utilization_in_percent": ram.percent
                    }

                    disk_utilization = {
                        "disk_utilization_in_bytes": disk.used,
                        "disk_utilization_in_percent": disk.percent
                    }
                    
                    # Only send if there's a significant change in any GPU
                    if should_send:
                        payload = {
                            "gpu_utilization": gpu_utilization,
                            "cpu_ram_utilization": cpu_ram_utilization,
                            "disk_utilization": disk_utilization,
                            "timestamp": timestamp,
                            "program_id": program_id,
                            "signature": signature,
                            "executor_id": executor_id,
                        }
                        # Send HTTP POST request
                        async with session.post(http_url, json=payload) as response:
                            if response.status == 200:
                                logger.info("Successfully sent metrics to backend")
                            else:
                                logger.error(f"Failed to send metrics. Status: {response.status}")
                                text = await response.text()
                                logger.error(f"Response: {text}")

                    await asyncio.sleep(interval)

                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)  # Wait before retrying

        except KeyboardInterrupt:
            logger.info("Stopping GPU scraping...")
        finally:
            pynvml.nvmlShutdown()


async def manage_docker_images(
    compute_rest_app_url: str,
    validator_hotkey: str,
    executor_id: str,
    program_id: str,
    signature: str,
    min_disk_space_multiplier: float = 3.0,  # Minimum disk space required (3x image size)
):
    """
    Manages Docker images on the executor by:
    1. Syncing with backend for most used templates
    2. Cleaning up unused images
    3. Ensuring sufficient disk space

    Args:
        compute_rest_app_url (str): URL of the compute REST application
        validator_hotkey (str): Validator's hotkey
        executor_id (str): Executor ID
        program_id (str): Program ID
        signature (str): Signature for verification
        min_disk_space_multiplier (float): Minimum disk space required as multiple of image size
    """
    # Get GPU name using NVML
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get first GPU
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        pynvml.nvmlShutdown()
    except Exception as e:
        logger.error(f"Failed to get GPU name: {e}")
        gpu_name = "unknown"

    backend_url = f"{compute_rest_app_url}/validator/{validator_hotkey}/cache-default-docker-image"

    async with aiohttp.ClientSession() as session:
        try:
            # Get disk space info
            disk = psutil.disk_usage('/')
            available_space = disk.free

            # Get templates from backend with GPU name and signature
            payload = {
                "gpu_name": gpu_name,
                "program_id": program_id,
                "signature": signature,
                "executor_id": executor_id
            }
            
            async with session.post(backend_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check if no templates were found for the GPU
                    if "message" in data and "No templates found for GPU" in data["message"]:
                        logger.warning(f"No templates found for GPU {gpu_name}")
                        # Skip Docker pull process and continue to cleanup
                    else:
                        # Clean up unused images
                        try:
                            # Get all local images
                            result = subprocess.run(
                                ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'],
                                capture_output=True,
                                text=True
                            )
                            if result.returncode == 0:
                                local_images = result.stdout.strip().split('\n')
                                
                                # Get repository names from current templates
                                current_repository = data['docker_image']
                                
                                for image in local_images:
                                    if image and image.split(':')[0] == current_repository and image.split(':')[1] != data['docker_image_tag']:
                                        try:
                                            remove_result = subprocess.run(
                                                ['docker', 'rmi', image],
                                                capture_output=True,
                                                text=True
                                            )
                                            if remove_result.returncode == 0:
                                                logger.info(f"Removed unused image: {image}")
                                            else:
                                                logger.warning(f"Failed to remove image {image}: {remove_result.stderr}")
                                        except Exception as e:
                                            logger.error(f"Error removing image {image}: {e}")
                        except Exception as e:
                            logger.error(f"Error during cleanup: {e}")

                        template = f"{data['docker_image']}:{data['docker_image_tag']}"
                        logger.info(f"Received most used template: {template} (usage count: {data['usage_count']})")
                        
                        try:
                            image_size = data['docker_image_size']
                            
                            # Check if we have enough space (3x image size)
                            if available_space < (image_size * min_disk_space_multiplier):
                                logger.warning(
                                    f"Skipping pull of {template} - insufficient disk space. "
                                    f"Required: {image_size * min_disk_space_multiplier}, "
                                    f"Available: {available_space}"
                                )
                            else:
                                # Pull the image
                                pull_result = subprocess.run(
                                    ['docker', 'pull', template],
                                    capture_output=True,
                                    text=True
                                )
                                if pull_result.returncode == 0:
                                    logger.info(f"Successfully pulled {template}")
                                else:
                                    logger.error(f"Failed to pull {template}: {pull_result.stderr}")
                        except Exception as e:
                            logger.error(f"Error processing template {template}: {e}")
                else:
                    logger.error(f"Failed to get template. Status: {response.status}")
            
        except Exception as e:
            logger.error(f"Error in Docker management loop: {e}")
            await asyncio.sleep(60)  # Wait before retrying


@click.command()
@click.option("--program_id", prompt="Program ID", help="Program ID for monitoring")
@click.option("--signature", prompt="Signature", help="Signature for verification")
@click.option("--executor_id", prompt="Executor ID", help="Executor ID")
@click.option("--validator_hotkey", prompt="Validator Hotkey", help="Validator hotkey")
@click.option("--compute_rest_app_url", prompt="Compute-app Url", help="Compute-app Url")
@click.option("--interval", default=5, type=int, help="Scraping interval in seconds")
def main(
    interval: int,
    program_id: str,
    signature: str,
    executor_id: str,
    validator_hotkey: str,
    compute_rest_app_url: str,
):
    async def run_all():
        await asyncio.gather(
            # scrape_gpu_metrics(
            #     interval, program_id, signature, executor_id, validator_hotkey, compute_rest_app_url
            # ),
            manage_docker_images(
                compute_rest_app_url, validator_hotkey, executor_id, program_id, signature
            )
        )

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
