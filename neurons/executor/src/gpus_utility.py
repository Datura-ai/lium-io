import asyncio
import logging
import subprocess
import time

import aiohttp
import click
import gpu_ghost_cure
import psutil
import pynvml
from services.pull_lock import cache_pull_lock

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GHOST_CONSECUTIVE_SAMPLES = 3
GHOST_CURE_COOLDOWN_SECONDS = 600


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
    min_disk_space_multiplier: float = 3.0,  # Minimum disk space required (3x image size)
    first_interval: int = 60 * 5,  # Retry interval when errors occur (network, Docker commands, etc.)
    success_interval: int = 2 * 60 * 60,  # Wait interval after successful Docker pull
):
    """
    Manages Docker images on the executor by:
    1. Syncing with backend for most used templates
    2. Cleaning up unused images
    3. Ensuring sufficient disk space

    Args:
        compute_rest_app_url (str): URL of the compute REST application
        min_disk_space_multiplier (float): Minimum disk space required as multiple of image size
    """
    # Get GPU name using NVML
    gpu_name = "unknown"
    driver_version = "unknown"
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get first GPU
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        pynvml.nvmlShutdown()
    except Exception as e:
        logger.error(f"Failed to get GPU name: {e}")

    backend_url = f"{compute_rest_app_url}/executors/default-docker-image?gpu_model={gpu_name}&driver_version={driver_version}"

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Get disk space info
                disk = psutil.disk_usage('/')
                available_space = disk.free

                async with session.get(backend_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get template. Status: {response.status}")
                        await asyncio.sleep(first_interval)
                        continue

                    docker_images_data = await response.json()
                    logger.info(f"Received {len(docker_images_data)} docker image templates")
                    
                    # Check if no templates were returned
                    if not docker_images_data:
                        logger.warning(f"No templates found for GPU {gpu_name}")
                        await asyncio.sleep(first_interval)
                        continue
                    
                    had_error = False
                    for data in docker_images_data:
                        docker_image = data['docker_image']
                        docker_image_tag = data['docker_image_tag']
                        docker_image_size = data['docker_image_size']
                        
                        # Check if no templates were found for the GPU
                        if not docker_image or not docker_image_tag:
                            logger.warning(f"No templates found for GPU {gpu_name}")
                            await asyncio.sleep(first_interval)
                            continue

                        template = f"{docker_image}:{docker_image_tag}"
                        logger.info(f"Processing template: {template}")

                        # Clean up unused images
                        docker_list_result = subprocess.run(
                            ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'],
                            capture_output=True,
                            text=True
                        )
                        
                        if docker_list_result.returncode == 0:
                            local_images = docker_list_result.stdout.strip().split('\n')
                            
                            for image in local_images:
                                if not image:
                                    continue
                                    
                                repo, tag = image.split(':')
                                if repo == docker_image and tag != docker_image_tag:
                                    remove_result = subprocess.run(
                                        ['docker', 'rmi', image],
                                        capture_output=True,
                                        text=True
                                    )
                                    if remove_result.returncode == 0:
                                        logger.info(f"Removed unused image: {image}")
                                    else:
                                        logger.warning(f"Failed to remove image {image}: {remove_result.stderr}")
                        else:
                            logger.warning(f"Failed to list Docker images: {docker_list_result.stderr}")

                        # Process template and pull image
                        required_space = int(docker_image_size * min_disk_space_multiplier)
                        
                        if available_space < required_space:
                            logger.warning(
                                f"Skipping pull of {template} - insufficient disk space. "
                                f"Required: {required_space}, Available: {available_space}"
                            )
                            await asyncio.sleep(first_interval)
                            continue

                        # Pull the image (guarded so we don't pull the same image
                        # concurrently with the executor's on-boot cache pre-pull).
                        with cache_pull_lock() as acquired:
                            if not acquired:
                                logger.info(
                                    f"Another puller holds the lock; skipping {template} this cycle"
                                )
                                continue
                            pull_result = subprocess.run(
                                ['docker', 'pull', template],
                                capture_output=True,
                                text=True
                            )

                        if pull_result.returncode == 0:
                            logger.info(f"Successfully pulled {template}")
                        else:
                            logger.error(f"Failed to pull {template}: {pull_result.stderr}")
                            had_error = True

                    # Sleep once per full sweep: retry sooner if anything failed.
                    await asyncio.sleep(first_interval if had_error else success_interval)

            except aiohttp.ClientError as e:
                logger.error(f"Network error: {e}")
                await asyncio.sleep(first_interval)
            except subprocess.CalledProcessError as e:
                logger.error(f"Docker command failed: {e.stderr}")
                await asyncio.sleep(first_interval)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(first_interval)


async def ghost_watchdog(interval: int = 60):
    # cure GPUs stuck in the Blackwell GSP utilization latch ("ghost GPU"): util pinned
    # at 100% with ~0 MiB and zero processes, sustained across samples -> run a clean
    # CUDA context open/close in a child process (see gpu_ghost_cure.py, DAH-2431)
    streak: dict[str, int] = {}
    last_cure: dict[str, float] = {}

    while True:
        try:
            ghosts = await asyncio.to_thread(gpu_ghost_cure.detect_ghosts)
            streak = {uuid: streak.get(uuid, 0) + 1 for uuid in ghosts}
            for uuid, samples in streak.items():
                if samples < GHOST_CONSECUTIVE_SAMPLES:
                    continue
                if time.time() - last_cure.get(uuid, 0.0) < GHOST_CURE_COOLDOWN_SECONDS:
                    continue
                logger.warning(
                    f"Ghost GPU {uuid}: util latched for {samples} samples, running cure"
                )
                last_cure[uuid] = time.time()
                cured = await asyncio.to_thread(gpu_ghost_cure.spawn_cure, uuid)
                logger.warning(
                    f"Ghost GPU {uuid}: cure result: {'CLEARED' if cured else 'STILL LATCHED'}"
                )
        except Exception as e:
            logger.error(f"Ghost watchdog error: {e}")
        await asyncio.sleep(interval)


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
            manage_docker_images(compute_rest_app_url),
            ghost_watchdog(),
        )

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
