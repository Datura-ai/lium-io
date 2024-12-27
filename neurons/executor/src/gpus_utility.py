import time
import pynvml
import click
import asyncio
import aiohttp
from clients.executor_client import ExecutorClient
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class GPUMetricsTracker:
    def __init__(self, threshold_percent: float = 10.0):
        self.previous_metrics: Dict[int, Dict] = {}
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
    
async def scrape_gpu_metrics(interval: int, program_id: str, signature: str, executor_id: str, validator_hotkey: str, compute_rest_app_url: str):
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            logger.warning("No NVIDIA GPUs found in the system")
            return
    except pynvml.NVMLError as e:
        logger.error(f"Failed to initialize NVIDIA Management Library: {e}")
        logger.error("This might be because no NVIDIA GPU is present or drivers are not properly installed")
        return

    http_url = f"{compute_rest_app_url}/validator/{validator_hotkey}"
    logger.info(f"Will send metrics to: {http_url}")
    
    # Initialize the tracker
    tracker = GPUMetricsTracker(threshold_percent=10.0)
    
    async with aiohttp.ClientSession() as session:
        logger.info(f"Scraping metrics for {device_count} GPUs...")
        try:
            while True:
                try:
                    gpu_metrics = []
                    should_send = False

                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        
                        name = pynvml.nvmlDeviceGetName(handle)
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')
                        
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

                        gpu_util = utilization.gpu
                        mem_used = memory.used / 1024**2
                        mem_total = memory.total / 1024**2
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                        # Check if there's a significant change for this GPU
                        if tracker.has_significant_change(i, gpu_util, mem_used):
                            should_send = True
                            logger.info(f"Significant change detected for GPU {i}")

                        print(f"{timestamp} | GPU {i} ({name}): GPU {gpu_util}% | Memory {mem_used:.2f}/{mem_total:.2f} MB")
                        
                        gpu_metrics.append({
                            "gpu_id": i,
                            "gpu_name": name,
                            "gpu_utilization": gpu_util,
                            "memory_usage": mem_used,
                            "memory_total": mem_total,
                        })

                    # Only send if there's a significant change in any GPU
                    if should_send:
                        payload = {
                            "gpus": gpu_metrics,
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

async def connect(websocket_url):
    execute_app_client = ExecutorClient(websocket_url)
    # Start the client's processing in the background
    asyncio.create_task(execute_app_client.run_forever())
    return execute_app_client

@click.command()
@click.option("--program_id", prompt="Program ID", help="Program ID for monitoring")
@click.option("--signature", prompt="Signature", help="Signature for verification")
@click.option("--executor_id", prompt="Executor ID", help="Executor ID")
@click.option("--validator_hotkey", prompt="Validator Hotkey", help="Validator hotkey")
@click.option("--compute_rest_app_url", prompt="Compute-app Url", help="Compute-app Url")
@click.option("--interval", default=5, type=int, help="Scraping interval in seconds")
def main(interval: int, program_id: str, signature: str, executor_id: str, validator_hotkey: str, compute_rest_app_url: str):
    asyncio.run(scrape_gpu_metrics(interval, program_id, signature, executor_id, validator_hotkey, compute_rest_app_url))

if __name__ == "__main__":
    main()
