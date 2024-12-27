import time
import pynvml
import click
import websockets
import json
import asyncio
from src.core.config import settings
from src.clients.executor_client import ExecutorClient
import logging

logger = logging.getLogger(__name__)

async def scrape_gpu_metrics(interval: int, program_id: str, signature: str, executor_id: str, validator_hotkey: str):
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
    
    ws_url = settings.COMPUTE_APP_URI + f"/{validator_hotkey}"
    print(f"Attempting to connect to WebSocket server at: {ws_url}")  # Debug print
    
    # Keep trying to connect
    websocket = await connect(ws_url)
    
    logger.info(f"Scraping metrics for {device_count} GPUs...")
    try:
        while True:
            try:
                gpu_metrics = []
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Handle potential bytes or str return for name
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_util = utilization.gpu
                mem_used = memory.used / 1024**2
                mem_total = memory.total / 1024**2
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                print(f"{timestamp} | GPU {i} ({name}): GPU {gpu_util}% | Memory {mem_used:.2f}/{mem_total:.2f} MB")
                
                gpu_metrics.append({
                    "gpu_id": i,
                    "gpu_name": name,
                    "gpu_utilization": gpu_util,
                    "memory_usage": mem_used,
                    "memory_total": mem_total,
                })

                await asyncio.sleep(interval)
                payload = {
                    "gpus": gpu_metrics,
                    "timestamp": timestamp,
                    "program_id": program_id,
                    "signature": signature,
                    "executor_id": executor_id,
                }
                
                # If connection is lost, reconnect before sending
                if not websocket.open:
                    logger.info("Connection lost, reconnecting...")
                    websocket = await connect(ws_url)
                
                await send_payload_to_backend(websocket, payload)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
    except KeyboardInterrupt:
        logger.info("Stopping GPU scraping...")
    finally:
        logger.info("Closing WebSocket connection...")
        await websocket.close()
        pynvml.nvmlShutdown()

async def connect(websocket_url):
    execute_app_client = ExecutorClient(
        websocket_url
    )
    async with execute_app_client:
        await execute_app_client.run_forever()

async def send_payload_to_backend(websocket, payload):
    while True:
        try:
            payload_json = json.dumps(payload)    
            await websocket.send(payload_json)
            logger.info(f"Payload sent: {payload_json}")
            return
        except websockets.ConnectionClosed as exc:
            logger.warning(
                    f"validator connection to backend app closed with code {exc.code} and reason {exc.reason}, reconnecting..."
            )
        except Exception as e:
            logger.info(f"Error during WebSocket communication: {e}")

@click.command()
@click.option("--program_id", prompt="Program ID", help="Program ID for monitoring")
@click.option("--signature", prompt="Signature", help="Signature for verification")
@click.option("--executor_id", prompt="Executor ID", help="Executor ID")
@click.option("--validator_hotkey", prompt="Validator Hotkey", help="Validator hotkey")
@click.option("--interval", default=10, type=int, help="Scraping interval in seconds")
def main(interval: int, program_id: str, signature: str, executor_id: str, validator_hotkey: str):
    asyncio.run(scrape_gpu_metrics(interval, program_id, signature, executor_id, validator_hotkey))

if __name__ == "__main__":
    main()
