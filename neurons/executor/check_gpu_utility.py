import time
import pynvml
import click
import asyncio
import websockets
import json
from src.core.config import settings

@click.command()
@click.option("--program_id", prompt="Program ID", help="Program ID for monitoring")
@click.option("--signature", prompt="Signature", help="Signature for verification")
@click.option("--executor_id", prompt="Executor ID", help="Executor ID")
@click.option("--validator_hotkey", prompt="Validator Hotkey", help="Validator hotkey")
@click.option("--public_key", prompt="Public Key", help="Public Key")
@click.option("--interval", default=10, type=int, help="Scraping interval in seconds")
def scrape_gpu_metrics(interval: int, program_id: str, signature: str, executor_id: str, validator_hotkey: str, public_key: str):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    print(f"Scraping metrics for {device_count} GPUs...")
    try:
        while True:
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

            time.sleep(interval)
            signature = "signature"
            payload = {
                "gpu_utilization": gpu_util,
                "memory_usage": mem_used,
                "memory_total": mem_total,
                "timestamp": timestamp,
                "program_id": program_id,
                "signature": signature,
                "executor_id": executor_id,
                "program_id": program_id,
                "validator_hotkey": validator_hotkey,
                "public_key": public_key,
            }
            # send payload to the backend-app using websocket
            send_payload_to_backend(payload, settings.COMPUTE_APP_URI)
            
    except KeyboardInterrupt:
        print("Stopping GPU scraping...")


async def send_payload_to_backend(payload, websocket_url):
    try:
        async with websockets.connect(websocket_url) as websocket:
            payload_json = json.dumps(payload)    
            

            await websocket.send(payload_json)
            print(f"Payload sent: {payload_json}")

            response = await websocket.recv()
            print(f"Response received: {response}")
    except Exception as e:
        print(f"Error during WebSocket communication: {e}")


if __name__ == "__main__":
    scrape_gpu_metrics()
