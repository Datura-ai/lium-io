import asyncio
import logging

import click
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import MinerJobRequestPayload

from services.ioc import ioc
from services.miner_service import MinerService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--miner_hotkey", prompt="Miner Hotkey", help="Hotkey of Miner")
@click.option("--miner_address", prompt="Miner Address", help="Miner IP Address")
@click.option("--miner_port", type=int, prompt="Miner Port", help="Miner Port")
def debug_send_job_to_miner(miner_hotkey: str, miner_address: str, miner_port: int):
    """Debug sending job to miner"""
    miner_service = ioc["MinerService"]
    asyncio.run(
        miner_service.request_job_to_miner(
            MinerJobRequestPayload(
                miner_hotkey=miner_hotkey, miner_address=miner_address, miner_port=miner_port
            )
        )
    )


@cli.command()
@click.option("--miner_hotkey", prompt="Miner Hotkey", help="Hotkey of Miner")
@click.option("--executor_address", prompt="Executor Address", help="Executor IP Address")
@click.option("--executor_port", type=int, prompt="Executor Port", help="Executor Port")
@click.option("--executor_uuid", type=str, prompt="Executor UUID", help="Executor UUID")
def debug_send_machine_specs_to_connector(
    miner_hotkey: str, executor_address: str, executor_port: int, executor_uuid: str
):
    """Debug sending machine specs to connector"""
    miner_service: MinerService = ioc["MinerService"]
    debug_specs = {
        "gpu": {
            "count": 1,
            "details": [
                {
                    "name": "NVIDIA RTX A5000",
                    "driver": "555.42.06",
                    "capacity": "24564",
                    "cuda": "8.6",
                    "power_limit": "230.00",
                    "graphics_speed": "435",
                    "memory_speed": "5000",
                    "pcei": "16",
                }
            ],
        },
        "cpu": {"count": 128, "model": "AMD EPYC 7452 32-Core Processor", "clocks": []},
        "ram": {"available": 491930408, "free": 131653212, "total": 528012784, "used": 396359572},
        "hard_disk": {"total": 20971520, "used": 13962880, "free": 7008640},
        "os": "Ubuntu 22.04.4 LTS",
    }
    asyncio.run(
        miner_service.publish_machine_specs(
            results=[
                (
                    debug_specs,
                    ExecutorSSHInfo(
                        uuid=executor_uuid,
                        address=executor_address,
                        port=executor_port,
                        ssh_username="",
                        ssh_port=22,
                        python_path="",
                        root_dir="",
                    ),
                )
            ],
            miner_hotkey=miner_hotkey,
        )
    )


if __name__ == "__main__":
    cli()
