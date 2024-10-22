import asyncio
import logging
import random
import time
import uuid

import click
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import MinerJobRequestPayload

from core.utils import configure_logs_of_other_modules
from core.validator import Validator
from services.ioc import ioc
from services.miner_service import MinerService

configure_logs_of_other_modules()
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


def generate_random_ip():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))


@cli.command()
def debug_send_machine_specs_to_connector():
    """Debug sending machine specs to connector"""
    miner_service: MinerService = ioc["MinerService"]
    counter = 0

    while counter < 10:
        counter += 1
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
            "ram": {
                "available": 491930408,
                "free": 131653212,
                "total": 528012784,
                "used": 396359572,
            },
            "hard_disk": {"total": 20971520, "used": 13962880, "free": 7008640},
            "os": "Ubuntu 22.04.4 LTS",
        }
        asyncio.run(
            miner_service.publish_machine_specs(
                results=[
                    (
                        debug_specs,
                        ExecutorSSHInfo(
                            uuid=str(uuid.uuid4()),
                            address=generate_random_ip(),
                            port="8001",
                            ssh_username="test",
                            ssh_port=22,
                            python_path="test",
                            root_dir="test",
                        ),
                    )
                ],
                miner_hotkey="5Cco1xUS8kXuaCzAHAXZ36nr6mLzmY5B9ufxrfb8Q3HB6ZdN",
            )
        )

        asyncio.run(
            miner_service.publish_machine_specs(
                results=[
                    (
                        debug_specs,
                        ExecutorSSHInfo(
                            uuid=str(uuid.uuid4()),
                            address=generate_random_ip(),
                            port="8001",
                            ssh_username="test",
                            ssh_port=22,
                            python_path="test",
                            root_dir="test",
                        ),
                    )
                ],
                miner_hotkey="5Cco1xUS8kXuaCzAHAXZ36nr6mLzmY5B9ufxrfb8Q3HB6ZdN",
            )
        )

        time.sleep(2)


@cli.command()
def debug_set_weights():
    """Debug setting weights"""
    validator = Validator()
    subtensor = validator.get_subtensor()
    # fetch miners
    miners = validator.fetch_miners(subtensor)
    asyncio.run(validator.set_weights(miners=miners, subtensor=subtensor))


if __name__ == "__main__":
    cli()
