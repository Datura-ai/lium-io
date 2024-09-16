import asyncio
import logging

import click
from payload_models.payloads import MinerJobRequestPayload

from services.ioc import ioc

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


if __name__ == "__main__":
    cli()
