import asyncio
import logging

from clients.compute_client import ComputeClient

from core.config import settings
from core.utils import wait_for_services_sync
from services.ioc import ioc

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


wait_for_services_sync()


async def run_forever():
    logger.info("Compute app connector started.")
    keypair = settings.get_bittensor_wallet().get_hotkey()
    compute_app_client = ComputeClient(
        keypair, f"{settings.COMPUTE_APP_URI}/validator/{keypair.ss58_address}", ioc["MinerService"]
    )
    async with compute_app_client:
        await compute_app_client.run_forever()


def start_process():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_forever())


if __name__ == "__main__":
    start_process()

# def start_connector_process():
#     p = multiprocessing.Process(target=start_process)
#     p.start()
#     return p
