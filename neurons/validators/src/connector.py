import asyncio
import logging

from clients.compute_client import ComputeClient

from core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


async def run_forever():
    logger.info("Compute app connector started.")
    keypair = settings.get_bittensor_wallet().get_hotkey()
    compute_app_client = ComputeClient(
        keypair, f"{settings.COMPUTE_APP_URI}/validator/{keypair.ss58_address}"
    )
    async with compute_app_client:
        await compute_app_client.run_forever()


if __name__ == "__main__":
    asyncio.run(run_forever())
