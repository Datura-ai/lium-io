import asyncio
import time
import logging

from clients.miner_portal_client import MinerPortalClient

from core.utils import configure_logs_of_other_modules, wait_for_services_sync
from services.ioc import initiate_services

configure_logs_of_other_modules()
wait_for_services_sync()

logger = logging.getLogger(__name__)


async def run_forever():
    initiate_services()

    logger.info("Miner portal connector started")
    miner_portal_client = MinerPortalClient()
    async with miner_portal_client:
        await miner_portal_client.run_forever()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_forever())
