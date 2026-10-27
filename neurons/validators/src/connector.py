import asyncio
import time

from clients.compute_client import ComputeClient, mark_connector_unready

from core.config import settings
from core.utils import get_logger, wait_for_services_sync, widen_default_thread_pool
from services.ioc import initiate_services, ioc

logger = get_logger(__name__)
wait_for_services_sync()


async def run_forever():
    logger.info("Compute app connector started")
    mark_connector_unready()  # DAH-5177: an in-process restart is not ready until it waits on the lead lock again
    await initiate_services()
    keypair = settings.get_bittensor_wallet().get_hotkey()
    compute_app_client = ComputeClient(
        keypair, f"{settings.COMPUTE_APP_URI}/validator/{keypair.ss58_address}", ioc["MinerService"]
    )
    async with compute_app_client:
        await compute_app_client.run_forever()


def start_process():
    while True:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            widen_default_thread_pool(loop)
            loop.run_until_complete(run_forever())
        except Exception as e:
            logger.error(f"Compute app connector crashed: {e}", exc_info=True)
            time.sleep(1)


if __name__ == "__main__":
    start_process()

# def start_connector_process():
#     p = multiprocessing.Process(target=start_process)
#     p.start()
#     return p
