import asyncio
import datetime as dt
import logging

import bittensor
from asgiref.sync import sync_to_async

from core.config import settings

logger = logging.getLogger(__name__)


class AsyncMetagraphClient:
    def __init__(self, cache_time=dt.timedelta(minutes=5)):
        self.cache_time = cache_time
        self._metagraph_future = None
        self._future_lock = asyncio.Lock()
        self._cached_metagraph = None
        self._cache_timestamp = None
        self.config = settings.get_bittensor_config()

    async def get_metagraph(self, ignore_cache=False):
        future = None
        set_result = False
        if self._cached_metagraph is not None:
            if not ignore_cache and dt.datetime.now() - self._cache_timestamp < self.cache_time:
                return self._cached_metagraph
        async with self._future_lock:
            if self._metagraph_future is None:
                loop = asyncio.get_running_loop()
                future = self._metagraph_future = loop.create_future()
                set_result = True
            else:
                future = self._metagraph_future
        if set_result:
            try:
                result = await self._get_metagraph()
            except Exception as exc:
                future.set_exception(exc)
                raise
            else:
                future.set_result(result)
                self._cache_timestamp = dt.datetime.now()
                self._cached_metagraph = result
                return result
            finally:
                async with self._future_lock:
                    self._metagraph_future = None
        else:
            return await future

    def _get_subtensor(self):
        return bittensor.subtensor(config=self.config)

    @sync_to_async(thread_sensitive=False)
    def _get_metagraph(self):
        return self._get_subtensor().metagraph(netuid=settings.BITTENSOR_NETUID)

    async def periodic_refresh(self, period=None):
        if period is None:
            period = self.cache_time.total_seconds()
        while True:
            try:
                await self.get_metagraph(ignore_cache=True)
            except Exception as exc:
                msg = f"Failed to refresh metagraph: {exc}"
                logger.warning(msg)

            await asyncio.sleep(period)


async_metagraph_client = AsyncMetagraphClient()


async def get_miner_axon_info(hotkey: str) -> bittensor.AxonInfo:
    metagraph = await async_metagraph_client.get_metagraph()
    neurons = [n for n in metagraph.neurons if n.hotkey == hotkey]
    if not neurons:
        raise ValueError(f"Miner with {hotkey=} not present in this subnetwork")
    return neurons[0].axon_info


def create_metagraph_refresh_task(period=None):
    return asyncio.create_task(async_metagraph_client.periodic_refresh(period=period))
