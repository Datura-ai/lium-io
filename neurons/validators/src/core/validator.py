import logging
import asyncio
import traceback

import bittensor
from core.config import settings
from core.db import get_db
from services.ssh_service import SSHService
from services.task_service import TaskService
from services.miner_service import MinerService, MinerRequestPayload
from daos.task import TaskDao

logger = logging.getLogger(__name__)

class Validator():
    wallet: bittensor.wallet
    subtensor: bittensor.subtensor
    netuid: int
    
    def __init__(self):
        config = settings.get_bittensor_config()

        self.wallet = settings.get_bittensor_wallet()
        self.subtensor = bittensor.subtensor(config=config)
        self.netuid = settings.BITTENSOR_NETUID
        
        self.check_registered()
        
        self.should_exit = False
        self.is_running = False
        
        # set miner service
        session = next(get_db())
        self.task_dao = TaskDao(session=session)

        ssh_service = SSHService()
        task_service = TaskService(task_dao=self.task_dao, ssh_service=ssh_service)
        self.miner_service = MinerService(ssh_service=ssh_service, task_service=task_service)
        
    async def check_registered(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.netuid,
            hotkey_ss58=self.wallet.get_hotkey().ss58_address,
        ):
            bittensor.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()
            
    async def fetch_minors(self):
        metagraph = self.subtensor.metagraph(netuid=self.netuid)
        miners = [neuron for neuron in metagraph.neurons if neuron.axon_info.is_serving]
        
        return miners
    
    async def set_weights(self):
        pass
    
    async def sync(self):
        bittensor.logging.debug('sync validator')
        # check registered
        await self.check_registered()

        # fetch miners
        miners = await self.fetch_minors()
        
        # run jobs
        for miner in miners:
            print(miner.hotkey)
            print(miner.axon_info.ip)
            print(miner.axon_info.port)
        
        
    async def start(self):
        bittensor.logging.debug('Start validator in background')
        try:
            while not self.should_exit:
                await self.sync()
                
                # sync every 10 mins
                await asyncio.sleep(2 * 60)
                
        except KeyboardInterrupt:
            logger.debug('Miner killed by keyboard interrupt.')
            exit()
        except Exception as e:
            logger.error(traceback.format_exc())
            
    async def stop(self):
        bittensor.logging.debug('Stop validator in background')
        self.should_exit = True