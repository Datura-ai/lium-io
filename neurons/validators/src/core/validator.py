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

SYNC_CYCLE = 2 * 60
WEIGHT_MAX_COUNTER = 24

class Validator():
    wallet: bittensor.wallet
    subtensor: bittensor.subtensor
    netuid: int
    
    def __init__(self):
        self.config = settings.get_bittensor_config()

        self.wallet = settings.get_bittensor_wallet()
        self.netuid = settings.BITTENSOR_NETUID
        
        self.should_exit = False
        self.is_running = False
        
        # set miner service
        session = next(get_db())
        self.task_dao = TaskDao(session=session)

        ssh_service = SSHService()
        task_service = TaskService(task_dao=self.task_dao, ssh_service=ssh_service)
        self.miner_service = MinerService(ssh_service=ssh_service, task_service=task_service)
        
        self.weight_counter = 0
        
    def get_subtensor(self):
        return bittensor.subtensor(config=self.config)
        
    async def check_registered(self, subtensor: bittensor.subtensor):
        if not subtensor.is_hotkey_registered(
            netuid=self.netuid,
            hotkey_ss58=self.wallet.get_hotkey().ss58_address,
        ):
            bittensor.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()
        bittensor.logging.info('Validator is registered')
            
    async def fetch_minors(self, subtensor: bittensor.subtensor):
        metagraph = subtensor.metagraph(netuid=self.netuid)
        miners = [neuron for neuron in metagraph.neurons if neuron.axon_info.is_serving]
        
        return miners
    
    def set_weights(self, miners, subtensor: bittensor.subtensor):
        avg_scores = self.task_dao.get_avg_scores_in_hours(24)
        
        hotkey_to_uid = {miner.hotkey: miner.uid for miner in miners}
        uids = [hotkey_to_uid[score.miner_hotkey] for score in avg_scores if score.miner_hotkey in hotkey_to_uid]

        scores = [score.avg_score for score in avg_scores]
        bittensor.logging.ino(f"uids: {uids}")
        bittensor.logging.ino(f"scores: {scores}")
        
        # result, msg = subtensor.set_weights(
        #     wallet=self.wallet,
        #     netuid=self.netuid,
        #     uids=uint_uids,
        #     weights=uint_weights,
        #     wait_for_finalization=False,
        #     wait_for_inclusion=False,
        # )
        # if result is True:
        #     bittensor.logging.info("set_weights on chain successfully!")
        # else:
        #     bittensor.logging.error("set_weights failed", msg)
    
    async def sync(self):
        self.weight_counter += 1

        subtensor = self.get_subtensor()
        # check registered
        await self.check_registered(subtensor)

        # fetch miners
        miners = await self.fetch_minors(subtensor)
        
        if self.weight_counter >= WEIGHT_MAX_COUNTER:
            self.weight_counter = 0
            self.set_weights(miners, subtensor)
        
        # request jobs
        jobs = [
            asyncio.create_task(
                asyncio.wait_for(
                self.miner_service.request_resource_to_miner(payload=MinerRequestPayload(
                    miner_hotkey=miner.hotkey,
                    miner_address=miner.axon_info.ip,
                    miner_port=miner.axon_info.port
                )),
                timeout=SYNC_CYCLE - 5
            )
            )
            for miner in miners
        ]
        
        await asyncio.gather(*jobs, return_exceptions=True)
        
    async def start(self):
        bittensor.logging.info('Start Validator in background')
        try:
            while not self.should_exit:
                await self.sync()
                
                # sync every 10 mins
                await asyncio.sleep(SYNC_CYCLE)
                
        except KeyboardInterrupt:
            logger.error('Miner killed by keyboard interrupt.')
            exit()
        except Exception as e:
            logger.error(traceback.format_exc())
            
    async def stop(self):
        bittensor.logging.info('Stop Validator process')
        self.should_exit = True