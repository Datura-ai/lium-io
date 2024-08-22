import logging
import asyncio
import traceback

import bittensor
from bittensor.utils.weight_utils import process_weights_for_netuid, convert_weights_and_uids_for_emit
from core.config import settings
from core.db import get_db
from services.ssh_service import SSHService
from services.task_service import TaskService
from services.miner_service import MinerService, MinerRequestPayload
from daos.task import TaskDao
import numpy as np


logger = logging.getLogger(__name__)

SYNC_CYCLE = 2 * 60
WEIGHT_MAX_COUNTER = 5

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
        
        hotkey_to_score = {score.miner_hotkey: score.avg_score for score in avg_scores}

        uids = np.zeros(len(miners), dtype=np.int64)
        weights = np.zeros(len(miners), dtype=np.float32)
        for ind, miner in enumerate(miners):
            uids[ind] = miner.uid
            weights[ind] = hotkey_to_score.get(miner.hotkey, 0.0)
            
        bittensor.logging.info(f"uids: {uids}")
        bittensor.logging.info(f"weights: {weights}")
        
        metagraph = subtensor.metagraph(netuid=self.netuid)
        processed_uids, processed_weights = process_weights_for_netuid(
            uids=uids,
            weights=weights,
            netuid=self.netuid,
            subtensor=subtensor,
            metagraph=metagraph,
        )
        
        bittensor.logging.info(f"processed_uids: {processed_uids}")
        bittensor.logging.info(f"processed_weights: {processed_weights}")
        
        uint_uids, uint_weights = convert_weights_and_uids_for_emit(
            uids=processed_uids, weights=processed_weights
        )
        
        bittensor.logging.info(f"uint_uids: {uint_uids}")
        bittensor.logging.info(f"uint_weights: {uint_weights}")
        
        result, msg = subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
        )
        if result is True:
            bittensor.logging.info("set_weights on chain successfully!")
        else:
            bittensor.logging.error("set_weights failed", msg)
    
    async def sync(self):
        self.weight_counter += 1

        subtensor = self.get_subtensor()
        # check registered
        await self.check_registered(subtensor)

        # fetch miners
        miners = await self.fetch_minors(subtensor)
        for miner in miners:
            bittensor.logging.info(f"miner {miner.hotkey}")
        
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