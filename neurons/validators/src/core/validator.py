import asyncio
import logging
import traceback

import bittensor
import numpy as np
from bittensor.utils.weight_utils import (
    convert_weights_and_uids_for_emit,
    process_weights_for_netuid,
)
from payload_models.payloads import MinerJobRequestPayload
from substrateinterface import SubstrateInterface

from core.config import settings
from core.db import get_db
from daos.executor import ExecutorDao
from daos.task import TaskDao
from services.miner_service import MinerService
from services.ssh_service import SSHService
from services.task_service import TaskService
from services.docker_service import DockerService
from daos.task import TaskDao
from daos.executor import ExecutorDao
import numpy as np
from payload_models.payloads import MinerJobRequestPayload


logger = logging.getLogger(__name__)

SYNC_CYCLE = 12
WEIGHT_MAX_COUNTER = 6


class Validator:
    wallet: bittensor.wallet
    subtensor: bittensor.subtensor
    netuid: int

    def __init__(self):
        self.config = settings.get_bittensor_config()

        self.wallet = settings.get_bittensor_wallet()
        self.netuid = settings.BITTENSOR_NETUID
        self.subtensor = bittensor.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.netuid)
        self.my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)

        self.should_exit = False
        self.is_running = False

        # set miner service
        session = next(get_db())
        self.task_dao = TaskDao(session=session)
        executor_dao = ExecutorDao(session=session)

        # Get network tempo
        self.tempo = self.subtensor.tempo(self.netuid)
        self.weights_rate_limit = self.get_weights_rate_limit()

        ssh_service = SSHService()
        task_service = TaskService(task_dao=self.task_dao, ssh_service=ssh_service, executor_dao=executor_dao)
        docker_service = DockerService(ssh_service=ssh_service, executor_dao=executor_dao)
        self.miner_service = MinerService(ssh_service=ssh_service, task_service=task_service, docker_service=docker_service)
        
        self.weight_counter = 0
        
    def get_subtensor(self):
        return bittensor.subtensor(config=self.config)
        
        task_service = TaskService(
            task_dao=self.task_dao, ssh_service=ssh_service, executor_dao=executor_dao
        )
        self.miner_service = MinerService(ssh_service=ssh_service, task_service=task_service)

    def check_registered(self, subtensor: bittensor.subtensor):
        try:
            if not subtensor.is_hotkey_registered(
                netuid=self.netuid,
                hotkey_ss58=self.wallet.get_hotkey().ss58_address,
            ):
                bittensor.logging.error(
                    f"Wallet: {self.wallet} is not registered on netuid {self.netuid}."
                    f" Please register the hotkey using `btcli subnets register` before trying again"
                )
                exit()
            bittensor.logging.info("Validator is registered")
        except Exception as e:
            bittensor.logging.error("Checking validator registered failed: %s", str(e))

    def fetch_miners(self, subtensor: bittensor.subtensor):
        metagraph = subtensor.metagraph(netuid=self.netuid)
        miners = [
            neuron
            for neuron in metagraph.neurons
            if neuron.axon_info.is_serving
            and (
                not settings.DEBUG
                or not settings.DEBUG_MINER_HOTKEY
                or settings.DEBUG_MINER_HOTKEY == neuron.axon_info.hotkey
            )
        ]
        bittensor.logging.info("Found %d miners", "fetch_miners", "fetch_miners", len(miners))
        return miners

    def get_my_uid(self, subtensor: bittensor.subtensor):
        return subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=self.wallet.get_hotkey().ss58_address, netuid=self.netuid
        )

    def set_weights(self, miners, subtensor: bittensor.subtensor):
        scores = self.task_dao.get_scores_for_last_epoch(self.tempo)

        hotkey_to_score = {score.miner_hotkey: score.total_score for score in scores}

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

    def get_last_update(self, block):
        try:
            last_update_blocks = (
                block
                - self.node.query("SubtensorModule", "LastUpdate", [self.netuid]).value[self.my_uid]
            )
        except Exception:
            bittensor.logging.error(f"Error getting last update: {traceback.format_exc()}")
            # means that the validator is not registered yet. The validator should break if this is the case anyways
            last_update_blocks = 1000

        bittensor.logging.info(f"last set weights successfully {last_update_blocks} blocks ago")
        return last_update_blocks

    def get_current_block(self):
        return self.node.query("System", "Number", []).value

    def get_weights_rate_limit(self):
        return self.node.query("SubtensorModule", "WeightsSetRateLimit", [self.netuid]).value

    async def should_set_weights(self) -> bool:
        """Check if current block is for setting weights."""
        try:
            current_block = self.get_current_block()
            last_update = self.get_last_update(current_block)
            blocks_till_epoch = self.tempo - (current_block + self.netuid + 1) % (self.tempo + 1)
            bittensor.logging.info(
                "Checking should set weights(weights_rate_limit=%d, tempo=%d): current_block=%d, last_update=%d, blocks_till_epoch=%d",
                "should_set_weights",
                "should_set_weights",
                self.weights_rate_limit,
                self.tempo,
                current_block,
                last_update,
                blocks_till_epoch,
            )
            return last_update >= self.tempo * 2 or (
                blocks_till_epoch < 20 and last_update >= self.weights_rate_limit
            )
        except Exception as e:
            bittensor.logging.error(
                "Checking set weights failed: error=%s",
                "should_set_weights",
                "should_set_weights",
                str(e),
            )
            return False

    async def sync(self):
        subtensor = self.subtensor
        # check registered
        self.check_registered(subtensor)

        # fetch miners
        try:
            miners = self.fetch_miners(subtensor)
        except Exception:
            miners = []

        if await self.should_set_weights():
            self.set_weights(miners, subtensor)

        current_block = self.get_current_block()
        if current_block % settings.BLOCKS_FOR_JOB == 0:
            bittensor.logging.info(
                "Send jobs to %d miners at block(%d)", "sync", "sync", len(miners), current_block
            )

            # request jobs
            jobs = [
                asyncio.create_task(
                    asyncio.wait_for(
                        self.miner_service.request_job_to_miner(
                            payload=MinerJobRequestPayload(
                                miner_hotkey=miner.hotkey,
                                miner_address=miner.axon_info.ip,
                                miner_port=miner.axon_info.port,
                            )
                        ),
                        timeout=60 * 5,
                    )
                )
                for miner in miners
            ]

            await asyncio.gather(*jobs, return_exceptions=True)
        else:
            remaining_blocks = (
                current_block // settings.BLOCKS_FOR_JOB + 1
            ) * settings.BLOCKS_FOR_JOB - current_block
            bittensor.logging.info(
                "Remaining blocks %d for next job", "sync", "sync", remaining_blocks
            )

    async def start(self):
        bittensor.logging.info("Start Validator in background")
        try:
            while not self.should_exit:
                await self.sync()

                # sync every 10 mins
                await asyncio.sleep(SYNC_CYCLE)

        except KeyboardInterrupt:
            logger.error("Miner killed by keyboard interrupt.")
            exit()
        except Exception:
            logger.error(traceback.format_exc())

    async def stop(self):
        bittensor.logging.info("Stop Validator process")
        self.should_exit = True
