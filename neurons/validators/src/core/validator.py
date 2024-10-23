import asyncio
import logging
import traceback
import json

import bittensor
import numpy as np
from bittensor.utils.weight_utils import (
    convert_weights_and_uids_for_emit,
    process_weights_for_netuid,
)
from payload_models.payloads import MinerJobRequestPayload

from core.config import settings
from services.docker_service import DockerService
from services.miner_service import MinerService
from services.ssh_service import SSHService
from services.task_service import TaskService
from services.redis_service import RedisService

logger = logging.getLogger(__name__)

SYNC_CYCLE = 12
WEIGHT_MAX_COUNTER = 6
MINER_SCORES_KEY = "miner_scores"


class Validator:
    wallet: bittensor.wallet
    netuid: int

    def __init__(self):
        self.config = settings.get_bittensor_config()

        self.wallet = settings.get_bittensor_wallet()
        self.netuid = settings.BITTENSOR_NETUID

        self.should_exit = False
        self.is_running = False
        self.last_job_run_blocks = 0

        subtensor = self.get_subtensor()

        # check registered
        self.check_registered(subtensor)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.initiate_services(subtensor))

    async def initiate_services(self, subtensor: bittensor.subtensor):
        ssh_service = SSHService()
        self.redis_service = RedisService()
        task_service = TaskService(
            ssh_service=ssh_service,
            redis_service=self.redis_service,
        )
        docker_service = DockerService(
            ssh_service=ssh_service,
            redis_service=self.redis_service,
        )
        self.miner_service = MinerService(
            ssh_service=ssh_service,
            task_service=task_service,
            docker_service=docker_service,
            redis_service=self.redis_service,
        )

        # init miner_scores
        try:
            if await self.should_set_weights(subtensor):
                self.miner_scores = {}
            else:
                miner_scores_json = await self.redis_service.get(MINER_SCORES_KEY)
                if miner_scores_json is None:
                    bittensor.logging.info("No data found in Redis for MINER_SCORES_KEY, initializing empty miner_scores.")
                    self.miner_scores = {}
                else:
                    self.miner_scores = json.loads(miner_scores_json)
        except Exception as e:
            bittensor.logging.error(f"Failed to initialize miner_scores: {str(e)}")
            self.miner_scores = {}

        bittensor.logging.info(f"miner scores: {self.miner_scores}", "init", "init")

    def get_subtensor(self):
        bittensor.logging.debug("Getting subtensor", "get_subtensor", "get_subtensor")
        return bittensor.subtensor(config=self.config)

    def get_metagraph(self, subtensor: bittensor.subtensor):
        return subtensor.metagraph(netuid=self.netuid)

    def get_node(self, subtensor: bittensor.subtensor):
        # return SubstrateInterface(url=self.config.subtensor.chain_endpoint)
        return subtensor.substrate

    def get_current_block(self, subtensor: bittensor.subtensor):
        node = self.get_node(subtensor)
        return node.query("System", "Number", []).value

    def get_weights_rate_limit(self, subtensor: bittensor.subtensor):
        node = self.get_node(subtensor)
        return node.query("SubtensorModule", "WeightsSetRateLimit", [self.netuid]).value

    def get_my_uid(self, subtensor: bittensor.subtensor):
        metagraph = self.get_metagraph(subtensor)
        return metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

    def get_tempo(self, subtensor: bittensor.subtensor):
        return subtensor.tempo(self.netuid)

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
        bittensor.logging.debug("Fetching miners started", "fetch_miners", "fetch_miners")
        metagraph = self.get_metagraph(subtensor)
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

    async def set_weights(self, miners, subtensor: bittensor.subtensor):
        bittensor.logging.info(f"[set_weights] scores: {self.miner_scores}")

        if not self.miner_scores:
            bittensor.logging.info("No miner scores available, skipping set_weights.")
            return

        for miner_hotkey in self.miner_scores.keys():
            bittensor.logging.info(
                "Total score for miner(%s) is %f",
                "set_weights",
                "set_weights",
                miner_hotkey,
                self.miner_scores.get(miner_hotkey, 0.0),
            )

        uids = np.zeros(len(miners), dtype=np.int64)
        weights = np.zeros(len(miners), dtype=np.float32)
        for ind, miner in enumerate(miners):
            uids[ind] = miner.uid
            weights[ind] = self.miner_scores.get(miner.hotkey, 0.0)

        bittensor.logging.info(f"uids: {uids}")
        bittensor.logging.info(f"weights: {weights}")

        metagraph = self.get_metagraph(subtensor)
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

        bittensor.logging.info('Reset miner scores')
        self.miner_scores = {}

    def get_last_update(self, subtensor: bittensor.subtensor, block):
        try:
            node = self.get_node(subtensor)
            last_update_blocks = (
                block
                - node.query("SubtensorModule", "LastUpdate", [self.netuid]).value[
                    self.get_my_uid(subtensor)
                ]
            )
        except Exception:
            bittensor.logging.error(f"Error getting last update: {traceback.format_exc()}")
            # means that the validator is not registered yet. The validator should break if this is the case anyways
            last_update_blocks = 1000

        bittensor.logging.info(f"last set weights successfully {last_update_blocks} blocks ago")
        return last_update_blocks

    async def should_set_weights(self, subtensor: bittensor.subtensor) -> bool:
        """Check if current block is for setting weights."""
        try:
            current_block = self.get_current_block(subtensor)
            last_update = self.get_last_update(subtensor, current_block)
            tempo = self.get_tempo(subtensor)
            weights_rate_limit = self.get_weights_rate_limit(subtensor)

            blocks_till_epoch = tempo - (current_block + self.netuid + 1) % (tempo + 1)
            bittensor.logging.info(
                "Checking should set weights(weights_rate_limit=%d, tempo=%d): current_block=%d, last_update=%d, blocks_till_epoch=%d",
                "should_set_weights",
                "should_set_weights",
                weights_rate_limit,
                tempo,
                current_block,
                last_update,
                blocks_till_epoch,
            )
            return last_update >= tempo * 2 or (
                blocks_till_epoch < 20 and last_update >= weights_rate_limit
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
        try:
            subtensor = self.get_subtensor()
            bittensor.logging.info("Syncing at subtensor %s", "sync", "sync", subtensor)

            # fetch miners
            miners = self.fetch_miners(subtensor)

            if await self.should_set_weights(subtensor):
                await self.set_weights(miners=miners, subtensor=subtensor)

            current_block = self.get_current_block(subtensor)
            bittensor.logging.info(f"Current block: {current_block}", "sync", "sync")

            if (
                current_block % settings.BLOCKS_FOR_JOB == 0
                or current_block - self.last_job_run_blocks > int(settings.BLOCKS_FOR_JOB * 1.5)
            ):
                bittensor.logging.info(
                    "Send jobs to %d miners at block(%d)",
                    "sync",
                    "sync",
                    len(miners),
                    current_block,
                )

                self.last_job_run_blocks = current_block

                # request jobs
                jobs = [
                    asyncio.create_task(
                        self.miner_service.request_job_to_miner(
                            payload=MinerJobRequestPayload(
                                miner_hotkey=miner.hotkey,
                                miner_address=miner.axon_info.ip,
                                miner_port=miner.axon_info.port,
                            )
                        )
                    )
                    for miner in miners
                ]

                try:
                    results = await asyncio.wait_for(asyncio.gather(*jobs), timeout=60 * 10)
                    for result in results:
                        if result:
                            bittensor.logging.info(f"Job score: {result}", "sync", "sync")
                            miner_hotkey = result.get('miner_hotkey')
                            job_score = result.get('score')
                            if miner_hotkey in self.miner_scores:
                                self.miner_scores[miner_hotkey] += job_score
                            else:
                                self.miner_scores[miner_hotkey] = job_score

                    bittensor.logging.info(f"miner scores: {self.miner_scores}", "sync", "sync")

                    for index, result in enumerate(results):
                        miner = miners[index]
                        if isinstance(result, Exception):
                            bittensor.logging.error(
                                f"Job for miner({miner.hotkey}-{miner.axon_info.ip}:{miner.axon_info.port}) resulted in an exception: {result}",
                                "sync",
                                "sync",
                            )
                        else:
                            bittensor.logging.info(
                                f"Job for miner({miner.hotkey}-{miner.axon_info.ip}:{miner.axon_info.port}) completed successfully: {result}",
                                "sync",
                                "sync",
                            )

                    bittensor.logging.info("All Jobs finished", "sync", "sync")
                    bittensor.logging.info(f"miner_scores: {self.miner_scores}", "sync", "sync")
                except TimeoutError:
                    bittensor.logging.error("Tasks timed out!", "sync", "sync")
                    # Cancel all tasks
                    for index, job in enumerate(jobs):
                        if not job.done():
                            bittensor.logging.error(
                                f"Cancelling job for miner({miners[index].hotkey}-{miners[index].axon_info.ip}:{miners[index].axon_info.port})",
                                "sync",
                                "sync",
                            )
                            job.cancel()
            else:
                remaining_blocks = (
                    current_block // settings.BLOCKS_FOR_JOB + 1
                ) * settings.BLOCKS_FOR_JOB - current_block
                bittensor.logging.info(
                    "Remaining blocks %d for next job. Last job run at block %d, current block %d",
                    "sync",
                    "sync",
                    remaining_blocks,
                    self.last_job_run_blocks,
                    current_block,
                )
        except Exception:
            bittensor.logging.error(
                f"Error in running task: {traceback.format_exc()}", "sync", "sync"
            )

    async def start(self):
        bittensor.logging.info("Start Validator in background", "start", "start")
        try:
            while not self.should_exit:
                await self.sync()

                # sync every 12 seconds
                await asyncio.sleep(SYNC_CYCLE)

        except KeyboardInterrupt:
            bittensor.logging.error("Miner killed by keyboard interrupt.", "start", "start")
            exit()
        except Exception:
            bittensor.logging.error(
                f"Error in running task: {traceback.format_exc()}", "start", "start"
            )

    async def stop(self):
        bittensor.logging.info("Stop Validator process")

        try:
            await self.redis_service.set(MINER_SCORES_KEY, json.dumps(self.miner_scores))
        except Exception as e:
            bittensor.logging.error(f"Failed to save miner_scores: {str(e)}")

        self.should_exit = True
