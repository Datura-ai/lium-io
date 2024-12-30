import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING

import bittensor
import numpy as np
from bittensor.utils.weight_utils import (
    convert_weights_and_uids_for_emit,
    process_weights_for_netuid,
)
from websockets.protocol import State as WebSocketClientState
from payload_models.payloads import MinerJobRequestPayload

from core.config import settings
from core.utils import _m, get_extra_info, get_logger
from services.docker_service import REPOSITORIES, DockerService
from services.file_encrypt_service import FileEncryptService
from services.miner_service import MinerService
from services.redis_service import EXECUTOR_COUNT_PREFIX, RedisService
from services.ssh_service import SSHService
from services.task_service import TaskService

if TYPE_CHECKING:
    from bittensor_wallet import Wallet

logger = get_logger(__name__)

SYNC_CYCLE = 12
WEIGHT_MAX_COUNTER = 6
MINER_SCORES_KEY = "miner_scores"


class Validator:
    wallet: "Wallet"
    netuid: int
    subtensor: bittensor.Subtensor

    def __init__(self, debug_miner=None):
        self.config = settings.get_bittensor_config()

        self.wallet = settings.get_bittensor_wallet()
        self.netuid = settings.BITTENSOR_NETUID

        self.should_exit = False
        self.is_running = False
        self.last_job_run_blocks = 0
        self.default_extra = {}

        self.subtensor = None
        self.set_subtensor()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.initiate_services())

        self.debug_miner = debug_miner

    async def initiate_services(self):
        ssh_service = SSHService()
        self.redis_service = RedisService()
        task_service = TaskService(
            ssh_service=ssh_service,
            redis_service=self.redis_service,
        )
        self.docker_service = DockerService(
            ssh_service=ssh_service,
            redis_service=self.redis_service,
        )
        self.miner_service = MinerService(
            ssh_service=ssh_service,
            task_service=task_service,
            redis_service=self.redis_service,
        )
        self.file_encrypt_service = FileEncryptService(ssh_service=ssh_service)

        # init miner_scores
        try:
            if await self.should_set_weights():
                self.miner_scores = {}

                # clear executor_counts
                try:
                    await self.redis_service.clear_all_executor_counts()
                    logger.info(
                        _m(
                            "[initiate_services] Cleared executor_counts",
                            extra=get_extra_info(self.default_extra),
                        ),
                    )
                except Exception as e:
                    logger.error(
                        _m(
                            "[initiate_services] Failed to clear executor_counts",
                            extra=get_extra_info({**self.default_extra, "error": str(e)}),
                        ),
                    )
            else:
                miner_scores_json = await self.redis_service.get(MINER_SCORES_KEY)
                if miner_scores_json is None:
                    logger.info(
                        _m(
                            "[initiate_services] No data found in Redis for MINER_SCORES_KEY, initializing empty miner_scores.",
                            extra=get_extra_info(self.default_extra),
                        ),
                    )
                    self.miner_scores = {}
                else:
                    self.miner_scores = json.loads(miner_scores_json)

            await self.redis_service.clear_all_ssh_ports()
        except Exception as e:
            logger.error(
                _m(
                    "[initiate_services] Failed to initialize miner_scores",
                    extra=get_extra_info({**self.default_extra, "error": str(e)}),
                ),
            )
            self.miner_scores = {}

        logger.info(
            _m(
                "[initiate_services] miner scores",
                extra=get_extra_info(
                    {
                        **self.default_extra,
                        **self.miner_scores,
                    }
                ),
            ),
        )

    def set_subtensor(self):
        try:
            if (
                self.subtensor
                and self.subtensor.substrate
                and self.subtensor.substrate.websocket
                and self.subtensor.substrate.websocket.state is WebSocketClientState.OPEN
            ):
                return

            logger.info(
                _m(
                    "Getting subtensor",
                    extra=get_extra_info(self.default_extra),
                ),
            )
            subtensor = bittensor.subtensor(config=self.config)

            # check registered
            self.check_registered(subtensor)

            self.subtensor = subtensor
        except Exception as e:
            logger.info(
                _m(
                    "[Error] Getting subtensor",
                    extra=get_extra_info({
                        ** self.default_extra,
                        "error": str(e),
                    }),
                ),
            )

    def check_registered(self, subtensor: bittensor.subtensor):
        try:
            if not subtensor.is_hotkey_registered(
                netuid=self.netuid,
                hotkey_ss58=self.wallet.get_hotkey().ss58_address,
            ):
                logger.error(
                    _m(
                        f"[check_registered] Wallet: {self.wallet} is not registered on netuid {self.netuid}.",
                        extra=get_extra_info(self.default_extra),
                    ),
                )
                exit()
            logger.info(
                _m(
                    "[check_registered] Validator is registered",
                    extra=get_extra_info(self.default_extra),
                ),
            )
        except Exception as e:
            logger.error(
                _m(
                    "[check_registered] Checking validator registered failed",
                    extra=get_extra_info({**self.default_extra, "error": str(e)}),
                ),
            )

    def get_metagraph(self):
        return self.subtensor.metagraph(netuid=self.netuid)

    def get_node(self):
        # return SubstrateInterface(url=self.config.subtensor.chain_endpoint)
        return self.subtensor.substrate

    def get_current_block(self):
        node = self.get_node()
        return node.query("System", "Number", []).value

    def get_weights_rate_limit(self):
        node = self.get_node()
        return node.query("SubtensorModule", "WeightsSetRateLimit", [self.netuid]).value

    def get_my_uid(self):
        metagraph = self.get_metagraph()
        return metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

    def get_tempo(self):
        return self.subtensor.tempo(self.netuid)

    def fetch_miners(self):
        logger.info(
            _m(
                "[fetch_miners] Fetching miners",
                extra=get_extra_info(self.default_extra),
            ),
        )

        if self.debug_miner:
            miners = [self.debug_miner]
        else:
            metagraph = self.get_metagraph()
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
        logger.info(
            _m(
                f"[fetch_miners] Found {len(miners)} miners",
                extra=get_extra_info(self.default_extra),
            ),
        )
        return miners

    async def set_weights(self, miners):
        logger.info(
            _m(
                "[set_weights] scores",
                extra=get_extra_info(
                    {
                        **self.default_extra,
                        **self.miner_scores,
                    }
                ),
            ),
        )

        if not self.miner_scores:
            logger.info(
                _m(
                    "[set_weights] No miner scores available, skipping set_weights.",
                    extra=get_extra_info(self.default_extra),
                ),
            )
            return

        uids = np.zeros(len(miners), dtype=np.int64)
        weights = np.zeros(len(miners), dtype=np.float32)
        for ind, miner in enumerate(miners):
            uids[ind] = miner.uid
            weights[ind] = self.miner_scores.get(miner.hotkey, 0.0)

        logger.info(
            _m(
                f"[set_weights] uids: {uids} weights: {weights}",
                extra=get_extra_info(self.default_extra),
            ),
        )

        metagraph = self.get_metagraph()
        processed_uids, processed_weights = process_weights_for_netuid(
            uids=uids,
            weights=weights,
            netuid=self.netuid,
            subtensor=self.subtensor,
            metagraph=metagraph,
        )

        logger.info(
            _m(
                f"[set_weights] processed_uids: {processed_uids} processed_weights: {processed_weights}",
                extra=get_extra_info(self.default_extra),
            ),
        )

        uint_uids, uint_weights = convert_weights_and_uids_for_emit(
            uids=processed_uids, weights=processed_weights
        )

        logger.info(
            _m(
                f"[set_weights] uint_uids: {uint_uids} uint_weights: {uint_weights}",
                extra=get_extra_info(self.default_extra),
            ),
        )

        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
        )
        if result is True:
            logger.info(
                _m(
                    "[set_weights] set weights successfully",
                    extra=get_extra_info(self.default_extra),
                ),
            )
        else:
            logger.error(
                _m(
                    "[set_weights] set weights failed",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "msg": msg,
                        }
                    ),
                ),
            )

        self.miner_scores = {}

        # clear executor_counts
        try:
            await self.redis_service.clear_all_executor_counts()
            logger.info(
                _m(
                    "[set_weights] Cleared executor_counts",
                    extra=get_extra_info(self.default_extra),
                ),
            )
        except Exception as e:
            logger.error(
                _m(
                    "[set_weights] Failed to clear executor_counts",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "error": str(e),
                        }
                    ),
                ),
            )

    def get_last_update(self, block):
        try:
            node = self.get_node()
            last_update_blocks = (
                block
                - node.query("SubtensorModule", "LastUpdate", [self.netuid]).value[
                    self.get_my_uid()
                ]
            )
        except Exception as e:
            logger.error(
                _m(
                    "[get_last_update] Error getting last update",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "error": str(e),
                        }
                    ),
                ),
            )
            # means that the validator is not registered yet. The validator should break if this is the case anyways
            last_update_blocks = 1000

        logger.info(
            _m(
                f"[get_last_update] last set weights successfully {last_update_blocks} blocks ago",
                extra=get_extra_info(self.default_extra),
            ),
        )
        return last_update_blocks

    async def should_set_weights(self) -> bool:
        """Check if current block is for setting weights."""
        try:
            current_block = self.get_current_block()
            last_update = self.get_last_update(current_block)
            tempo = self.get_tempo()
            weights_rate_limit = self.get_weights_rate_limit()

            blocks_till_epoch = tempo - (current_block + self.netuid + 1) % (tempo + 1)

            should_set_weights = last_update >= tempo * 2 or (
                blocks_till_epoch < 20 and last_update >= weights_rate_limit
            )

            logger.info(
                _m(
                    "[should_set_weights] Checking should set weights",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "weights_rate_limit": weights_rate_limit,
                            "tempo": tempo,
                            "current_block": current_block,
                            "last_update": last_update,
                            "blocks_till_epoch": blocks_till_epoch,
                            "should_set_weights": should_set_weights,
                        }
                    ),
                ),
            )
            return should_set_weights
        except Exception as e:
            logger.error(
                _m(
                    "[should_set_weights] Checking set weights failed",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "error": str(e),
                        }
                    ),
                ),
            )
            return False

    async def get_time_from_block(self, block: int):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                node = self.get_node()
                block_hash = node.get_block_hash(block)
                return datetime.fromtimestamp(
                    node.query("Timestamp", "Now", block_hash=block_hash).value / 1000
                ).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logger.error(
                    _m(
                        "[get_time_from_block] Error getting time from block",
                        extra=get_extra_info(
                            {
                                **self.default_extra,
                                "retries": retries,
                                "error": str(e),
                            }
                        ),
                    ),
                )
                retries += 1
        return "Unknown"

    async def sync(self):
        try:
            self.set_subtensor()

            logger.info(
                _m(
                    "[sync] Syncing at subtensor",
                    extra=get_extra_info(self.default_extra),
                ),
            )

            # fetch miners
            miners = self.fetch_miners()

            if await self.should_set_weights():
                await self.set_weights(miners=miners)

            current_block = self.get_current_block()
            logger.info(
                _m(
                    "[sync] Current block",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "current_block": current_block,
                        }
                    ),
                ),
            )

            if current_block - self.last_job_run_blocks >= settings.BLOCKS_FOR_JOB:
                job_block = (current_block // settings.BLOCKS_FOR_JOB) * settings.BLOCKS_FOR_JOB
                job_batch_id = await self.get_time_from_block(job_block)

                logger.info(
                    _m(
                        "[sync] Send jobs to miners",
                        extra=get_extra_info(
                            {
                                **self.default_extra,
                                "miners": len(miners),
                                "current_block": current_block,
                                "job_batch_id": job_batch_id,
                            }
                        ),
                    ),
                )

                self.last_job_run_blocks = current_block

                docker_hub_digests = await self.docker_service.get_docker_hub_digests(REPOSITORIES)
                logger.info(
                    _m(
                        "Docker Hub Digests",
                        extra=get_extra_info(
                            {"job_batch_id": job_batch_id, "docker_hub_digests": docker_hub_digests}
                        ),
                    ),
                )

                encypted_files = self.file_encrypt_service.ecrypt_miner_job_files()

                task_info = {}

                # request jobs
                jobs = [
                    asyncio.create_task(
                        self.miner_service.request_job_to_miner(
                            payload=MinerJobRequestPayload(
                                job_batch_id=job_batch_id,
                                miner_hotkey=miner.hotkey,
                                miner_address=miner.axon_info.ip,
                                miner_port=miner.axon_info.port,
                            ),
                            encypted_files=encypted_files,
                            docker_hub_digests=docker_hub_digests,
                        )
                    )
                    for miner in miners
                ]

                for miner, job in zip(miners, jobs):
                    task_info[job] = {
                        "miner_hotkey": miner.hotkey,
                        "miner_address": miner.axon_info.ip,
                        "miner_port": miner.axon_info.port,
                        "job_batch_id": job_batch_id,
                    }

                try:
                    # Run all jobs with asyncio.wait and set a timeout
                    done, pending = await asyncio.wait(jobs, timeout=60 * 10)

                    # Process completed jobs
                    for task in done:
                        try:
                            result = task.result()
                            if result:
                                logger.info(
                                    _m(
                                        "[sync] Job_Result",
                                        extra=get_extra_info(
                                            {
                                                **self.default_extra,
                                                "result": result,
                                            }
                                        ),
                                    ),
                                )
                                miner_hotkey = result.get("miner_hotkey")
                                job_score = result.get("score")

                                key = f"{EXECUTOR_COUNT_PREFIX}:{miner_hotkey}"

                                try:
                                    executor_counts = await self.redis_service.hgetall(key)
                                    parsed_counts = [
                                        {
                                            "job_batch_id": job_id.decode("utf-8"),
                                            **json.loads(data.decode("utf-8")),
                                        }
                                        for job_id, data in executor_counts.items()
                                    ]

                                    if parsed_counts:
                                        logger.info(
                                            _m(
                                                "[sync] executor counts list",
                                                extra=get_extra_info(
                                                    {
                                                        **self.default_extra,
                                                        "miner_hotkey": miner_hotkey,
                                                        "parsed_counts": parsed_counts,
                                                    }
                                                ),
                                            ),
                                        )

                                        max_executors = max(
                                            parsed_counts, key=lambda x: x["total"]
                                        )["total"]
                                        min_executors = min(
                                            parsed_counts, key=lambda x: x["total"]
                                        )["total"]

                                        logger.info(
                                            _m(
                                                "[sync] executor counts",
                                                extra=get_extra_info(
                                                    {
                                                        **self.default_extra,
                                                        "miner_hotkey": miner_hotkey,
                                                        "job_batch_id": job_batch_id,
                                                        "max_executors": max_executors,
                                                        "min_executors": min_executors,
                                                    }
                                                ),
                                            ),
                                        )

                                except Exception as e:
                                    logger.error(
                                        _m(
                                            "[sync] Get executor counts error",
                                            extra=get_extra_info(
                                                {
                                                    **self.default_extra,
                                                    "miner_hotkey": miner_hotkey,
                                                    "job_batch_id": job_batch_id,
                                                    "error": str(e),
                                                }
                                            ),
                                        ),
                                    )

                                if miner_hotkey in self.miner_scores:
                                    self.miner_scores[miner_hotkey] += job_score
                                else:
                                    self.miner_scores[miner_hotkey] = job_score
                            else:
                                info = task_info.get(task, {})
                                miner_hotkey = info.get("miner_hotkey", "unknown")
                                job_batch_id = info.get("job_batch_id", "unknown")
                                logger.error(
                                    _m(
                                        "[sync] No_Job_Result",
                                        extra=get_extra_info(
                                            {
                                                **self.default_extra,
                                                "miner_hotkey": miner_hotkey,
                                                "job_batch_id": job_batch_id,
                                            }
                                        ),
                                    ),
                                )

                        except Exception as e:
                            logger.error(
                                _m(
                                    "[sync] Error processing job result",
                                    extra=get_extra_info(
                                        {
                                            **self.default_extra,
                                            "job_batch_id": job_batch_id,
                                            "error": str(e),
                                        }
                                    ),
                                ),
                            )

                    # Handle pending jobs (those that did not complete within the timeout)
                    if pending:
                        for task in pending:
                            info = task_info.get(task, {})
                            miner_hotkey = info.get("miner_hotkey", "unknown")
                            job_batch_id = info.get("job_batch_id", "unknown")

                            logger.error(
                                _m(
                                    "[sync] Job_Timeout",
                                    extra=get_extra_info(
                                        {
                                            **self.default_extra,
                                            "miner_hotkey": miner_hotkey,
                                            "job_batch_id": job_batch_id,
                                        }
                                    ),
                                ),
                            )
                            task.cancel()

                    logger.info(
                        _m(
                            "[sync] All Jobs finished",
                            extra=get_extra_info(
                                {
                                    **self.default_extra,
                                    "job_batch_id": job_batch_id,
                                    "miner_scores": self.miner_scores,
                                }
                            ),
                        ),
                    )

                except Exception as e:
                    logger.error(
                        _m(
                            "[sync] Unexpected error",
                            extra=get_extra_info(
                                {
                                    **self.default_extra,
                                    "job_batch_id": job_batch_id,
                                    "error": str(e),
                                }
                            ),
                        ),
                    )
            else:
                remaining_blocks = (
                    current_block // settings.BLOCKS_FOR_JOB + 1
                ) * settings.BLOCKS_FOR_JOB - current_block

                logger.info(
                    _m(
                        "[sync] Remaining blocks for next job",
                        extra=get_extra_info(
                            {
                                **self.default_extra,
                                "remaining_blocks": remaining_blocks,
                                "last_job_run_blocks": self.last_job_run_blocks,
                                "current_block": current_block,
                            }
                        ),
                    ),
                )
        except Exception as e:
            logger.error(
                _m(
                    "[sync] Unknown error",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "error": str(e),
                        }
                    ),
                ),
            )

    async def start(self):
        logger.info(
            _m(
                "[start] Starting Validator in background",
                extra=get_extra_info(self.default_extra),
            ),
        )
        try:
            while not self.should_exit:
                await self.sync()

                # sync every 12 seconds
                await asyncio.sleep(SYNC_CYCLE)

        except KeyboardInterrupt:
            logger.info(
                _m(
                    "[start] Validator killed by keyboard interrupt",
                    extra=get_extra_info(self.default_extra),
                ),
            )
            exit()
        except Exception as e:
            logger.info(
                _m(
                    "[start] Unknown error",
                    extra=get_extra_info({**self.default_extra, "error": str(e)}),
                ),
            )

    async def stop(self):
        logger.info(
            _m(
                "[stop] Stopping Validator process",
                extra=get_extra_info(self.default_extra),
            ),
        )

        try:
            await self.redis_service.set(MINER_SCORES_KEY, json.dumps(self.miner_scores))
        except Exception as e:
            logger.info(
                _m(
                    "[stop] Failed to save miner_scores",
                    extra=get_extra_info({**self.default_extra, "error": str(e)}),
                ),
            )

        self.should_exit = True
