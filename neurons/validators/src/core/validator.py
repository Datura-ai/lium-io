import asyncio
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING
import random

import bittensor
import numpy as np
from bittensor.utils.weight_utils import (
    convert_weights_and_uids_for_emit,
    process_weights_for_netuid,
)
from payload_models.payloads import MinerJobRequestPayload
from websockets.protocol import State as WebSocketClientState

from core.config import settings
from core.utils import _m, get_extra_info, get_logger
from services.docker_service import DockerService
from services.file_encrypt_service import FileEncryptService
from services.miner_service import MinerService
from services.redis_service import PENDING_PODS_PREFIX, NORMALIZED_SCORE_CHANNEL, RedisService
from services.ssh_service import SSHService
from services.task_service import TaskService, JobResult
from services.matrix_validation_service import ValidationService
from services.collateral_contract_service import CollateralContractService
from services.const import GPU_MODEL_RATES, TOTAL_BURN_EMISSION, BURNER_EMISSION, JOB_TIME_OUT

if TYPE_CHECKING:
    from bittensor_wallet import bittensor_wallet

logger = get_logger(__name__)

SYNC_CYCLE = 12
WEIGHT_MAX_COUNTER = 6
MINER_SCORES_KEY = "miner_scores"


class Validator:
    wallet: "bittensor_wallet"
    netuid: int
    subtensor: bittensor.Subtensor

    def __init__(self, debug_miner=None):
        self.config = settings.get_bittensor_config()

        self.wallet = settings.get_bittensor_wallet()
        self.netuid = settings.BITTENSOR_NETUID

        self.should_exit = False
        self.last_job_run_blocks = 0
        self.default_extra = {}

        self.subtensor = None
        self.debug_miner = debug_miner
        self.miner_scores = {}
        self.evm_address_map = {}  # Map: miner_hotkey -> evm_address

        major, minor, patch = map(int, settings.VERSION.split('.'))
        self.version_key = major * 1000 + minor * 100 + patch

    async def initiate_services(self):
        ssh_service = SSHService()
        self.redis_service = RedisService()
        self.file_encrypt_service = FileEncryptService(ssh_service=ssh_service)
        self.validation_service = ValidationService()
        self.collateral_contract_service = CollateralContractService()
        task_service = TaskService(
            ssh_service=ssh_service,
            redis_service=self.redis_service,
            validation_service=self.validation_service,
            collateral_contract_service=self.collateral_contract_service
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

        # init miner_scores
        try:
            if await self.should_set_weights():
                self.miner_scores = {}
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

            # remove pod renting-in-progress status
            await self.redis_service.delete(PENDING_PODS_PREFIX)
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
                and self.subtensor.substrate.ws
                and self.subtensor.substrate.ws.state is WebSocketClientState.OPEN
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
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "error": str(e),
                        }
                    ),
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

    def get_last_mechansim_step_block(self):
        node = self.get_node()
        return node.query("SubtensorModule", "LastMechansimStepBlock", [self.netuid]).value

    def get_uid_for_hotkey(self, hotkey):
        metagraph = self.get_metagraph()
        return metagraph.hotkeys.index(hotkey)
    
    def get_associated_evm_address(self, hotkey):
        if self.subtensor is None:
            self.set_subtensor()
        node = self.get_node()
        uid = self.get_uid_for_hotkey(hotkey)
        associated_evm = node.query("SubtensorModule", "AssociatedEvmAddress", [self.netuid, uid])

        if associated_evm is None or associated_evm.value is None:
            return None
        # associated_evm.value is expected to be a tuple: ((address_bytes,), block_number)
        value = associated_evm.value
        address_bytes_tuple = value[0][0]  # value[0] is a tuple with one element: the address bytes
        # Convert bytes tuple to bytes, then to hex string
        address_bytes = bytes(address_bytes_tuple)
        evm_address_hex = '0x' + address_bytes.hex()
        return evm_address_hex

    def get_my_uid(self):
        metagraph = self.get_metagraph()
        return metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

    def get_tempo(self):
        return self.subtensor.tempo(self.netuid)

    async def update_evm_address_map(self, miners=None):
        """Update the map of miner_hotkey -> evm_address for all miners."""
        if miners is None:
            miners = self.fetch_miners()
        for miner in miners:
            try:
                evm_address = self.get_associated_evm_address(miner.hotkey)
                self.evm_address_map[miner.hotkey] = evm_address
            except Exception as e:
                logger.error(_m(
                    f"[update_evm_address_map] Error getting EVM address for miner {miner.hotkey}",
                    extra=get_extra_info({**self.default_extra, "error": str(e)})
                ))

        logger.info(
            _m(
                "[update_evm_address_map] Updated ethereum addresses map",
                extra=get_extra_info({**self.default_extra, "evm_address_map": self.evm_address_map})
            ),
        )

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
                if neuron.axon_info.is_serving or neuron.uid in settings.BURNERS
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

        last_mechansim_step_block = self.get_last_mechansim_step_block()
        main_burner = random.Random(last_mechansim_step_block).choice(settings.BURNERS)
        logger.info(
            _m(
                "[set_weights] main burner",
                extra=get_extra_info({
                    "last_mechansim_step_block": last_mechansim_step_block,
                    "main_burner": main_burner,
                }),
            ),
        )
        other_burners = [uid for uid in settings.BURNERS if uid != main_burner]

        metagraph = self.get_metagraph()
        miner_hotkeys = []
        total_score = sum(self.miner_scores.values())
        if total_score <= 0:
            uids[0] = main_burner
            weights[0] = 1 - (len(settings.BURNERS) - 1) * BURNER_EMISSION
            miner_hotkeys.append(metagraph.hotkeys[main_burner])
            for ind, uid in enumerate(other_burners):
                uids[ind + 1] = uid
                weights[ind + 1] = BURNER_EMISSION
                miner_hotkeys.append(metagraph.hotkeys[uid])
        else:
            for ind, miner in enumerate(miners):
                uids[ind] = miner.uid
                miner_hotkeys.append(metagraph.hotkeys[miner.uid])
                if miner.uid == main_burner:
                    weights[ind] = TOTAL_BURN_EMISSION - (len(settings.BURNERS) - 1) * BURNER_EMISSION
                elif miner.uid in other_burners:
                    weights[ind] = BURNER_EMISSION
                else:
                    weights[ind] = (1 - TOTAL_BURN_EMISSION) * self.miner_scores.get(miner.hotkey, 0.0) / total_score

            # uids[ind] = miner.uid
            # weights[ind] = self.miner_scores.get(miner.hotkey, 0.0)

        logger.debug(
            _m(
                f"[set_weights] uids: {uids} weights: {weights}",
                extra=get_extra_info(self.default_extra),
            ),
        )
        normalized_scores = [
            {"uid": int(uid), "weight": float(weight), "miner_hotkey": miner_hotkey}
            for uid, weight, miner_hotkey in zip(uids, weights, miner_hotkeys)
        ]
        message = {
            "normalized_scores": normalized_scores,
        }
        await self.redis_service.publish(NORMALIZED_SCORE_CHANNEL, message)

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
                extra=get_extra_info({
                    **self.default_extra,
                    "version_key": self.version_key,
                }),
            ),
        )

        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=uint_uids,
            weights=uint_weights,
            version_key=self.version_key,
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

            should_set_weights = last_update >= tempo

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

    async def calc_job_score(self, total_gpu_model_count_map: dict, job_result: JobResult):
        if job_result.score == 0:
            logger.info(
                _m(
                    "Debug: No need to calc score, score is 0",
                    extra=get_extra_info({
                        "executor_id": str(job_result.executor_info.uuid),
                        "job_batch_id": job_result.job_batch_id,
                        "gpu_model": job_result.gpu_model,
                        "gpu_count": job_result.gpu_count,
                        "score": 0,
                        "sysbox_runtime": job_result.sysbox_runtime,
                    })
                )
            )
            return 0

        gpu_model_rate = GPU_MODEL_RATES.get(job_result.gpu_model, 0)
        total_gpu_count = total_gpu_model_count_map.get(job_result.gpu_model, 0)

        if total_gpu_count == 0:
            return 0

        revenue_per_gpu_type = await self.redis_service.get_revenue_per_gpu_type(job_result.gpu_model)
        score_portion = gpu_model_rate + settings.TIME_DELTA_FOR_EMISSION * (revenue_per_gpu_type - gpu_model_rate)
        score = score_portion * job_result.gpu_count / total_gpu_count

        # get uptime of the executor
        uptime_in_minutes = await self.redis_service.get_executor_uptime(job_result.executor_info)
        # if uptime in the subnet is exceed 5 days, then it'll get max score
        # if uptime is less than 5 days, then it'll get score based on the uptime
        # give 50% of max still to avoid 0 score all miners at deployment
        # If sysbox_runtime is true, then the score will be increased by PORTION_FOR_SYSBOX per cent.
        five_days_in_minutes = 60 * 24 * 5
        score = score * (settings.PORTION_FOR_UPTIME + min((1 - settings.PORTION_FOR_UPTIME), uptime_in_minutes / five_days_in_minutes))

        if job_result.sysbox_runtime:
            score = score * (1 + settings.PORTION_FOR_SYSBOX)

        logger.info(
            _m(
                "Debug: calculating score",
                extra=get_extra_info({
                    "executor_id": str(job_result.executor_info.uuid),
                    "job_batch_id": job_result.job_batch_id,
                    "gpu_model": job_result.gpu_model,
                    "total_gpu_count": total_gpu_count,
                    "gpu_count": job_result.gpu_count,
                    "gpu_model_rate": gpu_model_rate,
                    "revenue_per_gpu_type": revenue_per_gpu_type,
                    "score_portion": score_portion,
                    "score": score,
                    "sysbox_runtime": job_result.sysbox_runtime,
                })
            )
        )

        return score

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

            try:
                if await self.should_set_weights():
                    await self.set_weights(miners=miners)
            except Exception as e:
                logger.error(
                    _m(
                        "[sync] Error setting weights",
                        extra=get_extra_info(
                            {
                                **self.default_extra,
                                "error": str(e),
                            }
                        ),
                    ),
                )
            
            # Update EVM address map for all miners
            try:
                asyncio.create_task(self.update_evm_address_map(miners))
            except Exception as e:
                logger.error(_m(
                    "[fetch_miners] Error updating EVM address map",
                    extra=get_extra_info({**self.default_extra, "error": str(e)})
                ))
                    
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

                encrypted_files = self.file_encrypt_service.ecrypt_miner_job_files()

                task_info = {}

                # request jobs
                jobs = [
                    asyncio.create_task(
                        self.miner_service.request_job_to_miner(
                            payload=MinerJobRequestPayload(
                                job_batch_id=job_batch_id,
                                miner_hotkey=miner.hotkey,
                                miner_coldkey=miner.coldkey,
                                miner_address=miner.axon_info.ip,
                                miner_port=miner.axon_info.port,
                            ),
                            encrypted_files=encrypted_files,
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
                    total_gpu_model_count_map = {}
                    all_job_results = {}

                    # Run all jobs with asyncio.wait and set a timeout
                    done, pending = await asyncio.wait(jobs, timeout=JOB_TIME_OUT - 50)

                    # Process completed jobs
                    for task in done:
                        try:
                            result = task.result()
                            if result:
                                miner_hotkey = result.get("miner_hotkey")
                                job_results: list[JobResult] = result.get("results", [])

                                logger.info(
                                    _m(
                                        "[sync] Job_Result",
                                        extra=get_extra_info(
                                            {
                                                **self.default_extra,
                                                "miner_hotkey": miner_hotkey,
                                                "results": len(job_results)
                                            }
                                        ),
                                    ),
                                )

                                all_job_results[miner_hotkey] = job_results

                                for job_result in job_results:
                                    total_gpu_model_count_map[job_result.gpu_model] = total_gpu_model_count_map.get(job_result.gpu_model, 0) + job_result.gpu_count

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

                    open_fd_count = len(os.listdir(f'/proc/self/fd'))

                    logger.info(
                        _m(
                            "Total GPU model count",
                            extra=get_extra_info(
                                {
                                    **self.default_extra,
                                    "job_batch_id": job_batch_id,
                                    "data": total_gpu_model_count_map,
                                }
                            ),
                        ),
                    )

                    for miner_hotkey, results in all_job_results.items():
                        for result in results:
                            score = await self.calc_job_score(total_gpu_model_count_map, result)
                            self.miner_scores[miner_hotkey] = self.miner_scores.get(miner_hotkey, 0) + score

                    logger.info(
                        _m(
                            "[sync] All Jobs finished",
                            extra=get_extra_info(
                                {
                                    **self.default_extra,
                                    "job_batch_id": job_batch_id,
                                    "miner_scores": self.miner_scores,
                                    "open_fd_count": open_fd_count,
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
                exc_info=True,
            )

    async def start(self):
        logger.info(
            _m(
                "[start] Starting Validator in background",
                extra=get_extra_info(self.default_extra),
            ),
        )
        try:
            self.set_subtensor()
            await self.initiate_services()
            self.should_exit = False

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

    async def warm_up_subtensor(self):
        while True:
            try:
                self.set_subtensor()

                # sync every 12 seconds
                await asyncio.sleep(SYNC_CYCLE)
            except Exception as e:
                logger.error(
                    _m(
                        "[stop] Failed to connect into subtensor",
                        extra=get_extra_info({**self.default_extra, "error": str(e)}),
                    ),
                )
                await asyncio.sleep(SYNC_CYCLE)
