import asyncio
import bittensor
import numpy as np
from typing import TYPE_CHECKING
from bittensor.utils.weight_utils import (
    convert_weights_and_uids_for_emit,
    process_weights_for_netuid,
)
from websockets.protocol import State as WebSocketClientState
import random
from datetime import datetime

from core.config import settings
from core.utils import _m, get_extra_info, get_logger

from services.const import TOTAL_BURN_EMISSION, BURNER_EMISSION
from services.redis_service import NORMALIZED_SCORE_CHANNEL, RedisService

if TYPE_CHECKING:
    from bittensor_wallet import bittensor_wallet

logger = get_logger(__name__)

SYNC_CYCLE = 12


class SubtensorClient:
    # Static class variables (shared across all instances)
    _instance = None
    _initialized = False
    _subtensor = None
    _warm_up_task = None

    wallet: "bittensor_wallet"
    miners: list[bittensor.NeuronInfo] = []
    uid_to_evm_address: dict[int, str] = {}
    hotkey_to_evm_address: dict[str, str] = {}

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of SubtensorClient."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Prevent multiple initializations
        if SubtensorClient._initialized:
            return

        # Instance variables (belong to this specific instance)
        self.wallet = settings.get_bittensor_wallet()
        self.netuid = settings.BITTENSOR_NETUID
        self.config = settings.get_bittensor_config()
        self.redis_service = RedisService()

        # Calculate version key
        major, minor, patch = map(int, settings.VERSION.split('.'))
        self.version_key = major * 10000 + minor * 100 + patch

        self.default_extra = {
            "version_key": self.version_key,
        }

        # Set debug miner if configured
        self.debug_miner = None
        if settings.DEBUG:
            self.debug_miner = settings.get_debug_miner()

        self.initialize_subtensor()

        SubtensorClient._initialized = True

        logger.info(
            _m(
                "SubtensorClient initialized",
                extra=get_extra_info(self.default_extra),
            ),
        )

    # Property for static variable
    @property
    def subtensor(self):
        return SubtensorClient._subtensor

    def initialize_subtensor(self):
        try:
            logger.info(
                _m(
                    "Initializing subtensor",
                    extra=get_extra_info(self.default_extra),
                ),
            )
            subtensor = bittensor.subtensor(config=self.config)

            # check registered
            self.check_registered(subtensor)

            SubtensorClient._subtensor = subtensor
        except Exception as e:
            logger.info(
                _m(
                    "[Error] failed initializing subtensor",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "error": str(e),
                        }
                    ),
                ),
            )

    def set_subtensor(self):
        if (
            SubtensorClient._subtensor
            and SubtensorClient._subtensor.substrate
            and SubtensorClient._subtensor.substrate.ws
            and not SubtensorClient._subtensor.substrate.ws.close_code
        ):
            return

        self.initialize_subtensor()

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

    def get_evm_address_for_hotkey(self, hotkey):
        return self.hotkey_to_evm_address.get(hotkey, None)

    def sync_evm_address_maps(self):
        node = self.get_node()
        associated_evms = node.query_map(module="SubtensorModule", storage_function="AssociatedEvmAddress", params=[self.netuid])
        for uid, evm_address in associated_evms:
            address_bytes = evm_address.value[0][0]
            evm_address_hex = "0x" + bytes(address_bytes).hex()
            self.uid_to_evm_address[uid] = evm_address_hex

        """Update the map of miner_hotkey -> evm_address for all miners."""
        for miner in self.miners:
            self.hotkey_to_evm_address[miner.hotkey] = self.uid_to_evm_address.get(miner.uid, None)

        logger.info(
            _m(
                "Synced ethereum addresses map",
                extra=get_extra_info({
                    **self.default_extra,
                    "uid_to_evm_address": len(self.uid_to_evm_address),
                }),
            ),
        )

    def get_my_uid(self):
        return self.get_uid_for_hotkey(self.wallet.hotkey.ss58_address)

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
                if neuron.axon_info.is_serving or neuron.uid in settings.BURNERS
            ]
        logger.info(
            _m(
                f"[fetch_miners] Found {len(miners)} miners",
                extra=get_extra_info(self.default_extra),
            ),
        )

        self.miners = miners

    def get_miner(self, hotkey: str) -> bittensor.NeuronInfo:
        miners = self.get_miners()

        neurons = [n for n in miners if n.hotkey == hotkey]
        if not neurons:
            raise ValueError(f"Miner with {hotkey=} not present in this subnetwork")
        return neurons[0]

    def get_miners(self) -> list[bittensor.NeuronInfo]:
        if not self.miners:
            self.fetch_miners()
        return self.miners

    async def set_weights(self, miner_scores: dict[str, float]):
        miners = self.get_miners()
        logger.info(
            _m(
                "[set_weights] scores",
                extra=get_extra_info(
                    {
                        **self.default_extra,
                        **miner_scores,
                    }
                ),
            ),
        )

        if not miner_scores:
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
        total_score = sum(miner_scores.values())
        for ind, miner in enumerate(miners):
            uids[ind] = miner.uid
            miner_hotkeys.append(metagraph.hotkeys[miner.uid])
            if miner.uid == main_burner:
                weights[ind] = TOTAL_BURN_EMISSION - (len(settings.BURNERS) - 1) * BURNER_EMISSION
            elif miner.uid in other_burners:
                weights[ind] = BURNER_EMISSION
            else:
                weights[ind] = 0 if total_score <= 0 else (1 - TOTAL_BURN_EMISSION) * miner_scores.get(miner.hotkey, 0.0) / total_score

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

    async def _warm_up_subtensor(self):
        count = 0
        while True:
            try:
                self.set_subtensor()

                if count == 0:
                    self.fetch_miners()
                    self.sync_evm_address_maps()

                count += 1
                if count > 10:
                    self.fetch_miners()
                    self.sync_evm_address_maps()
                    count = 1

                # sync every 12 seconds
                await asyncio.sleep(SYNC_CYCLE)
            except Exception as e:
                logger.error(
                    _m(
                        "[_warm_up_subtensor] Failed to connect into subtensor",
                        extra=get_extra_info({**self.default_extra, "error": str(e)}),
                    ),
                )
                self.initialize_subtensor()
                await asyncio.sleep(SYNC_CYCLE)

    @classmethod
    async def initialize(cls):
        """Initialize the singleton instance asynchronously."""
        instance = cls.get_instance()

        # Start warm-up task only once (static)
        if cls._warm_up_task is None or cls._warm_up_task.done():
            cls._warm_up_task = asyncio.create_task(instance._warm_up_subtensor())

        return instance

    @classmethod
    async def shutdown(cls):
        """Shutdown the singleton instance and cancel the warm-up task."""
        if cls._warm_up_task and not cls._warm_up_task.done():
            cls._warm_up_task.cancel()
            try:
                await cls._warm_up_task
            except asyncio.CancelledError:
                pass
        cls._warm_up_task = None
        cls._instance = None
        cls._initialized = False

    @classmethod
    def get_subtensor(cls):
        """Get the subtensor instance directly."""
        if cls._subtensor is None:
            instance = cls.get_instance()
            instance.set_subtensor()
        return cls._subtensor
