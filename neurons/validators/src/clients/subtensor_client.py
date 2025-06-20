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

    wallet: "bittensor_wallet"
    evm_address_map: dict[str, str] = {}

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
        self.version_key = major * 1000 + minor * 100 + patch

        self.default_extra = {
            "version_key": self.version_key,
        }

        # Set debug miner if configured
        self.debug_miner = None
        if settings.DEBUG:
            self.debug_miner = settings.get_debug_miner()

        # Start warm-up task only once (static)
        asyncio.create_task(self._warm_up_subtensor())

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

    def set_subtensor(self):
        try:
            if (
                SubtensorClient._subtensor
                and SubtensorClient._subtensor.substrate
                and SubtensorClient._subtensor.substrate.ws
                and SubtensorClient._subtensor.substrate.ws.state is WebSocketClientState.OPEN
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

            SubtensorClient._subtensor = subtensor
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
        return miners

    async def set_weights(self, miners, miner_scores: dict[str, float]):
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
                weights[ind] = (1 - TOTAL_BURN_EMISSION) * miner_scores.get(miner.hotkey, 0.0) / total_score
        # if total_score <= 0:
        #     uids[0] = main_burner
        #     weights[0] = 1 - (len(settings.BURNERS) - 1) * BURNER_EMISSION
        #     miner_hotkeys.append(metagraph.hotkeys[main_burner])
        #     for ind, uid in enumerate(other_burners):
        #         uids[ind + 1] = uid
        #         weights[ind + 1] = BURNER_EMISSION
        #         miner_hotkeys.append(metagraph.hotkeys[uid])
        # else:
        #     for ind, miner in enumerate(miners):
        #         uids[ind] = miner.uid
        #         miner_hotkeys.append(metagraph.hotkeys[miner.uid])
        #         if miner.uid == main_burner:
        #             weights[ind] = TOTAL_BURN_EMISSION - (len(settings.BURNERS) - 1) * BURNER_EMISSION
        #         elif miner.uid in other_burners:
        #             weights[ind] = BURNER_EMISSION
        #         else:
        #             weights[ind] = (1 - TOTAL_BURN_EMISSION) * miner_scores.get(miner.hotkey, 0.0) / total_score

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
                
                count += 1
                if count >= 10:
                    await self.update_evm_address_map()
                    count = 0

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

    @classmethod
    def initialize(cls):
        """Initialize the singleton instance asynchronously."""
        instance = cls.get_instance()
        # The warm-up task is already started in __init__
        return instance

    @classmethod
    def get_subtensor(cls):
        """Get the subtensor instance directly."""
        if cls._subtensor is None:
            instance = cls.get_instance()
            instance.set_subtensor()
        return cls._subtensor
