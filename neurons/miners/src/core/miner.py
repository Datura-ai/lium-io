from typing import TYPE_CHECKING
import logging
import traceback
import asyncio
import bittensor
from websockets.protocol import State as WebSocketClientState

from core.config import settings
from core.db import get_db
from core.utils import _m, get_extra_info
from daos.validator import ValidatorDao, Validator

if TYPE_CHECKING:
    from bittensor_wallet import bittensor_wallet

logger = logging.getLogger(__name__)

VALIDATORS_LIMIT = 24
SYNC_CYCLE = 2 * 60


class Miner:
    wallet: "bittensor_wallet"
    subtensor: bittensor.subtensor
    netuid: int

    def __init__(self):
        self.config = settings.get_bittensor_config()
        self.wallet = settings.get_bittensor_wallet()
        self.netuid = settings.BITTENSOR_NETUID

        self.default_extra = {
            "external_port": settings.EXTERNAL_PORT,
            "external_ip": settings.EXTERNAL_IP_ADDRESS,
        }

        self.axon = bittensor.axon(
            wallet=self.wallet,
            external_port=settings.EXTERNAL_PORT,
            external_ip=settings.EXTERNAL_IP_ADDRESS,
            port=settings.INTERNAL_PORT,
            ip=settings.EXTERNAL_IP_ADDRESS,
        )
        self.subtensor = None
        self.initialize_subtensor()

        self.should_exit = False
        self.session = next(get_db())
        self.validator_dao = ValidatorDao(session=self.session)
        self.last_announced_block = 0

    def initialize_subtensor(self):
        try:
            logger.info(
                _m(
                    "Initializing subtensor",
                    extra=get_extra_info(self.default_extra),
                ),
            )

            self.subtensor = bittensor.subtensor(config=self.config)

            # check registered
            self.check_registered()
        except Exception as e:
            logger.info(
                _m(
                    "[Error] failed initializing subtensor",
                    extra=get_extra_info({
                        ** self.default_extra,
                        "error": str(e),
                    }),
                ),
            )

    def set_subtensor(self):
        if (
            self.subtensor
            and self.subtensor.substrate
            and self.subtensor.substrate.ws
            and not self.subtensor.substrate.ws.close_code
        ):
            return

        self.initialize_subtensor()

    def check_registered(self):
        try:
            logger.info(
                _m(
                    '[check_registered] checking miner is registered',
                    extra=get_extra_info(self.default_extra),
                ),
            )

            if not self.subtensor.is_hotkey_registered(
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
        except Exception as e:
            logger.error(
                _m(
                    '[check_registered] Checking miner registered failed',
                    extra=get_extra_info({
                        **self.default_extra,
                        "error": str(e)
                    }),
                ),
            )

    def get_node(self):
        # return SubstrateInterface(url=self.config.subtensor.chain_endpoint)
        return self.subtensor.substrate

    def get_current_block(self):
        node = self.get_node()
        return node.query("System", "Number", []).value

    def get_tempo(self):
        return self.subtensor.tempo(self.netuid)

    def get_serving_rate_limit(self):
        node = self.get_node()
        return node.query("SubtensorModule", "ServingRateLimit", [self.netuid]).value

    def announce(self):
        try:
            current_block = self.get_current_block()
            tempo = self.get_tempo()

            if current_block - self.last_announced_block >= tempo:
                self.last_announced_block = current_block

                logger.info(
                    _m(
                        '[announce] Announce miner',
                        extra=get_extra_info(self.default_extra),
                    ),
                )
                self.axon.serve(netuid=self.netuid, subtensor=self.subtensor)
        except Exception as e:
            logger.error(
                _m(
                    '[announce] Announcing miner error',
                    extra=get_extra_info({
                        **self.default_extra,
                        "error": str(e)
                    }),
                ),
            )

    async def fetch_validators(self):
        a = self.subtensor.get_metagraph_info(self.netuid)
        metagraph = self.subtensor.metagraph(netuid=self.netuid)
        neurons = [
            n for n in metagraph.neurons
            if (n.stake.tao >= settings.MIN_ALPHA_STAKE and a.total_stake[n.uid] >= settings.MIN_TOTAL_STAKE)
        ]
        return neurons

    async def save_validators(self, validators):
        logger.info(
            _m(
                '[save_validators] Sync validators',
                extra=get_extra_info(self.default_extra),
            ),
        )
        for v in validators:
            existing = self.validator_dao.get_validator_by_hotkey(v.hotkey)
            if not existing:
                self.validator_dao.save(
                    Validator(
                        validator_hotkey=v.hotkey,
                        active=True
                    )
                )

    async def sync(self):
        try:
            self.set_subtensor()

            self.announce()

            validators = await self.fetch_validators()
            await self.save_validators(validators)
        except Exception as e:
            logger.error(
                _m(
                    '[sync] Miner sync failed',
                    extra=get_extra_info({
                        **self.default_extra,
                        "error": str(e)
                    }),
                ),
            )
            self.initialize_subtensor()

    async def start(self):
        logger.info(
            _m(
                'Start Miner in background',
                extra=get_extra_info(self.default_extra),
            ),
        )
        try:
            while not self.should_exit:
                await self.sync()

                # sync every 2 mins
                await asyncio.sleep(SYNC_CYCLE)
        except KeyboardInterrupt:
            logger.debug('Miner killed by keyboard interrupt.')
            exit()
        except Exception as e:
            logger.error(traceback.format_exc())

    async def stop(self):
        logger.info(
            _m(
                'Stop Miner process',
                extra=get_extra_info(self.default_extra),
            ),
        )
        self.should_exit = True
