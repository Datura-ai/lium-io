import logging
import traceback
import asyncio
import bittensor

from core.config import settings
from core.db import get_db
from core.utils import _m, get_extra_info
from daos.validator import ValidatorDao, Validator

logger = logging.getLogger(__name__)

MIN_STAKE = 10
VALIDATORS_LIMIT = 24
SYNC_CYCLE = 2 * 60


class Miner:
    wallet: bittensor.wallet
    subtensor: bittensor.subtensor
    netuid: int

    def __init__(self):
        self.config = settings.get_bittensor_config()
        self.wallet = settings.get_bittensor_wallet()
        self.netuid = settings.BITTENSOR_NETUID

        self.axon = bittensor.axon(
            wallet=self.wallet,
            external_port=settings.EXTERNAL_PORT,
            external_ip=settings.EXTERNAL_IP_ADDRESS,
            port=settings.INTERNAL_PORT,
            ip=settings.EXTERNAL_IP_ADDRESS,
        )

        self.should_exit = False
        self.session = next(get_db())
        self.validator_dao = ValidatorDao(session=self.session)
        self.last_announced_block = 0

        self.default_extra = {
            "external_port": settings.EXTERNAL_PORT,
            "external_ip": settings.EXTERNAL_IP_ADDRESS,
        }

    def get_subtensor(self):
        logger.info(
            _m(
                'Getting subtensor',
                extra=get_extra_info(self.default_extra),
            ),
        )
        return bittensor.subtensor(config=self.config)

    def get_metagraph(self, subtensor: bittensor.subtensor):
        return subtensor.metagraph(netuid=self.netuid)

    def get_node(self, subtensor: bittensor.subtensor):
        return subtensor.substrate

    def get_my_uid(self, subtensor: bittensor.subtensor):
        metagraph = self.get_metagraph(subtensor)
        return metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

    def get_current_block(self, subtensor: bittensor.subtensor):
        node = self.get_node(subtensor)
        return node.query("System", "Number", []).value

    def get_tempo(self, subtensor: bittensor.subtensor):
        return subtensor.tempo(self.netuid)

    async def check_registered(self, subtensor: bittensor.subtensor):
        try:
            logger.info(
                _m(
                    'checking miner is registered',
                    extra=get_extra_info(self.default_extra),
                ),
            )
            if not subtensor.is_hotkey_registered(
                netuid=self.netuid,
                hotkey_ss58=self.wallet.get_hotkey().ss58_address,
            ):
                logger.error(
                    _m(
                        f"Wallet: {self.wallet} is not registered on netuid {self.netuid}.",
                        extra=get_extra_info(self.default_extra),
                    ),
                )
                exit()
        except Exception as e:
            logger.error(
                _m(
                    'Checking validator registered failed',
                    extra=get_extra_info({
                        **self.default_extra,
                        "error": str(e)
                    }),
                ),
            )

    async def announce(self, subtensor: bittensor.subtensor):
        try:
            current_block = self.get_current_block(subtensor)
            tempo = self.get_tempo(subtensor)

            if current_block - self.last_announced_block >= tempo:
                self.last_announced_block = current_block

                logger.info(
                    _m(
                        'Announce miner',
                        extra=get_extra_info(self.default_extra),
                    ),
                )
                self.axon.serve(netuid=self.netuid, subtensor=subtensor)
        except Exception as e:
            logger.error(
                _m(
                    'Annoucing miner error',
                    extra=get_extra_info({
                        **self.default_extra,
                        "error": str(e)
                    }),
                ),
            )

    async def fetch_validators(self, subtensor: bittensor.subtensor):
        metagraph = subtensor.metagraph(netuid=self.netuid)
        neurons = [n for n in metagraph.neurons if (n.stake.tao >= MIN_STAKE)]
        return neurons[:VALIDATORS_LIMIT]

    async def save_validators(self, validators):
        logger.info(
            _m(
                'Sync validators',
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
            subtensor = self.get_subtensor()

            await self.check_registered(subtensor)
            await self.announce(subtensor)

            validators = await self.fetch_validators(subtensor)
            await self.save_validators(validators)
        except Exception as e:
            logger.error(
                _m(
                    'Miner sync failed',
                    extra=get_extra_info({
                        **self.default_extra,
                        "error": str(e)
                    }),
                ),
            )

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
