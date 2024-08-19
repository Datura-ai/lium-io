import logging
import traceback
import asyncio
import bittensor

from core.config import settings
from core.db import get_db
from daos.validator import ValidatorDao, Validator

logger = logging.getLogger(__name__)

MIN_STAKE = 1000
VALIDATORS_LIMIT = 24

class Miner:
    wallet: bittensor.wallet
    subtensor: bittensor.subtensor
    netuid: int

    def __init__(self):
        config = settings.get_bittensor_config()
        self.wallet = settings.get_bittensor_wallet()
        self.subtensor = bittensor.subtensor(config=config)
        self.netuid = settings.BITTENSOR_NETUID

        self.axon = bittensor.axon(
            wallet=self.wallet,
            external_port=settings.PORT,
            external_ip=settings.IP_ADDRESS,
            port=settings.PORT,
            ip=settings.IP_ADDRESS,
        )
        
        self.should_exit = False
        self.session = next(get_db())
        self.validator_dao = ValidatorDao(session=self.session)

    async def check_registered(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.netuid,
            hotkey_ss58=self.wallet.get_hotkey().ss58_address,
        ):
            bittensor.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    async def announce(self):
        await self.check_registered()
        self.axon.serve(netuid=self.netuid, subtensor=self.subtensor)

    async def fetch_validators(self):
        metagraph = self.subtensor.metagraph(netuid=self.netuid)
        neurons = [n for n in metagraph.neurons if (n.stake.tao >= MIN_STAKE)]
        return neurons[:VALIDATORS_LIMIT]

    async def save_validators(self):
        validators = await self.fetch_validators()
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
        await self.announce()
        await self.save_validators()

    async def start(self):
        try:
            while not self.should_exit:
                await self.sync()
                await asyncio.sleep(30)
        except KeyboardInterrupt:
            logger.debug('Miner killed by keyboard interrupt.')
            exit()
        except Exception as e:
            logger.error(traceback.format_exc())

    async def stop(self):
        self.should_exit = True
