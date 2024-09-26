import logging
import traceback
import asyncio
import bittensor

from core.config import settings
from core.db import get_db
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
        
    def get_subtensor(self):
        return bittensor.subtensor(config=self.config)

    async def check_registered(self, subtensor: bittensor.subtensor):
        bittensor.logging.info('checking miner is registered')
        if not subtensor.is_hotkey_registered(
            netuid=self.netuid,
            hotkey_ss58=self.wallet.get_hotkey().ss58_address,
        ):
            bittensor.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    async def announce(self, subtensor: bittensor.subtensor):
        bittensor.logging.info('Announce miner')
        self.axon.serve(netuid=self.netuid, subtensor=subtensor)

    async def fetch_validators(self, subtensor: bittensor.subtensor):
        metagraph = subtensor.metagraph(netuid=self.netuid)
        neurons = [n for n in metagraph.neurons if (n.stake.tao >= MIN_STAKE)]
        return neurons[:VALIDATORS_LIMIT]

    async def save_validators(self, validators):
        bittensor.logging.info('Sync validators')
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
        subtensor = self.get_subtensor()
        
        await self.check_registered(subtensor)
        await self.announce(subtensor)
        
        validators = await self.fetch_validators(subtensor)
        await self.save_validators(validators)

    async def start(self):
        bittensor.logging.info('Start Miner in background')
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
        bittensor.logging.info('Stop Miner process')
        self.should_exit = True
