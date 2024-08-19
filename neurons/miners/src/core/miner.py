import logging
import threading
import traceback
import time

import bittensor
from core.config import settings
from core.db import get_db

from daos.validator import ValidatorDao, Validator

logger = logging.getLogger(__name__)

MIN_STAKE = 1000
VALIDATORS_LIMIT = 24

class Miner():
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
        
        self.check_registered()
        self.announce()
        
        self.should_exit = False
        self.is_running = False
        
        session = next(get_db())
        self.validator_dao = ValidatorDao(session=session)
        
    def __enter__(self):
        self.run_in_background_thread()
        
    def __exit__(self):
        if self.is_running:
            logger.debug("Stopping miner in background thread.")
            self.should_exit = True
            if self.thread is not None:
                self.thread.join(5)
            self.is_running = False
            logger.debug("Stopped")
        
    def check_registered(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.netuid,
            hotkey_ss58=self.wallet.get_hotkey().ss58_address,
        ):
            bittensor.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()
    
    def announce(self):
        self.axon.serve(netuid=self.netuid, subtensor=self.subtensor)
        # # Start  starts the miner's axon, making it active on the network.
        # self.axon.start()
        
    def fetch_validators(self):
        metagraph = self.subtensor.metagraph(netuid=self.netuid)
        neurons = [n for n in metagraph.neurons if (n.stake.tao >= MIN_STAKE)]
        
        return neurons[:VALIDATORS_LIMIT]
    
    def save_validators(self):
        validators = self.fetch_validators()
        for v in validators:
            existing = self.validator_dao.get_validator_by_hotkey(v.hotkey)
            if not existing:
                self.validator_dao.save(
                    Validator(
                        validator_hotkey=v.hotkey,
                        active=True
                    )
                )
    
    def sync(self):
        # announce
        self.announce()
        
        # get validators and store it in database
        self.save_validators()

            
    def run(self):
        try:
            while not self.should_exit:
                self.sync()
                
                time.sleep(2 * 60)
                
        except KeyboardInterrupt:
            logger.debug('Miner killed by keyboard interrupt.')
            exit()
        except Exception as e:
            logger.error(traceback.format_exc())
        
    def run_in_background_thread(self):
        if not self.is_running:
            logger.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
