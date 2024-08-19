import logging
import threading
import traceback
import time

import bittensor
from core.config import settings

logger = logging.getLogger(__name__)

class Validator():
    wallet: bittensor.wallet
    subtensor: bittensor.subtensor
    netuid: int
    
    def __init__(self):
        config = settings.get_bittensor_config()

        self.wallet = settings.get_bittensor_wallet()
        self.subtensor = bittensor.subtensor(config=config)
        self.netuid = settings.BITTENSOR_NETUID
        
        self.check_registered()
        
        self.should_exit = False
        self.is_running = False
        
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
            
    def fetch_minors(self):
        metagraph = self.subtensor.metagraph(netuid=self.netuid)
        miners = [neuron for neuron in metagraph.neurons if neuron.axon_info.is_serving]
        
        return miners
    
    def set_scores(self):
        pass
    
    def set_weights(self):
        pass
    
    def sync(self):
        # fetch miners
        miners = self.fetch_minors()
        
        # run jobs
        
        # set scores
        
        # set rates
        
        pass
        
    def run(self):
        try:
            while not self.should_exit:
                self.sync()
                
                # sync every 10 mins
                time.sleep(10 * 60)
                
        except KeyboardInterrupt:
            logger.debug('Miner killed by keyboard interrupt.')
            exit()
        except Exception as e:
            logger.error(traceback.format_exc())
            
    def run_in_background_thread(self):
        if not self.is_running:
            bittensor.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bittensor.logging.debug("Started")