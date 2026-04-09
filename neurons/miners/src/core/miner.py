import asyncio
import logging
import traceback
from collections.abc import Sequence
from typing import TYPE_CHECKING

import bittensor
from sqlmodel import Session, select

from core.config import settings
from core.db import engine
from core.utils import _m, get_extra_info
from models.validator import Validator

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor_wallet import bittensor_wallet

logger = logging.getLogger(__name__)

VALIDATORS_LIMIT = 24
RECONNECT_POLL_CYCLE = 2 * 60


class Miner:
    wallet: "bittensor_wallet"
    subtensor: "AsyncSubtensor | None"
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

        self.should_exit = False
        self.bootstrap_complete = False

    async def initialize_subtensor(self):
        self.bootstrap_complete = False
        subtensor = None
        try:
            logger.info(
                _m(
                    "Initializing subtensor",
                    extra=get_extra_info(self.default_extra),
                ),
            )

            await self.close_subtensor()
            if self.should_exit:
                return

            subtensor = await bittensor.async_subtensor(config=self.config).initialize()
            if self.should_exit:
                await subtensor.close()
                return

            self.subtensor = subtensor

            # check registered
            await self.check_registered()
        except Exception as e:
            self.subtensor = None
            if subtensor is not None:
                try:
                    await subtensor.close()
                except Exception as close_error:
                    logger.warning(
                        _m(
                            "Failed to close subtensor cleanly",
                            extra=get_extra_info(
                                {**self.default_extra, "error": str(close_error)}
                            ),
                        ),
                    )
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

    async def close_subtensor(self):
        subtensor = self.subtensor
        self.subtensor = None
        if subtensor is None:
            return

        try:
            await subtensor.close()
        except Exception as e:
            logger.warning(
                _m(
                    "Failed to close subtensor cleanly",
                    extra=get_extra_info({**self.default_extra, "error": str(e)}),
                ),
            )

    async def set_subtensor(self):
        if self.subtensor is not None:
            return

        await self.initialize_subtensor()

    async def check_registered(self):
        try:
            logger.info(
                _m(
                    "[check_registered] checking miner is registered",
                    extra=get_extra_info(self.default_extra),
                ),
            )

            if self.subtensor is None:
                raise RuntimeError("Subtensor is not initialized")

            if not await self.subtensor.is_hotkey_registered(
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
                    "[check_registered] Checking miner registered failed",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "error": str(e),
                        }
                    ),
                ),
            )

    def get_node(self):
        if self.subtensor is None:
            raise RuntimeError("Subtensor is not initialized")
        return self.subtensor.substrate

    async def get_serving_rate_limit(self):
        node = self.get_node()
        result = await node.query("SubtensorModule", "ServingRateLimit", [self.netuid])
        return result.value

    def _axon_info_matches_local_config(self, axon_info) -> bool:
        if axon_info is None or not getattr(axon_info, "is_serving", False):
            return False

        expected_ip = str(self.default_extra["external_ip"])
        expected_port = int(self.default_extra["external_port"])
        actual_ip = str(getattr(axon_info, "ip", ""))

        try:
            actual_port = int(getattr(axon_info, "port", -1))
        except (TypeError, ValueError):
            return False

        return actual_ip == expected_ip and actual_port == expected_port

    async def ensure_axon_registration(self):
        if self.subtensor is None:
            raise RuntimeError("Subtensor is not initialized")

        try:
            neuron = await self.subtensor.get_neuron_for_pubkey_and_subnet(
                hotkey_ss58=self.wallet.get_hotkey().ss58_address,
                netuid=self.netuid,
            )

            if self._axon_info_matches_local_config(getattr(neuron, "axon_info", None)):
                return

            logger.info(
                _m(
                    "[ensure_axon_registration] Announce miner",
                    extra=get_extra_info(self.default_extra),
                ),
            )
            await self.subtensor.serve_axon(netuid=self.netuid, axon=self.axon)
        except Exception as e:
            logger.error(
                _m(
                    "[ensure_axon_registration] Axon registration check failed",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "error": str(e),
                        }
                    ),
                ),
            )
            raise

    async def fetch_validators(self):
        if self.subtensor is None:
            raise RuntimeError("Subtensor is not initialized")

        metagraph_info = await self.subtensor.get_metagraph_info(self.netuid)
        metagraph = await self.subtensor.metagraph(netuid=self.netuid)
        neurons = [
            neuron
            for neuron in metagraph.neurons
            if (
                neuron.stake.tao >= settings.MIN_ALPHA_STAKE
                and metagraph_info.total_stake[neuron.uid] >= settings.MIN_TOTAL_STAKE
            )
        ]
        return neurons

    async def save_validators(self, validators):
        logger.info(
            _m(
                "[save_validators] Sync validators",
                extra=get_extra_info(self.default_extra),
            ),
        )
        validator_hotkeys = [validator.hotkey for validator in validators]
        await asyncio.to_thread(self._save_validators_sync, validator_hotkeys)

    @staticmethod
    def _save_validators_sync(validator_hotkeys: Sequence[str]):
        unique_hotkeys = list(dict.fromkeys(validator_hotkeys))
        if not unique_hotkeys:
            return

        with Session(engine) as session:
            existing_hotkeys = set(
                session.exec(
                    select(Validator.validator_hotkey).where(
                        Validator.validator_hotkey.in_(unique_hotkeys)
                    )
                ).all()
            )
            missing_hotkeys = [
                hotkey for hotkey in unique_hotkeys if hotkey not in existing_hotkeys
            ]

            if not missing_hotkeys:
                return

            session.add_all(
                [Validator(validator_hotkey=hotkey, active=True) for hotkey in missing_hotkeys]
            )
            session.commit()

    async def bootstrap(self):
        await self.ensure_axon_registration()

        validators = await self.fetch_validators()
        await self.save_validators(validators)
        self.bootstrap_complete = True

    async def sync(self):
        try:
            await self.set_subtensor()
            if not self.bootstrap_complete:
                await self.bootstrap()
        except Exception as e:
            logger.error(
                _m(
                    "[sync] Miner sync failed",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "error": str(e),
                        }
                    ),
                ),
            )
            if not self.should_exit:
                await self.initialize_subtensor()

    async def start(self):
        logger.info(
            _m(
                "Start Miner in background",
                extra=get_extra_info(self.default_extra),
            ),
        )
        try:
            while not self.should_exit:
                if not settings.debug.SKIP_SYNC_FLOW:
                    await self.sync()

                await asyncio.sleep(RECONNECT_POLL_CYCLE)
        except KeyboardInterrupt:
            logger.debug("Miner killed by keyboard interrupt.")
            exit()
        except Exception:
            logger.error(traceback.format_exc())

    async def stop(self):
        logger.info(
            _m(
                "Stop Miner process",
                extra=get_extra_info(self.default_extra),
            ),
        )
        self.should_exit = True
        await self.close_subtensor()
