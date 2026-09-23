import asyncio
import logging
import traceback
from collections.abc import Sequence
from typing import TYPE_CHECKING

import bittensor
from datura.chain import (
    ChainConnection,
    ChainEndpoint,
    EndpointCursor,
    EndpointSource,
    is_chain_error,
)
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

        self.axon = bittensor.Axon(
            wallet=self.wallet,
            external_port=settings.EXTERNAL_PORT,
            external_ip=settings.EXTERNAL_IP_ADDRESS,
            port=settings.INTERNAL_PORT,
            ip=settings.EXTERNAL_IP_ADDRESS,
        )
        self.subtensor = None
        self._endpoint_cursor = EndpointCursor(
            settings.get_chain_endpoints(),
            retry_after_seconds=settings.BITTENSOR_CHAIN_ENDPOINT_RETRY_AFTER_SECONDS,
        )

        self.should_exit = False
        self.bootstrap_complete = False

    def _log_endpoint_switched(
        self, previous: ChainEndpoint, current: ChainEndpoint, reason: str, error: Exception
    ) -> None:
        logger.warning(
            _m(
                f"Subtensor endpoint switched from={previous.value} to={current.value}",
                extra=get_extra_info(
                    {
                        **self.default_extra,
                        "from": previous.value,
                        "to": current.value,
                        "from_source": previous.source,
                        "to_source": current.source,
                        "reason": reason,
                        "error": str(error),
                    }
                ),
            ),
        )

    async def _connect_subtensor(self) -> ChainConnection[bittensor.AsyncSubtensor]:
        """Dial the current entry of the ordered endpoint list; when it refuses, move to the next
        one (`Subtensor endpoint switched from=… to=…`) until one answers, so a proxy outage never
        leaves the central miner without a chain client. Providers set no endpoint and dial the
        network name as before. Raises the last error when every entry failed. Returns the client
        and which setting chose the endpoint."""
        cursor = self._endpoint_cursor
        last_error: Exception | None = None
        for _attempt in range(len(cursor.candidates)):
            endpoint = cursor.current
            try:
                subtensor = await bittensor.AsyncSubtensor(
                    network=endpoint.value, config=self.config
                ).initialize()
            except Exception as e:
                last_error = e
                if len(cursor.candidates) == 1:
                    raise
                previous, current = cursor.advance()
                self._log_endpoint_switched(previous, current, "connect failed", e)
                continue
            return ChainConnection(subtensor, cursor.source_label())
        assert last_error is not None
        raise last_error

    async def _switch_endpoint_after_read_failure(self, error: Exception) -> None:
        """A chain read on the connected endpoint failed: close the client and move the cursor
        to the next entry, so the redial that follows skips the failing one. A database or
        other local error leaves the cursor on the healthy proxy."""
        if not is_chain_error(error):
            return
        if self.subtensor is None or len(self._endpoint_cursor.candidates) == 1:
            return
        await self.close_subtensor()
        previous, current = self._endpoint_cursor.advance()
        self._log_endpoint_switched(previous, current, "read failed", error)

    async def _return_to_first_endpoint(self) -> None:
        """A sync cycle starts on the first entry that is not resting: after a cycle ran on a
        fallback node, once the failed endpoint's retry window is over, the client is closed and
        the next dial tries it again (the proxy may be back). Inside the window the fallback client
        stays, so a dead proxy is not redialled every cycle."""
        if not self._endpoint_cursor.move_to_first_ready_endpoint():
            return
        await self.close_subtensor()

    def _log_subtensor_connected(
        self, subtensor: bittensor.AsyncSubtensor, endpoint_source: EndpointSource
    ) -> None:
        logger.info(
            _m(
                "Subtensor connected",
                extra=get_extra_info(
                    {
                        **self.default_extra,
                        "chain_endpoint": subtensor.chain_endpoint,
                        "network": subtensor.network,
                        "endpoint_source": endpoint_source,
                    }
                ),
            ),
        )

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

            subtensor, endpoint_source = await self._connect_subtensor()
            if self.should_exit:
                await subtensor.close()
                return

            self.subtensor = subtensor
            self._log_subtensor_connected(subtensor, endpoint_source)

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
        if settings.CENTRAL_MODE:
            logger.info(
                _m(
                    "[check_registered] Skipping registration check (CENTRAL_MODE)",
                    extra=get_extra_info(self.default_extra),
                ),
            )
            return

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
            # every cycle, so a cycle that failed on a fallback node (a database error in
            # save_validators) still returns once the first endpoint's retry window is over
            await self._return_to_first_endpoint()
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
            if not self.should_exit and is_chain_error(e):
                # a chain read on the connected endpoint failed: the redial below dials the next one
                await self._switch_endpoint_after_read_failure(e)
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
