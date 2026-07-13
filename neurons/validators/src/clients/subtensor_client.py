import asyncio
import bittensor
import contextlib
import numpy as np
import time
from typing import Self, TYPE_CHECKING
from bittensor.utils.weight_utils import process_weights_for_netuid
from websockets.protocol import State as WebSocketClientState
from datetime import datetime
import json
import aiohttp

from core.config import settings
from core.utils import _m, get_extra_info, get_logger

from services.redis_service import NORMALIZED_SCORE_CHANNEL, RedisService
from clients.validator_portal_api import ValidatorPortalAPI

if TYPE_CHECKING:
    from bittensor_wallet import bittensor_wallet

logger = get_logger(__name__)

SYNC_CYCLE = 12
SUBTENSOR_BACKOFF_INITIAL = 12
SUBTENSOR_BACKOFF_MAX = 300


@contextlib.contextmanager
def _log_sync_block(name: str, *, extra: dict | None = None):
    """Time a synchronous call that runs inside the asyncio event loop. Emits a single
    log on exit with elapsed ms — doubles as a breadcrumb trail when async HTTP clients
    report timeouts that were actually caused by the loop being frozen here.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            _m(
                f"[sync-block] {name} ({elapsed_ms}ms)",
                extra=get_extra_info({
                    **(extra or {}),
                    "sync_block": name,
                    "elapsed_ms": elapsed_ms,
                }),
            )
        )


def _convert_weights_with_positive_floor(
    uids,
    weights,
) -> tuple[list[int], list[int], list[tuple[int, float]]]:
    """Max-upscale floats to u16 like bittensor's `convert_weights_and_uids_for_emit`,
    but bump u16=0 → u16=1 whenever the source float was strictly positive. SDK-zeroed
    entries (quantile/permit/max-limit exclusions) stay dropped.

    Returns (uids, weights, floored) where `floored` is a list of `(uid, original_weight)`
    tuples for entries whose float weight was positive but rounded to 0 and were lifted to 1.
    """
    uids_l = uids.tolist() if hasattr(uids, "tolist") else list(uids)
    weights_l = weights.tolist() if hasattr(weights, "tolist") else list(weights)
    if not weights_l:
        return [], [], []
    max_w = float(max(weights_l))
    if max_w <= 0:
        return [], [], []
    out_uids: list[int] = []
    out_vals: list[int] = []
    floored: list[tuple[int, float]] = []
    for u, w in zip(uids_l, weights_l):
        wf = float(w)
        if wf <= 0:
            continue
        v = round(wf / max_w * 65535)
        if v == 0:
            v = 1
            floored.append((int(u), wf))
        out_uids.append(int(u))
        out_vals.append(int(v))
    return out_uids, out_vals, floored


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
    def get_instance(cls) -> Self:
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
        if settings.debug.USE_LOCAL_MINER:
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
            subtensor = bittensor.Subtensor(config=self.config)

            # check registered
            self.check_registered(subtensor)

            SubtensorClient._subtensor = subtensor
        except Exception as e:
            logger.error(
                _m(
                    "[Error] failed initializing subtensor",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "error": str(e),
                        }
                    ),
                ),
                exc_info=True,
            )

    def set_subtensor(self):
        if (
            SubtensorClient._subtensor
            and SubtensorClient._subtensor.substrate
            and SubtensorClient._subtensor.substrate.ws
            and not SubtensorClient._subtensor.substrate.ws.close_code
        ):
            return

        with _log_sync_block("set_subtensor.initialize", extra=self.default_extra):
            self.initialize_subtensor()

    def check_registered(self, subtensor: bittensor.Subtensor):
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
        with _log_sync_block("get_metagraph", extra=self.default_extra):
            return self.subtensor.metagraph(netuid=self.netuid)

    def get_node(self):
        # return SubstrateInterface(url=self.config.subtensor.chain_endpoint)
        return self.subtensor.substrate

    def get_current_block(self):
        with _log_sync_block("get_current_block"):
            node = self.get_node()
            return node.query("System", "Number", []).value

    def get_weights_rate_limit(self):
        with _log_sync_block("get_weights_rate_limit"):
            node = self.get_node()
            return node.query("SubtensorModule", "WeightsSetRateLimit", [self.netuid]).value

    def get_last_mechansim_step_block(self):
        with _log_sync_block("get_last_mechansim_step_block"):
            node = self.get_node()
            return node.query("SubtensorModule", "LastMechansimStepBlock", [self.netuid]).value

    def get_uid_for_hotkey(self, hotkey):
        metagraph = self.get_metagraph()
        return metagraph.hotkeys.index(hotkey)

    def get_evm_address_for_hotkey(self, hotkey):
        return self.hotkey_to_evm_address.get(hotkey, None)

    def sync_evm_address_maps(self):
        with _log_sync_block("sync_evm_address_maps", extra=self.default_extra):
            node = self.get_node()
            associated_evms = node.query_map(module="SubtensorModule", storage_function="AssociatedEvmAddress", params=[self.netuid])
            for uid, evm_address in associated_evms:
                # async-substrate-interface 2.x yields decoded records: ("0x…", block_number)
                evm_address_hex = evm_address[0]
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

    async def fetch_miners(self):
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

            miners = metagraph.neurons

            # Get miners that have opted in from portal BE
            miners_with_opt_in_status = await ValidatorPortalAPI.get_opted_in_miners()

            logger.info(
                _m(
                    f"[fetch_miners] Found {len(miners_with_opt_in_status)} opted-in miners from portal",
                    extra=get_extra_info(self.default_extra),
                ),
            )
            # Update miners in `miners` list based on miners_with_opt_in_status.
            hotkey_to_opt_in = {
                opt_in["miner_hotkey"]: opt_in for opt_in in miners_with_opt_in_status if opt_in.get("miner_hotkey", None)
            }
            for miner in miners:
                opt_in = hotkey_to_opt_in.get(miner.hotkey)
                if opt_in is not None:
                    miner.axon_info.ip = opt_in.get("central_miner_ip")
                    miner.axon_info.port = opt_in.get("central_miner_port")
                    
            miners = [
                neuron
                for neuron in metagraph.neurons
                if neuron.axon_info.is_serving or neuron.uid in (settings.BURNERS + settings.NEW_BURNERS)
            ]

        logger.info(
            _m(
                f"[fetch_miners] Found {len(miners)} miners",
                extra=get_extra_info(self.default_extra),
            ),
        )

        self.miners = miners

    async def get_miner(self, hotkey: str) -> bittensor.NeuronInfo:
        miners = await self.get_miners()

        neurons = [n for n in miners if n.hotkey == hotkey]
        if not neurons:
            raise ValueError(f"Miner with {hotkey=} not present in this subnetwork")
        return neurons[0]

    async def get_miners(self) -> list[bittensor.NeuronInfo]:
        if not self.miners:
            await self.fetch_miners()
        return self.miners
    
    async def send_weights_to_lium(self, payload: dict):
        """Send weights to lium."""
        try:
            keypair = settings.get_bittensor_wallet().get_hotkey()
            validator_hotkey = keypair.ss58_address
            api_url = f"{settings.COMPUTE_REST_API_URL}/validator/{validator_hotkey}/latest-set-weights"
            blob_for_signing = json.dumps(payload, sort_keys=True)
            signature = f"0x{keypair.sign(blob_for_signing).hex()}"
            logger.info(
                _m(
                    "[send_weights_to_lium] Sending weights to lium",
                    extra=get_extra_info({"api_url": api_url}),
                ),
            )
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                url = f"{api_url}"
                await session.post(url, json=payload, headers={"Authorization": f"{signature}"})
        except Exception as e:
            logger.error(_m("[send_weights_to_lium] Failed to post latest-set-weights", extra=get_extra_info({"error": str(e)})))

    async def set_weights(self, miner_scores: dict[str, float]):
        """Set weights using accumulated scores with burning already applied.

        The miner_scores dict already includes burning logic from calculate_final_weights
        called per cycle. This method just normalizes and sends to chain.
        """
        miners = await self.get_miners()
        logger.info(
            _m(
                "[set_weights] accumulated scores",
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

        # Build uids and weights arrays
        uids = np.zeros(len(miners), dtype=np.int64)
        weights = np.zeros(len(miners), dtype=np.float32)
        miner_hotkeys = []

        for ind, miner in enumerate(miners):
            uids[ind] = miner.uid
            weights[ind] = miner_scores.get(miner.hotkey, 0.0)
            miner_hotkeys.append(miner.hotkey)

        logger.debug(
            _m(
                f"[set_weights] uids: {uids} weights: {weights}",
                extra=get_extra_info(self.default_extra),
            ),
        )

        # Publish normalized scores
        normalized_scores = [
            {"uid": int(uid), "weight": float(weight), "miner_hotkey": miner_hotkey}
            for uid, weight, miner_hotkey in zip(uids, weights, miner_hotkeys)
        ]
        message = {"normalized_scores": normalized_scores}
        await self.redis_service.publish(NORMALIZED_SCORE_CHANNEL, message)

        # Process weights for blockchain
        metagraph = self.get_metagraph()
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

        uint_uids, uint_weights, floored = _convert_weights_with_positive_floor(
            processed_uids, processed_weights
        )

        # Resolve floored uids back to hotkeys so the log row is human-debuggable.
        uid_to_hotkey = dict(zip([int(u) for u in uids], miner_hotkeys))
        floored_entries = [
            {"uid": uid, "hotkey": uid_to_hotkey.get(uid), "original_score": original_score}
            for uid, original_score in floored
        ]

        logger.info(
            _m(
                f"[set_weights] uint_uids: {uint_uids} uint_weights: {uint_weights}",
                extra=get_extra_info({
                    **self.default_extra,
                    "version_key": self.version_key,
                    "floored_from_zero_count": len(floored),
                    "floored_from_zero": floored_entries,
                }),
            ),
        )

        try:
            current_block = self.get_current_block()
        except Exception:
            current_block = None

        # Send weights to lium
        payload = {
            "netuid": int(self.netuid),
            "uids": [int(u) for u in list(uint_uids)],
            "weights": [int(w) for w in list(uint_weights)],
            "version_key": int(self.version_key),
            "wait_for_finalization": False,
            "wait_for_inclusion": False,
            "current_block": current_block,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        await self.send_weights_to_lium(payload)

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
        backoff = SUBTENSOR_BACKOFF_INITIAL
        while True:
            try:
                self.set_subtensor()

                if SubtensorClient._subtensor is None:
                    raise RuntimeError("subtensor is not initialized")

                if count == 0:
                    await self.fetch_miners()
                    self.sync_evm_address_maps()

                count += 1
                if count > 10:
                    await self.fetch_miners()
                    self.sync_evm_address_maps()
                    count = 1

                backoff = SUBTENSOR_BACKOFF_INITIAL
                await asyncio.sleep(SYNC_CYCLE)
            except Exception as e:
                logger.error(
                    _m(
                        "[_warm_up_subtensor] Failed to connect into subtensor",
                        extra=get_extra_info({
                            **self.default_extra,
                            "error": str(e),
                            "backoff": backoff,
                        }),
                    ),
                    exc_info=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, SUBTENSOR_BACKOFF_MAX)

    @classmethod
    async def initialize(cls) -> Self:
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
    def get_subtensor(cls) -> bittensor.Subtensor:
        """Get the subtensor instance directly."""
        if cls._subtensor is None:
            instance = cls.get_instance()
            instance.set_subtensor()
        return cls._subtensor

    def get_alpha_rate(self) -> float:
        # `subtensor.get_subnet_price()` reads the `Swap.AlphaSqrtPrice` storage item,
        # which the finney runtime removed when the `Swap` pallet was migrated to a new
        # AMM — querying it now raises `Storage function "Swap.AlphaSqrtPrice" not found`.
        # `subtensor.subnet()` is resilient: it attempts `get_subnet_price()` and, on that
        # failure, falls back to the `tao_in / alpha_in` reserve ratio, yielding the same
        # alpha price in TAO. This keeps the query working without a bittensor SDK upgrade.
        subnet = self.subtensor.subnet(netuid=self.netuid)
        if subnet is None or subnet.price is None:
            raise RuntimeError(f"No subnet price available for netuid {self.netuid}")
        return subnet.price.tao