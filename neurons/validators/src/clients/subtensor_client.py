import asyncio
import contextlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Self

import aiohttp
import bittensor
import numpy as np
from bittensor.utils.weight_utils import process_weights_for_netuid
from pydantic import BaseModel, Field, ValidationError
from websockets.protocol import State as WebSocketClientState

from clients.validator_portal_api import OptedInMiner, ValidatorPortalAPI
from core.config import settings
from core.utils import _m, get_extra_info, get_logger
from services.redis_service import NORMALIZED_SCORE_CHANNEL, RedisService

if TYPE_CHECKING:
    from bittensor_wallet import bittensor_wallet

logger = get_logger(__name__)

SYNC_CYCLE = 12
SUBTENSOR_BACKOFF_INITIAL = 12
SUBTENSOR_BACKOFF_MAX = 300
PORTAL_MINERS_CACHE_KEY: str = "validator:portal:opted_in_miners:last_good"
PORTAL_MINERS_CACHE_ALERT_SECONDS: int = 60 * 60


class OptedInMinerSnapshot(BaseModel):
    cached_at: float = Field(ge=0)
    miners: list[OptedInMiner]


@dataclass(frozen=True)
class MissingScoredHotkeys:
    active_in_last_cycle: list[str]
    inactive_in_last_cycle: list[str]


class ProviderPortalDataUnavailable(RuntimeError):
    """No live or Redis-cached provider snapshot is available."""


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


def _select_floor_candidates(
    uint_uids,
    miners,
    active_hotkeys: set[str],
    burn_uids: set[int],
) -> list[tuple[int, str]]:
    """Pick the hotkeys that had a live executor last cycle but are missing from the
    u16 vector. Burn uids are never candidates — they fund the floor, they don't take it.

    Returns a list of `(uid, hotkey)` tuples sorted by uid so the transfer is deterministic.
    """
    present = {int(u) for u in uint_uids}
    return sorted(
        (int(miner.uid), miner.hotkey)
        for miner in miners
        if miner.hotkey in active_hotkeys
        and int(miner.uid) not in present
        and int(miner.uid) not in burn_uids
    )


def _largest_burn_index(uids: list[int], weights: list[int], burn_uids: set[int]) -> int | None:
    """Index of the burn entry currently carrying the most u16 mass, or None when the vector has no burner."""
    burn_indexes = [ind for ind, uid in enumerate(uids) if uid in burn_uids]
    if not burn_indexes:
        return None
    return max(burn_indexes, key=lambda ind: weights[ind])


def _apply_eligibility_floor(
    uint_uids,
    uint_weights,
    miners,
    active_hotkeys: set[str],
    burn_uids: set[int],
) -> tuple[list[int], list[int], list[str]]:
    """Give one u16 unit to every live hotkey the vector dropped, taken from the largest
    burn entry so no earning miner is diluted. Builds new lists, never mutates the inputs.

    Returns (uids, weights, floored_hotkeys).
    """
    out_uids = [int(u) for u in uint_uids]
    out_weights = [int(w) for w in uint_weights]
    floored_hotkeys: list[str] = []
    for uid, hotkey in _select_floor_candidates(out_uids, miners, active_hotkeys, burn_uids):
        donor = _largest_burn_index(out_uids, out_weights, burn_uids)
        if donor is None or out_weights[donor] <= 1:
            break
        out_weights[donor] -= 1
        out_uids.append(uid)
        out_weights.append(1)
        floored_hotkeys.append(hotkey)
    return out_uids, out_weights, floored_hotkeys


def _classify_missing_scored_hotkeys(
    miner_scores: dict[str, float],
    snapshot_miner_hotkeys: set[str],
    registered_hotkeys: set[str],
    active_hotkeys: set[str],
) -> MissingScoredHotkeys:
    missing_hotkeys = sorted(
        hotkey
        for hotkey, score in miner_scores.items()
        if score > 0
        and hotkey in registered_hotkeys
        and hotkey not in snapshot_miner_hotkeys
    )
    active_missing_hotkeys = [
        hotkey for hotkey in missing_hotkeys if hotkey in active_hotkeys
    ]
    inactive_missing_hotkeys = [
        hotkey for hotkey in missing_hotkeys if hotkey not in active_hotkeys
    ]
    return MissingScoredHotkeys(
        active_in_last_cycle=active_missing_hotkeys,
        inactive_in_last_cycle=inactive_missing_hotkeys,
    )


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
        self._has_alerted_for_stale_portal_snapshot = False

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

    async def _load_cached_opted_in_miners_snapshot(
        self,
    ) -> OptedInMinerSnapshot | None:
        try:
            cached_json = await self.redis_service.get(PORTAL_MINERS_CACHE_KEY)
        except Exception as exc:
            logger.warning(
                _m(
                    "[fetch_miners] failed to read provider snapshot cache",
                    extra=get_extra_info(
                        {**self.default_extra, "error": str(exc)}
                    ),
                )
            )
            return None

        if cached_json is None:
            return None

        try:
            return OptedInMinerSnapshot.model_validate_json(cached_json)
        except (ValidationError, TypeError) as exc:
            logger.error(
                _m(
                    "[fetch_miners] invalid provider snapshot cache",
                    extra=get_extra_info(
                        {**self.default_extra, "error": str(exc)}
                    ),
                )
            )
            return None

    async def _store_opted_in_miners_snapshot(
        self,
        miners: list[OptedInMiner],
    ) -> None:
        snapshot = OptedInMinerSnapshot(cached_at=time.time(), miners=miners)
        try:
            await self.redis_service.set(
                PORTAL_MINERS_CACHE_KEY,
                snapshot.model_dump_json(),
            )
        except Exception as exc:
            logger.warning(
                _m(
                    "[fetch_miners] failed to cache provider snapshot",
                    extra=get_extra_info(
                        {**self.default_extra, "error": str(exc)}
                    ),
                )
            )

    def _report_cached_opted_in_miners_usage(
        self,
        snapshot: OptedInMinerSnapshot,
    ) -> None:
        cache_age_seconds = int(max(0, time.time() - snapshot.cached_at))
        log_context = {
            **self.default_extra,
            "provider_count": len(snapshot.miners),
            "cache_age_seconds": cache_age_seconds,
            "snapshot_source": "redis_cache",
        }
        logger.warning(
            _m(
                "[fetch_miners] using cached provider snapshot",
                extra=get_extra_info(log_context),
            )
        )
        if (
            cache_age_seconds >= PORTAL_MINERS_CACHE_ALERT_SECONDS
            and not self._has_alerted_for_stale_portal_snapshot
        ):
            logger.critical(
                _m(
                    "[fetch_miners] provider snapshot cache exceeded alert threshold",
                    extra=get_extra_info(
                        {
                            **log_context,
                            "alert_threshold_seconds": PORTAL_MINERS_CACHE_ALERT_SECONDS,
                        }
                    ),
                )
            )
            self._has_alerted_for_stale_portal_snapshot = True

    async def _resolve_opted_in_miners(self) -> list[OptedInMiner]:
        live_miners = await ValidatorPortalAPI.get_opted_in_miners()
        if live_miners is not None:
            if not live_miners:
                previous_snapshot = await self._load_cached_opted_in_miners_snapshot()
                if previous_snapshot is not None and previous_snapshot.miners:
                    logger.warning(
                        _m(
                            "[fetch_miners] opted-in provider count dropped to zero",
                            extra=get_extra_info(
                                {
                                    **self.default_extra,
                                    "previous_provider_count": len(previous_snapshot.miners),
                                    "provider_count": 0,
                                }
                            ),
                        )
                    )
            await self._store_opted_in_miners_snapshot(live_miners)
            self._has_alerted_for_stale_portal_snapshot = False
            logger.info(
                _m(
                    "[fetch_miners] resolved opted-in providers",
                    extra=get_extra_info(
                        {
                            **self.default_extra,
                            "provider_count": len(live_miners),
                            "snapshot_source": "portal",
                        }
                    ),
                )
            )
            return live_miners

        cached_snapshot = await self._load_cached_opted_in_miners_snapshot()
        if cached_snapshot is None:
            raise ProviderPortalDataUnavailable(
                "provider portal unavailable and no cached snapshot exists"
            )
        self._report_cached_opted_in_miners_usage(cached_snapshot)
        return cached_snapshot.miners

    def _build_miners_from_opted_in_snapshot(
        self,
        opted_in_miners: Sequence[OptedInMiner],
    ) -> list[bittensor.NeuronInfo]:
        metagraph = self.get_metagraph()
        opted_in_by_hotkey = {
            opted_in.miner_hotkey: opted_in for opted_in in opted_in_miners
        }
        for neuron in metagraph.neurons:
            opted_in = opted_in_by_hotkey.get(neuron.hotkey)
            if opted_in is not None:
                neuron.axon_info.ip = opted_in.central_miner_ip
                neuron.axon_info.port = opted_in.central_miner_port

        burner_uids = {*settings.BURNERS, *settings.NEW_BURNERS}
        return [
            neuron
            for neuron in metagraph.neurons
            if neuron.axon_info.is_serving or neuron.uid in burner_uids
        ]

    async def fetch_miners(self) -> None:
        if self.debug_miner:
            miners = [self.debug_miner]
        else:
            try:
                opted_in_miners = await self._resolve_opted_in_miners()
            except ProviderPortalDataUnavailable:
                if not self.miners:
                    raise
                logger.warning(
                    _m(
                        "[fetch_miners] using in-memory provider snapshot",
                        extra=get_extra_info(
                            {
                                **self.default_extra,
                                "provider_count": len(self.miners),
                            }
                        ),
                    )
                )
                return
            miners = self._build_miners_from_opted_in_snapshot(opted_in_miners)

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

    def _log_scored_hotkeys_missing_from_snapshot(
        self,
        miner_scores: dict[str, float],
        snapshot_miners: Sequence[bittensor.NeuronInfo],
        registered_miners: Sequence[bittensor.NeuronInfo],
        active_hotkeys: set[str],
    ) -> None:
        missing_hotkeys = _classify_missing_scored_hotkeys(
            miner_scores=miner_scores,
            snapshot_miner_hotkeys={miner.hotkey for miner in snapshot_miners},
            registered_hotkeys={miner.hotkey for miner in registered_miners},
            active_hotkeys=active_hotkeys,
        )
        if not (
            missing_hotkeys.active_in_last_cycle
            or missing_hotkeys.inactive_in_last_cycle
        ):
            return

        log_diagnostic = (
            logger.critical
            if missing_hotkeys.active_in_last_cycle
            else logger.warning
        )
        log_diagnostic(
            _m(
                "[set_weights] scored miners missing from provider snapshot",
                extra=get_extra_info(
                    {
                        **self.default_extra,
                        "active_missing_hotkeys": missing_hotkeys.active_in_last_cycle,
                        "inactive_missing_hotkeys": missing_hotkeys.inactive_in_last_cycle,
                    }
                ),
            ),
        )

    async def set_weights(
        self,
        miner_scores: dict[str, float],
        active_hotkeys: set[str] | None = None,
    ) -> None:
        """Set weights using accumulated scores with burning already applied.

        The miner_scores dict already includes burning logic from calculate_final_weights
        called per cycle. This method just normalizes and sends to chain.

        `active_hotkeys` are the hotkeys with at least one live executor in the last
        completed cycle — they get the u16 floor when the vector would drop them.
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

        metagraph = self.get_metagraph()
        self._log_scored_hotkeys_missing_from_snapshot(
            miner_scores=miner_scores,
            snapshot_miners=miners,
            registered_miners=metagraph.neurons,
            active_hotkeys=active_hotkeys or set(),
        )

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

        # DAH-2622: a hotkey that had a live executor last cycle keeps u16=1 even when it
        # scored nothing, so the chain never deregisters it for a zero emission.
        uint_uids, uint_weights, floor_hotkeys = _apply_eligibility_floor(
            uint_uids,
            uint_weights,
            miners,
            active_hotkeys or set(),
            # only the uids that actually receive burn, mirroring BurnService.is_burner —
            # a retired burner slot is an ordinary miner uid and must be able to take the floor
            set(settings.NEW_BURNERS if settings.ENABLE_NEW_BURN_LOGIC else settings.BURNERS),
        )

        logger.info(
            _m(
                "[set_weights] eligibility floor applied",
                extra=get_extra_info({
                    **self.default_extra,
                    "floor_hotkeys": floor_hotkeys,
                    "floor_count": len(floor_hotkeys),
                }),
            ),
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
            except ProviderPortalDataUnavailable as exc:
                logger.error(
                    _m(
                        "[_warm_up_subtensor] Provider portal snapshot unavailable",
                        extra=get_extra_info(
                            {**self.default_extra, "error": str(exc), "backoff": backoff}
                        ),
                    ),
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, SUBTENSOR_BACKOFF_MAX)
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
