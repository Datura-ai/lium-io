import asyncio
import json
import logging
import time
from collections.abc import Mapping
from datetime import UTC, datetime

import aiohttp
import bittensor
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from core.config import settings
from core.utils import _m, get_extra_info


logger = logging.getLogger(__name__)


class PortalExecutor(BaseModel):
    """One row of the portal's bulk executor snapshot (DAH-2958). Only what the express lane reads."""

    model_config = ConfigDict(extra="ignore")

    id: str
    validator_hotkey: str
    # The portal stamps rows with a naive datetime.utcnow(); registration → AVAILABLE is measured
    # from here, so it is read as UTC.
    created_at: datetime | None = None

    @field_validator("created_at")
    @classmethod
    def as_utc(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


# DAH-6740: per own-key listing hotkey, the own hotkeys of its owner's other accounts, from this validator's own read of
# the portal's /miners/owner-hotkeys (never from what a miner reports: a miner could then have keys signed for hotkeys
# it names). That answer is complete on every call, so each one replaces this copy whole (a removed link stops counting);
# a failed read keeps the last answer. Filled by ValidatorPortalAPI.refresh_owner_hotkeys, which the express lane calls
# every tick.
_owner_hotkeys: dict[str, frozenset[str]] = {}
# bounds the signatures one upload carries; an owner has a few accounts, not dozens
MAX_OWNER_HOTKEYS = 16


def remember_owner_hotkeys(answer: object) -> bool:
    """Replace the owner map with the portal's answer ({listing hotkey: [hotkeys]}); False, keeping the map, when the
    answer is not that shape. An entry over MAX_OWNER_HOTKEYS gets no signatures at all."""
    if not isinstance(answer, dict) or not all(
        isinstance(listing, str) and isinstance(hotkeys, list) and all(isinstance(h, str) for h in hotkeys)
        for listing, hotkeys in answer.items()
    ):
        return False
    owners = {
        listing: frozenset(hotkeys) - {listing}
        for listing, hotkeys in answer.items()
        if len(set(hotkeys)) <= MAX_OWNER_HOTKEYS
    }
    _owner_hotkeys.clear()
    _owner_hotkeys.update({listing: hotkeys for listing, hotkeys in owners.items() if hotkeys})
    return True


def portal_owner_hotkeys(listing_hotkey: str) -> frozenset[str]:
    return _owner_hotkeys.get(listing_hotkey, frozenset())


def portal_miner_auth_blob(hotkey: str, timestamp: int) -> str:
    """The exact bytes the portal's signature_auth_miner verifies (AuthenticationPayload)."""
    return json.dumps({"miner_hotkey": hotkey, "timestamp": timestamp}, sort_keys=True)


class OptedInMiner(BaseModel):
    miner_hotkey: str = Field(min_length=1)
    central_miner_ip: str = Field(min_length=1)
    central_miner_port: int = Field(strict=True, ge=1, le=65535)

    @field_validator("miner_hotkey", "central_miner_ip")
    @classmethod
    def validate_nonempty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("central_miner_ip")
    @classmethod
    def validate_serving_ip(cls, value: str) -> str:
        if value == "0.0.0.0":
            raise ValueError("must identify a serving endpoint")
        return value


class ValidatorPortalAPI:
    @staticmethod
    def _validate_opted_in_miners_response(
        response_data: object,
        url: str,
    ) -> list[OptedInMiner] | None:
        if not isinstance(response_data, list):
            logger.error(
                _m(
                    "Invalid opted-in miners response from portal",
                    extra=get_extra_info({"url": url}),
                )
            )
            return None

        try:
            return [OptedInMiner.model_validate(item) for item in response_data]
        except ValidationError as exc:
            logger.error(
                _m(
                    "Invalid opted-in miner record from portal",
                    extra=get_extra_info(
                        {
                            "url": url,
                            "record_count": len(response_data),
                            "error": str(exc),
                        }
                    ),
                )
            )
            return None

    @staticmethod
    async def _request_opted_in_miners(
        url: str,
        headers: Mapping[str, str],
    ) -> list[OptedInMiner] | None:
        # aiohttp's timer uses the event loop, so short timeouts fire during synchronous
        # bittensor/subtensor calls elsewhere in this process.
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    response_body = await response.text()
                    logger.error(
                        _m(
                            "Failed to fetch opted-in miners from portal",
                            extra=get_extra_info(
                                {
                                    "status": response.status,
                                    "body": response_body,
                                    "url": url,
                                }
                            ),
                        )
                    )
                    return None

                response_data = await response.json()
                return ValidatorPortalAPI._validate_opted_in_miners_response(
                    response_data,
                    url,
                )

    @staticmethod
    async def get_opted_in_miners() -> list[OptedInMiner] | None:
        """Return None on failure and an empty list for a successful empty response."""
        api_base = (
            settings.MINER_PORTAL_REST_API_URL.rstrip("/")
            if settings.MINER_PORTAL_REST_API_URL
            else ""
        )
        if not api_base:
            return None

        url = f"{api_base}/validators/opted-in"
        try:
            keypair: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
            timestamp = int(time.time())
            return await ValidatorPortalAPI._request_opted_in_miners(
                url=url,
                headers={
                    "hotkey": keypair.ss58_address,
                    "timestamp": str(timestamp),
                    "signature": f"0x{keypair.sign(str(timestamp)).hex()}",
                },
            )
        except asyncio.TimeoutError:
            logger.error(
                _m(
                    "Timeout fetching opted-in miners from portal",
                    extra=get_extra_info({"url": url}),
                )
            )
            return None
        except Exception as exc:
            logger.error(
                _m(
                    "Error fetching opted-in miners from portal",
                    extra=get_extra_info({"url": url, "error": str(exc)}),
                )
            )
            return None

    @staticmethod
    async def get_all_executors() -> dict[str, list[PortalExecutor]] | None:
        """{miner_hotkey: [executor]} for every opted-in miner; None on any failure.

        DAH-2958: the express lane discovers never-validated executors here, between cycles.
        This is the bulk snapshot the central miner already polls (DAH-2469); the endpoint is
        guarded by signature_auth_miner, which verifies the signature over AuthenticationPayload,
        so the validator signs the same blob with its own hotkey. The guard's subnet-registration
        check (portal auth/signature_auth.py, commented out today with a note to restore it) is
        SubtensorClient.get_miner, a lookup over every neuron of the subnet, which a registered
        validator hotkey passes too.
        """
        api_base = (
            settings.MINER_PORTAL_REST_API_URL.rstrip("/")
            if settings.MINER_PORTAL_REST_API_URL
            else ""
        )
        if not api_base:
            return None

        url = f"{api_base}/miners/executors"
        try:
            keypair: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
            timestamp = int(time.time())
            headers = {
                "hotkey": keypair.ss58_address,
                "timestamp": str(timestamp),
                "signature": f"0x{keypair.sign(portal_miner_auth_blob(keypair.ss58_address, timestamp)).hex()}",
            }
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(
                            _m(
                                "Failed to fetch executor snapshot from portal",
                                extra=get_extra_info(
                                    {
                                        "status": response.status,
                                        "body": await response.text(),
                                        "url": url,
                                    }
                                ),
                            )
                        )
                        return None
                    data = await response.json()
            if not isinstance(data, dict):
                raise ValueError(f"unexpected portal bulk response shape: {type(data).__name__}")
            return {
                miner_hotkey: [PortalExecutor.model_validate(item) for item in executors]
                for miner_hotkey, executors in data.items()
            }
        except Exception as exc:
            logger.error(
                _m(
                    "Error fetching executor snapshot from portal",
                    extra=get_extra_info({"url": url, "error": str(exc)}),
                )
            )
            return None

    @staticmethod
    async def refresh_owner_hotkeys() -> bool:
        """DAH-6740: replace the owner map with the portal's /miners/owner-hotkeys; False (map kept) on any failure."""
        api_base = settings.MINER_PORTAL_REST_API_URL.rstrip("/") if settings.MINER_PORTAL_REST_API_URL else ""
        if not api_base:
            return False
        url = f"{api_base}/miners/owner-hotkeys"
        try:
            keypair: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
            timestamp = int(time.time())
            headers = {
                "hotkey": keypair.ss58_address,
                "timestamp": str(timestamp),
                "signature": f"0x{keypair.sign(portal_miner_auth_blob(keypair.ss58_address, timestamp)).hex()}",
            }
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        raise ValueError(f"status {response.status}")
                    answer = await response.json()
            if not remember_owner_hotkeys(answer):
                raise ValueError("unexpected owner-hotkeys response shape")
            return True
        except Exception as exc:
            logger.warning(
                _m("Owner hotkeys not refreshed from portal", extra=get_extra_info({"url": url, "error": str(exc)}))
            )
            return False
