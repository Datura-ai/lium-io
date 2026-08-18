import asyncio
import logging
import time

import aiohttp
import bittensor
from pydantic import BaseModel, Field, ValidationError, field_validator

from core.config import settings
from core.utils import _m, get_extra_info


logger = logging.getLogger(__name__)


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
    async def get_opted_in_miners() -> list[OptedInMiner] | None:
        """Return None on failure and an empty list for a successful empty response."""
        try:
            keypair: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
            validator_hotkey = keypair.ss58_address

            api_base = (
                settings.MINER_PORTAL_REST_API_URL.rstrip("/")
                if settings.MINER_PORTAL_REST_API_URL
                else ""
            )
            if not api_base:
                return None

            url = f"{api_base}/validators/opted-in"

            timestamp = int(time.time())
            signature = f"0x{keypair.sign(str(timestamp)).hex()}"

            headers = {
                "hotkey": validator_hotkey,
                "timestamp": str(timestamp),
                "signature": signature,
            }

            # Generous total timeout so we survive short event-loop stalls from concurrent
            # sync bittensor/subtensor calls in this process. aiohttp's timer is driven by
            # the event loop, so a 10s cap fires spuriously whenever the loop stays blocked.
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.get(url, headers=headers) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            logger.error(
                                _m(
                                    "Failed to fetch opted-in miners from portal",
                                    extra=get_extra_info({
                                        "status": resp.status,
                                        "body": text,
                                        "url": url,
                                    }),
                                )
                            )
                            return None

                        data = await resp.json()
                        if not isinstance(data, list):
                            logger.error(
                                _m(
                                    "Invalid opted-in miners response from portal",
                                    extra=get_extra_info({"url": url}),
                                )
                            )
                            return None
                        try:
                            return [OptedInMiner.model_validate(item) for item in data]
                        except ValidationError as exc:
                            logger.error(
                                _m(
                                    "Invalid opted-in miner record from portal",
                                    extra=get_extra_info(
                                        {
                                            "url": url,
                                            "record_count": len(data),
                                            "error": str(exc),
                                        }
                                    ),
                                )
                            )
                            return None
                except asyncio.TimeoutError:
                    logger.error(
                        _m(
                            "Timeout fetching opted-in miners from portal",
                            extra=get_extra_info({"url": url}),
                        )
                    )
                    return None
                except Exception as e:
                    logger.error(
                        _m(
                            "Error fetching opted-in miners from portal",
                            extra=get_extra_info({"url": url, "error": str(e)}),
                        )
                    )
                    return None
        except Exception as e:
            logger.error(
                _m(
                    "Unexpected error during opted-in miners fetch",
                    extra=get_extra_info({"error": str(e)}),
                )
            )
            return None
