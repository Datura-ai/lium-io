import asyncio
import logging
import time
from collections.abc import Mapping

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
