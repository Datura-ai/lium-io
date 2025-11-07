import aiohttp
import asyncio
import json
import logging
import time

import bittensor

from core.config import settings
from core.utils import _m, get_extra_info


logger = logging.getLogger(__name__)


class ValidatorPortalAPI:
    @staticmethod
    async def get_opted_in_miners() -> list[dict]:
        """Fetch list of miners that have opted in.

        Returns list of dicts with shape:
            [
                {
                    "miner_hotkey": str,
                    "miner_coldkey": str,
                    "central_miner_ip": str,
                    "central_miner_port": int,
                },
                ...
            ]
        Returns empty list on error.
        """
        try:
            keypair: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
            validator_hotkey = keypair.ss58_address

            api_base = settings.MINER_PORTAL_REST_API_URL.rstrip("/") if settings.MINER_PORTAL_REST_API_URL else ""
            if not api_base:
                return []

            url = f"{api_base}/validators/opted-in"

            timestamp = int(time.time())
            signature = f"0x{keypair.sign(str(timestamp)).hex()}"

            headers = {
                "hotkey": validator_hotkey,
                "timestamp": str(timestamp),
                "signature": signature,
            }

            timeout = aiohttp.ClientTimeout(total=10)
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
                            return []

                        data = await resp.json()
                        return data if isinstance(data, list) else []
                except asyncio.TimeoutError:
                    logger.error(_m("Timeout fetching opted-in miners from portal", extra=get_extra_info({"url": url})))
                    return []
                except Exception as e:
                    logger.error(_m("Error fetching opted-in miners from portal", extra=get_extra_info({"url": url, "error": str(e)})))
                    return []
        except Exception as e:
            logger.error(_m("Unexpected error during opted-in miners fetch", extra=get_extra_info({"error": str(e)})))
            return []
