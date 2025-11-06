import aiohttp
import asyncio
import logging
from typing import Any

from core.config import settings
from core.utils import _m, get_extra_info
from protocol.miner_request import AuthenticateRequest

logger = logging.getLogger(__name__)


class MinerPortalAPI:
    @staticmethod
    async def fetch_executors(miner_hotkey: str, executor_id: str | None) -> list[dict[str, Any]]:
        api_url = f"{settings.MINER_PORTAL_API_URL}/miners/{miner_hotkey}/executors"
        if executor_id:
            api_url += f"?executor_id={executor_id}"

        keypair = settings.get_bittensor_wallet().get_hotkey()
        auth = AuthenticateRequest.from_keypair(keypair)

        headers = {
            "hotkey": auth.payload.miner_hotkey,
            "timestamp": str(auth.payload.timestamp),
            "signature": auth.signature,
        }

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(api_url, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(
                            _m(
                                "Failed to fetch executors from portal",
                                extra=get_extra_info(
                                    {
                                        "status": resp.status,
                                        "body": text,
                                        "url": api_url,
                                    }
                                ),
                            )
                        )
                        return []
                    data = await resp.json()
                    # Expecting list of executors in JSON
                    if not isinstance(data, list):
                        logger.error(
                            _m(
                                "Unexpected portal response shape",
                                extra=get_extra_info({"url": api_url, "type": type(data).__name__}),
                            )
                        )
                        return []
                    return data
            except asyncio.TimeoutError:
                logger.error(
                    _m("Timeout fetching executors from portal", extra=get_extra_info({"url": api_url}))
                )
                return []
            except Exception as e:
                logger.error(
                    _m(
                        "Error fetching executors from portal",
                        extra=get_extra_info({"url": api_url, "error": str(e)}),
                    )
                )
                return []


