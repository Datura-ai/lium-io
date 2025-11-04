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
    async def get_opt_in_connection_info(miner_hotkey: str) -> dict:
        """Fetch opt-in status and central miner connection details if available.

        Returns dict shape:
            {
              "opt_in_status": bool,
              "miner_hotkey": Optional[str],
            }
        Any error returns opt_in_status=False with None details to preserve behavior.
        """
        try:
            keypair: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
            validator_hotkey = keypair.ss58_address

            api_base = settings.MINER_PORTAL_REST_API_URL.rstrip("/") if settings.MINER_PORTAL_REST_API_URL else ""
            if not api_base:
                return {"opt_in_status": False, "miner_hotkey": None}

            url = f"{api_base}/validators/opt-in-status?miner_hotkey={miner_hotkey}"

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
                                    "Failed to fetch miner opt-in status from portal",
                                    extra=get_extra_info({
                                        "status": resp.status,
                                        "body": text,
                                        "url": url,
                                    }),
                                )
                            )
                            return {"opt_in_status": False, "miner_hotkey": None}

                        data = await resp.json()
                        opt_in_status = bool(data.get("opt_in_status") is True)

                        return {
                            "opt_in_status": opt_in_status,
                            "central_miner_ip": data.get("central_miner_ip"),
                            "central_miner_port": data.get("central_miner_port"),
                        }
                except asyncio.TimeoutError:
                    logger.error(_m("Timeout fetching miner opt-in status from portal", extra=get_extra_info({"url": url})))
                    return {"opt_in_status": False, "miner_hotkey": None, "miner_ip": None, "miner_port": None}
                except Exception as e:
                    logger.error(_m("Error fetching miner opt-in status from portal", extra=get_extra_info({"url": url, "error": str(e)})))
                    return {"opt_in_status": False, "miner_hotkey": None}
        except Exception as e:
            logger.error(_m("Unexpected error during opt-in check", extra=get_extra_info({"error": str(e)})))
            return {"opt_in_status": False, "miner_hotkey": None}

    @staticmethod
    async def check_miner_opt_in_status(miner_hotkey: str) -> bool:
        info = await ValidatorPortalAPI.get_opt_in_connection_info(miner_hotkey)
        return bool(info.get("opt_in_status") is True)


