"""Standardized HTTP client for backend API requests with validator signature."""

import asyncio
import json
import logging
import time
from typing import Any, ClassVar, TypeVar

import aiohttp
import bittensor
from pydantic import BaseModel, ValidationError

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BackendClient:
    """HTTP client with session pooling and validator signature headers."""

    _session: ClassVar[aiohttp.ClientSession | None] = None
    _session_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self, base_url: str, keypair: bittensor.Keypair):
        self.base_url = base_url.rstrip("/")
        self.keypair = keypair

    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        if cls._session is None or cls._session.closed:
            async with cls._session_lock:
                if cls._session is None or cls._session.closed:
                    connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
                    cls._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=30),
                        connector=connector,
                    )
        return cls._session

    @classmethod
    async def close_session(cls) -> None:
        if cls._session is not None and not cls._session.closed:
            await cls._session.close()
            cls._session = None

    def _get_signature_headers(self) -> dict[str, str]:
        timestamp = int(time.time())
        return {
            "hotkey": self.keypair.ss58_address,
            "timestamp": str(timestamp),
            "signature": f"0x{self.keypair.sign(str(timestamp)).hex()}",
        }

    async def get(
        self,
        path: str,
        response_model: type[T],
        *,
        add_signature: bool = True,
        timeout: int = 10,
        extra_headers: dict[str, str] | None = None,
    ) -> T | None:
        url = f"{self.base_url}/{path.lstrip('/')}"
        context = {"url": url, "method": "GET"}

        try:
            headers = self._get_signature_headers() if add_signature else {}
            if extra_headers:
                headers.update(extra_headers)

            session = await self.get_session()
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    logger.error(
                        _m(
                            "HTTP GET failed",
                            extra=get_extra_info({**context, "status": resp.status}),
                        )
                    )
                    return None

                try:
                    data = await resp.json()
                except (aiohttp.ContentTypeError, json.JSONDecodeError) as e:
                    logger.error(
                        _m("Invalid JSON", extra=get_extra_info({**context, "error": str(e)}))
                    )
                    return None

                try:
                    return response_model.model_validate(data)
                except ValidationError as e:
                    logger.error(
                        _m("Validation failed", extra=get_extra_info({**context, "error": str(e)}))
                    )
                    return None

        except TimeoutError:
            logger.error(_m("GET timeout", extra=get_extra_info(context)))
            return None
        except aiohttp.ClientError as e:
            logger.error(_m("GET client error", extra=get_extra_info({**context, "error": str(e)})))
            return None
        except Exception as e:
            logger.error(
                _m("GET error", extra=get_extra_info({**context, "error": str(e)})), exc_info=True
            )
            return None

    async def post(
        self,
        path: str,
        response_model: type[T],
        *,
        json_data: dict[str, Any] | None = None,
        add_signature: bool = True,
        timeout: int = 10,
        extra_headers: dict[str, str] | None = None,
    ) -> T | None:
        url = f"{self.base_url}/{path.lstrip('/')}"
        context = {"url": url, "method": "POST"}

        try:
            headers = self._get_signature_headers() if add_signature else {}
            if extra_headers:
                headers.update(extra_headers)

            session = await self.get_session()
            async with session.post(
                url, headers=headers, json=json_data, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    logger.error(
                        _m(
                            "HTTP POST failed",
                            extra=get_extra_info({**context, "status": resp.status}),
                        )
                    )
                    return None

                try:
                    data = await resp.json()
                except (aiohttp.ContentTypeError, json.JSONDecodeError) as e:
                    logger.error(
                        _m("Invalid JSON", extra=get_extra_info({**context, "error": str(e)}))
                    )
                    return None

                try:
                    return response_model.model_validate(data)
                except ValidationError as e:
                    logger.error(
                        _m("Validation failed", extra=get_extra_info({**context, "error": str(e)}))
                    )
                    return None

        except TimeoutError:
            logger.error(_m("POST timeout", extra=get_extra_info(context)))
            return None
        except aiohttp.ClientError as e:
            logger.error(
                _m("POST client error", extra=get_extra_info({**context, "error": str(e)}))
            )
            return None
        except Exception as e:
            logger.error(
                _m("POST error", extra=get_extra_info({**context, "error": str(e)})), exc_info=True
            )
            return None
