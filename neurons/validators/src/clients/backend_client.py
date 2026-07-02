"""Standardized HTTP client for backend API requests with validator signature."""

import asyncio
import json
import logging
import time
from typing import Any, ClassVar, TypeVar
from urllib.parse import urlencode

import aiohttp
import bittensor
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_result, stop_after_attempt, wait_fixed

from core.utils import _m, get_extra_info
from protocol.vc_protocol.compute_requests import (
    DefaultDockerImage,
    DefaultDockerImagesResponse,
    ExecutorHealthCheckResponse,
    PodRentalActiveResponse,
    RentedExecutorsResponse,
)

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
                        timeout=aiohttp.ClientTimeout(total=None),
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
        timeout: int = 30,
        extra_headers: dict[str, str] | None = None,
    ) -> T | None:
        url = f"{self.base_url}/{path.lstrip('/')}"
        context = {"url": url, "method": "GET"}

        try:
            headers = self._get_signature_headers() if add_signature else {}
            if extra_headers:
                headers.update(extra_headers)

            session = await self.get_session()
            start_time = time.perf_counter()
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    _m(
                        "HTTP GET completed",
                        extra=get_extra_info({**context, "status": resp.status, "elapsed_ms": round(elapsed_ms, 1)}),
                    )
                )
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
        timeout: int = 30,
        extra_headers: dict[str, str] | None = None,
    ) -> T | None:
        url = f"{self.base_url}/{path.lstrip('/')}"
        context = {"url": url, "method": "POST"}

        try:
            headers = self._get_signature_headers() if add_signature else {}
            if extra_headers:
                headers.update(extra_headers)

            session = await self.get_session()
            start_time = time.perf_counter()
            async with session.post(
                url, headers=headers, json=json_data, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    _m(
                        "HTTP POST completed",
                        extra=get_extra_info({**context, "status": resp.status, "elapsed_ms": round(elapsed_ms, 1)}),
                    )
                )
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(30),
        retry=retry_if_result(lambda x: x is None),
    )
    async def get_all_rented_executors(self) -> RentedExecutorsResponse | None:
        """Fetch all rented executors with their ports from the backend."""
        return await self.get(
            "/internal/executors/rented",
            RentedExecutorsResponse,
            timeout=30,
        )

    async def get_default_docker_image(
        self, gpu_model: str, driver_version: str
    ) -> list[DefaultDockerImage] | None:
        """Fetch the recommended/default template image(s) for a GPU + driver combo.

        Hits the same public `/executors/default-docker-image` endpoint the executor
        cache pre-pull consumes (DAH-2265 Plan 2). Advisory caller: no retry (must not
        block the validation pipeline), unsigned (mirrors the executor), and `get()`
        already returns None on any error / non-200 / bad JSON, so this fails open.
        """
        query = urlencode({"gpu_model": gpu_model, "driver_version": driver_version})
        response = await self.get(
            f"/executors/default-docker-image?{query}",
            DefaultDockerImagesResponse,
            add_signature=False,
            timeout=15,
        )
        return response.root if response is not None else None

    async def get_pod_rental_active(self, pod_id: str) -> PodRentalActiveResponse | None:
        """Fetch whether the backend still considers a pod rental active."""
        return await self.get(
            f"/internal/pods/{pod_id}/rental-active",
            PodRentalActiveResponse,
            timeout=10,
        )

    async def check_executor_health(
        self,
        miner_address: str,
        miner_port: int,
        miner_hotkey: str,
        container_port: int,
        executor_id: str | None = None,
        rental_in_progress: bool = False,
    ) -> ExecutorHealthCheckResponse | None:
        """Check executor health via backend API.

        Args:
            miner_address: Miner IP address
            miner_port: Miner SSH port
            miner_hotkey: Miner hotkey (SS58 address)
            container_port: Container port to check
            executor_id: Executor ID (optional)
            rental_in_progress: True if this validator already sees an active customer rental for the
                executor. Lets the backend skip the container-creating check immediately; the backend
                still re-checks the DB itself when this is False.

        Returns:
            ExecutorHealthCheckResponse if successful, None otherwise
        """
        validator_hotkey = self.keypair.ss58_address
        path = f"/validator/{validator_hotkey}/executor-health-check"

        json_data = {
            "miner_address": miner_address,
            "miner_port": miner_port,
            "miner_hotkey": miner_hotkey,
            "container_port": container_port,
            "rental_in_progress": rental_in_progress,
        }

        if executor_id:
            json_data["executor_id"] = executor_id

        return await self.post(
            path,
            ExecutorHealthCheckResponse,
            json_data=json_data,
            add_signature=True,  # Use standard signature headers
            timeout=300,  # 5 minutes - backend needs time to SSH and verify container
        )
