"""Standardized HTTP client for backend API requests with validator signature."""

import asyncio
import json
import logging
import time
from typing import Any, ClassVar, TypeVar
from urllib.parse import urlencode

import aiohttp
import bittensor
from protocol.vc_protocol.compute_requests import (
    DefaultDockerImage,
    DefaultDockerImagesResponse,
    ExecutorHealthCheckResponse,
    FillerRunActiveResponse,
    NvmlReportAckResponse,
    PodHostRebootRecoveredResponse,
    PodRentalActiveResponse,
    RentedExecutorsResponse,
    VerificationStartedResponse,
)
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_result, stop_after_attempt, wait_fixed

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BackendClient:
    """HTTP client with session pooling and validator signature headers."""

    _session: ClassVar[aiohttp.ClientSession | None] = None
    _session_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    # DAH-2360: writing to a stale pooled keep-alive connection fails with
    # EPIPE before the request reaches the backend, so these errors are safe
    # to retry even for non-idempotent POSTs. Timeouts and 5xx are NOT retried:
    # by then the request may have reached the backend and run server-side.
    CONNECTION_ERRORS: ClassVar[tuple[type[Exception], ...]] = (
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientConnectorError,
        aiohttp.ClientOSError,
    )
    # Short first pause (a stale connection heals instantly), longer second one
    # so a restarting backend pod has time to become ready.
    CONNECTION_RETRY_BACKOFF_SECONDS: ClassVar[tuple[float, ...]] = (1.0, 5.0)
    CONNECTION_RETRY_ATTEMPTS: ClassVar[int] = len(CONNECTION_RETRY_BACKOFF_SECONDS)
    # Drop idle pooled connections well before the server/ingress closes them.
    KEEPALIVE_TIMEOUT_SECONDS: ClassVar[int] = 30

    def __init__(self, base_url: str, keypair: bittensor.Keypair):
        self.base_url = base_url.rstrip("/")
        self.keypair = keypair

    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        if cls._session is None or cls._session.closed:
            async with cls._session_lock:
                if cls._session is None or cls._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=100,
                        limit_per_host=10,
                        keepalive_timeout=cls.KEEPALIVE_TIMEOUT_SECONDS,
                    )
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
        return await self._request(
            "GET",
            path,
            response_model,
            add_signature=add_signature,
            timeout=timeout,
            extra_headers=extra_headers,
        )

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
        return await self._request(
            "POST",
            path,
            response_model,
            json_data=json_data,
            add_signature=add_signature,
            timeout=timeout,
            extra_headers=extra_headers,
        )

    async def _request(
        self,
        method: str,
        path: str,
        response_model: type[T],
        *,
        json_data: dict[str, Any] | None = None,
        add_signature: bool = True,
        timeout: int = 30,
        extra_headers: dict[str, str] | None = None,
    ) -> T | None:
        # single signed round-trip, retrying connection-level errors per backoff schedule
        url = f"{self.base_url}/{path.lstrip('/')}"
        context = {"url": url, "method": method}

        try:
            headers = self._get_signature_headers() if add_signature else {}
            if extra_headers:
                headers.update(extra_headers)

            session = await self.get_session()
            for attempt in range(1 + self.CONNECTION_RETRY_ATTEMPTS):
                try:
                    start_time = time.perf_counter()
                    async with session.request(
                        method,
                        url,
                        headers=headers,
                        json=json_data,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        logger.info(
                            _m(
                                f"HTTP {method} completed",
                                extra=get_extra_info({**context, "status": resp.status, "elapsed_ms": round(elapsed_ms, 1)}),
                            )
                        )
                        if resp.status != 200:
                            logger.error(
                                _m(
                                    f"HTTP {method} failed",
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
                except self.CONNECTION_ERRORS as e:
                    if attempt >= self.CONNECTION_RETRY_ATTEMPTS:
                        raise
                    logger.warning(
                        _m(
                            f"{method} connection error, retrying",
                            extra=get_extra_info(
                                {**context, "error": str(e), "attempt": attempt + 1}
                            ),
                        )
                    )
                    await asyncio.sleep(self.CONNECTION_RETRY_BACKOFF_SECONDS[attempt])

        except TimeoutError:
            logger.error(_m(f"{method} timeout", extra=get_extra_info(context)))
            return None
        except aiohttp.ClientError as e:
            logger.error(
                _m(f"{method} client error", extra=get_extra_info({**context, "error": str(e)}))
            )
            return None
        except Exception as e:
            logger.error(
                _m(f"{method} error", extra=get_extra_info({**context, "error": str(e)})), exc_info=True
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

    async def report_pod_host_reboot_recovered(
        self, pod_id: str, container_finished_at: str
    ) -> PodHostRebootRecoveredResponse | None:
        """Tell the backend a pod was revived after its host rebooted, so the renter can be told.

        container_finished_at is the container's exit time, which the backend uses to recognise
        the same downtime reported by a later cycle while keeping a genuine second reboot of the
        same pod a separate event. Older backends 404 and the caller treats that as nothing to do.
        """
        return await self.post(
            f"/internal/pods/{pod_id}/host-reboot-recovered",
            PodHostRebootRecoveredResponse,
            json_data={"container_finished_at": container_finished_at},
            timeout=10,
        )

    async def get_filler_run_active(
        self, filler_run_id: str, *, container_missing: bool = False
    ) -> FillerRunActiveResponse | None:
        """Fetch whether the backend still considers a Lium filler run active.

        container_missing=True tells the backend this re-check was triggered by the filler
        container being ABSENT on the host: the backend accumulates those reports and closes a
        run the validator keeps seeing containerless (DAH-2471 zombie reconciliation). Older
        backends simply ignore the query param.
        """
        path = f"/internal/filler-runs/{filler_run_id}/active"
        if container_missing:
            path += "?container_missing=true"
        return await self.get(path, FillerRunActiveResponse, timeout=10)

    async def check_executor_health(
        self,
        miner_address: str,
        miner_port: int,
        miner_hotkey: str,
        container_port: int,
        executor_id: str | None = None,
        rental_in_progress: bool = False,
        gpu_uuids: list[str] | None = None,
        cpu_count: int | None = None,
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
            gpu_uuids: GPU UUIDs from the scrape this cycle. The backend's probe asks the container
                runtime to resolve exactly these instead of `--gpus all`, which never names a card
                and so let a host advertising GPUs it cannot hand over pass the check (DAH-2614).
            cpu_count: Advertised CPU-core count from the scrape this cycle. The backend creates the
                probe with `--cpus=<cpu_count>`, so a host claiming more cores than it physically has
                is rejected by its own Docker daemon (CPU_QUOTA_EXCEEDS_HOST) rather than only in a
                paying customer's rent (DAH-2671). None omits the flag.

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
            # Sent from this cycle's scrape rather than read from the backend's stored specs: those
            # are a cycle behind and absent entirely on an executor's first pass — the very check
            # where a fabricated GPU matters most.
            "gpu_uuids": gpu_uuids or [],
        }

        # Same rationale as gpu_uuids: send the count this cycle's scrape produced, not the backend's
        # stored specs. Omitted when unset so an unchanged backend keeps the pre-DAH-2671 behaviour.
        if cpu_count is not None:
            json_data["cpu_count"] = cpu_count

        if executor_id:
            json_data["executor_id"] = executor_id

        return await self.post(
            path,
            ExecutorHealthCheckResponse,
            json_data=json_data,
            add_signature=True,  # Use standard signature headers
            timeout=300,  # 5 minutes - backend needs time to SSH and verify container
        )

    async def report_verification_started(
        self, executor_uuid: str, *, job_batch_id: str, pipeline_id: str, miner_hotkey: str
    ) -> None:
        """Tell the backend this executor's validation pipeline has just started (DAH-3019).

        One call per executor per run. The provider portal turns it, together with the per-step
        durations of earlier runs, into "verifying · step 3/6 · ~70 s left"; without it the portal
        only learns of a run when the whole cycle publishes, minutes after the node's own checks.
        Fire-and-forget: never raises, a failure costs the provider a progress bar, not a verdict.
        Older backends 404 and that is logged at the client's error level like any non-200.
        """
        try:
            await self.post(
                f"/validator/{self.keypair.ss58_address}/executors/{executor_uuid}/verification-started",
                VerificationStartedResponse,
                json_data={
                    "job_batch_id": job_batch_id,
                    "pipeline_id": pipeline_id,
                    "miner_hotkey": miner_hotkey,
                },
                add_signature=True,
                timeout=10,
            )
        except Exception as exc:
            logger.warning(
                _m(
                    "Failed to report verification start",
                    extra={"executor_uuid": executor_uuid, "job_batch_id": job_batch_id, "error": str(exc)},
                )
            )

    async def report_unknown_driver(self, driver_version: str) -> None:
        """Ask the backend to verify an unknown NVIDIA driver (DAH-2451).

        Fire-and-forget: the backend enqueues + resolves the driver against NVIDIA's
        official installer, and the verdict reaches this validator later via shared
        config. Never raises — a reporting failure must not disturb the pipeline.
        """
        try:
            await self.post(
                "/v1/nvml-driver/report-unknown",
                NvmlReportAckResponse,
                json_data={"driver_version": driver_version},
                add_signature=True,
                timeout=10,
            )
        except Exception as exc:
            logger.warning(_m("Failed to report unknown driver", extra={"driver": driver_version, "error": str(exc)}))
