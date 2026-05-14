import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from uuid import uuid4

from payload_models.forced_validation import (
    ForceValidationError,
    ForceValidationRequestRecord,
    ForceValidationResult,
)
from payload_models.payloads import MinerJobRequestPayload

from clients.backend_client import BackendClient
from clients.subtensor_client import SubtensorClient
from core.config import settings
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.file_encrypt_service import FileEncryptService
from services.miner_service import MinerService
from services.redis_service import RedisService
from services.task_service import JobResult

logger = logging.getLogger(__name__)

FORCED_VALIDATION_REQUEST_PREFIX = "forced_validation:request"
FORCED_VALIDATION_ACTIVE_PREFIX = "forced_validation:active_executor"
FORCED_VALIDATION_LATEST_PREFIX = "forced_validation:latest_executor"
DEFAULT_REQUEST_TTL_SECONDS = 24 * 60 * 60


class ForceValidationConflict(Exception):
    pass


class ForceValidationNotFound(Exception):
    pass


class ForceValidationRequestStore:
    def __init__(
        self,
        redis_service: RedisService,
        *,
        request_ttl_seconds: int = DEFAULT_REQUEST_TTL_SECONDS,
        active_ttl_seconds: int | None = None,
    ):
        self.redis_service = redis_service
        self.request_ttl_seconds = request_ttl_seconds
        self.active_ttl_seconds = active_ttl_seconds or settings.JOB_TIME_OUT + 600

    def _request_key(self, request_id: str) -> str:
        return f"{FORCED_VALIDATION_REQUEST_PREFIX}:{request_id}"

    def _active_key(self, executor_id: str) -> str:
        return f"{FORCED_VALIDATION_ACTIVE_PREFIX}:{executor_id}"

    def _latest_key(self, executor_id: str) -> str:
        return f"{FORCED_VALIDATION_LATEST_PREFIX}:{executor_id}"

    async def create_request(
        self, *, executor_id: str, miner_hotkey: str
    ) -> ForceValidationRequestRecord:
        request_id = str(uuid4())
        active_created = await self.redis_service.redis.set(
            self._active_key(executor_id),
            request_id,
            nx=True,
            ex=self.active_ttl_seconds,
        )
        if not active_created:
            raise ForceValidationConflict()

        record = ForceValidationRequestRecord(
            request_id=request_id,
            executor_id=executor_id,
            miner_hotkey=miner_hotkey,
            status="queued",
            stage="queued",
        )
        await self._save(record)
        return record

    async def get_request(self, request_id: str) -> ForceValidationRequestRecord:
        raw = await self.redis_service.redis.get(self._request_key(request_id))
        if raw is None:
            raise ForceValidationNotFound()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return ForceValidationRequestRecord.model_validate_json(raw)

    async def get_latest_request(self, executor_id: str) -> ForceValidationRequestRecord:
        request_id = await self.redis_service.redis.get(self._latest_key(executor_id))
        if request_id is None:
            raise ForceValidationNotFound()
        if isinstance(request_id, bytes):
            request_id = request_id.decode("utf-8")
        return await self.get_request(request_id)

    async def update(
        self,
        request_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        result: ForceValidationResult | None = None,
        error: ForceValidationError | None = None,
    ) -> ForceValidationRequestRecord:
        record = await self.get_request(request_id)
        update_data = {}
        if status is not None:
            update_data["status"] = status
        if stage is not None:
            update_data["stage"] = stage
        if result is not None:
            update_data["result"] = result
            update_data["error"] = None
        if error is not None:
            update_data["error"] = error
            update_data["result"] = None
        updated = record.model_copy(
            update={**update_data, "updated_at": datetime.now(UTC)}
        )
        await self._save(updated)
        return updated

    async def release_active_executor(self, executor_id: str, request_id: str) -> None:
        active_key = self._active_key(executor_id)
        active_request_id = await self.redis_service.redis.get(active_key)
        if isinstance(active_request_id, bytes):
            active_request_id = active_request_id.decode("utf-8")
        if active_request_id == request_id:
            await self.redis_service.redis.delete(active_key)

    async def _save(self, record: ForceValidationRequestRecord) -> None:
        await self.redis_service.redis.set(
            self._request_key(record.request_id),
            record.model_dump_json(),
            ex=self.request_ttl_seconds,
        )
        await self.redis_service.redis.set(
            self._latest_key(record.executor_id),
            record.request_id,
            ex=self.request_ttl_seconds,
        )


class ForceValidationService:
    def __init__(
        self,
        *,
        store: ForceValidationRequestStore,
        miner_service: MinerService,
        subtensor_client: SubtensorClient,
        backend_client: BackendClient,
        file_encrypt_service: FileEncryptService,
        task_factory: Callable[[Coroutine], asyncio.Task] = asyncio.create_task,
    ):
        self.store = store
        self.miner_service = miner_service
        self.subtensor_client = subtensor_client
        self.backend_client = backend_client
        self.file_encrypt_service = file_encrypt_service
        self.task_factory = task_factory

    async def create_request(
        self, executor_id: str, miner_hotkey: str
    ) -> ForceValidationRequestRecord:
        record = await self.store.create_request(
            executor_id=executor_id,
            miner_hotkey=miner_hotkey,
        )
        self.task_factory(self.validate(record.request_id))
        return record

    async def get_request(self, request_id: str) -> ForceValidationRequestRecord:
        return await self.store.get_request(request_id)

    async def get_latest_request(self, executor_id: str) -> ForceValidationRequestRecord:
        return await self.store.get_latest_request(executor_id)

    async def validate(self, request_id: str) -> ForceValidationRequestRecord:
        record = await self.store.get_request(request_id)
        try:
            await self.store.update(
                request_id,
                status="running",
                stage="resolving_miner",
            )
            miner = await self._resolve_miner(record.miner_hotkey)

            await self.store.update(request_id, stage="preparing_validation")
            rented_data = await self.backend_client.get_all_rented_executors()
            if rented_data is None:
                rented_data = RentedExecutorsResponse(executors={})
            encrypted_files = self.file_encrypt_service.ecrypt_miner_job_files()

            payload = MinerJobRequestPayload(
                job_batch_id=f"forced-{request_id}",
                miner_hotkey=miner.hotkey,
                miner_coldkey=miner.coldkey,
                miner_address=miner.axon_info.ip,
                miner_port=miner.axon_info.port,
            )

            await self.store.update(request_id, stage="running_validation")
            result = await self.miner_service.request_single_executor_validation(
                payload=payload,
                encrypted_files=encrypted_files,
                rented_data=rented_data,
                executor_id=record.executor_id,
            )

            await self.store.update(request_id, stage="finalizing")
            await self.miner_service.publish_machine_specs(
                [result],
                record.miner_hotkey,
                miner.coldkey,
            )

            terminal_result = self._build_terminal_result(result)
            return await self.store.update(
                request_id,
                status="succeeded" if terminal_result.success else "failed",
                stage="completed",
                result=terminal_result,
            )
        except Exception as exc:
            logger.exception(
                "Forced validation failed",
                extra={
                    "request_id": request_id,
                    "executor_id": record.executor_id,
                    "miner_hotkey": record.miner_hotkey,
                },
            )
            return await self.store.update(
                request_id,
                status="failed",
                stage="completed",
                error=ForceValidationError(message=str(exc)),
            )
        finally:
            await self.store.release_active_executor(record.executor_id, request_id)

    async def _resolve_miner(self, miner_hotkey: str):
        miners = await self.subtensor_client.get_miners()
        matches = [miner for miner in miners if miner.hotkey == miner_hotkey]
        if not matches:
            raise ValueError(f"Miner {miner_hotkey} was not found in subtensor")
        return matches[0]

    @staticmethod
    def _build_terminal_result(result: JobResult) -> ForceValidationResult:
        success = result.log_status == "info"
        return ForceValidationResult(
            success=success,
            message=result.log_text,
            score=result.score,
            job_score=result.job_score,
            gpu_model=result.gpu_model,
            gpu_count=result.gpu_count,
        )
