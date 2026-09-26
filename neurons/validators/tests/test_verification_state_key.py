"""The verified-job record is keyed by (miner hotkey, executor uuid).

Through the real RedisService, Pipeline and ResultHandler on a fakeredis server.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from neurons.validators.src.services.task.checks.gpu_fingerprint import GpuFingerprintCheck
from neurons.validators.src.services.task.checks.spec_change import SpecChangeCheck
from neurons.validators.src.services.task.pipeline import Pipeline
from neurons.validators.src.services.task.result_handler import ResultHandler
from services.redis_service import VERIFIED_JOB_COUNT_KEY, RedisService, verified_job_field

from tests.helpers import build_state

EXECUTOR = "22222222-2222-2222-2222-222222222222"
HOTKEY_A = "hotkey-a"
HOTKEY_B = "hotkey-b"
SETTINGS = "neurons.validators.src.services.task.checks.gpu_fingerprint.settings"


class _Sink:
    async def emit(self, event) -> None:
        return None


def _redis_service() -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis(server=FakeServer())
    service.lock = asyncio.Lock()
    service.publish = AsyncMock()
    return service


async def _field(service: RedisService, field: str) -> dict | None:
    data = await service.redis.hget(VERIFIED_JOB_COUNT_KEY, field)
    return json.loads(data) if data else None


async def _cycle(service, context_factory, *, hotkey: str, uuids: str) -> bool:
    handler = ResultHandler(redis_service=service, dry_run=False)
    verified = await service.get_verified_job_info(EXECUTOR, hotkey)
    ctx = context_factory(
        state=build_state(gpu_uuids=uuids, gpu_model_count=f"A100:{len(uuids.split(','))}"), verified=verified
    )
    with patch(SETTINGS) as s:
        s.GPU_ANCHOR_HARD_ENABLED = False
        ok, _, ctx = await Pipeline([GpuFingerprintCheck(), SpecChangeCheck()], _Sink()).run(ctx)
    await handler._persist_verification_data(
        miner_hotkey=hotkey, executor_id=EXECUTOR, verified_job_info=verified, context=ctx, success=ok
    )
    return ok


@pytest.mark.asyncio
async def test_verification_state_is_keyed_by_hotkey_and_uuid(context_factory):
    service = _redis_service()
    for _ in range(3):
        assert await _cycle(service, context_factory, hotkey=HOTKEY_A, uuids="gpu-001") is True
    before = await _field(service, verified_job_field(HOTKEY_A, EXECUTOR))
    assert before["count"] == 3

    assert await _cycle(service, context_factory, hotkey=HOTKEY_B, uuids="gpu-009") is True
    await service.clear_verified_job_info(HOTKEY_B, EXECUTOR, prev_info={})

    assert await _field(service, verified_job_field(HOTKEY_A, EXECUTOR)) == before
    assert (await service.get_verified_job_info(EXECUTOR, HOTKEY_A))["count"] == 3


@pytest.mark.asyncio
async def test_state_only_changes_after_fingerprint_match(context_factory):
    service = _redis_service()
    legacy = {"count": 5, "failed": 0, "spec": "A100:1", "uuids": "gpu-001"}
    await service.redis.hset(VERIFIED_JOB_COUNT_KEY, EXECUTOR, json.dumps(legacy))

    assert await _cycle(service, context_factory, hotkey=HOTKEY_B, uuids="gpu-009") is False
    assert await service.redis.hget(VERIFIED_JOB_COUNT_KEY, EXECUTOR) is not None
    assert (await service.get_verified_job_info(EXECUTOR, HOTKEY_A))["count"] == 5

    assert await _cycle(service, context_factory, hotkey=HOTKEY_A, uuids="gpu-001") is True
    record = await _field(service, verified_job_field(HOTKEY_A, EXECUTOR))
    assert (record["count"], record["uuids"]) == (6, "gpu-001")


@pytest.mark.asyncio
async def test_existing_state_is_preserved_on_upgrade(context_factory):
    service = _redis_service()
    legacy = {"count": 40, "failed": 2, "spec": "A100:1", "uuids": "gpu-001"}
    await service.redis.hset(VERIFIED_JOB_COUNT_KEY, EXECUTOR, json.dumps(legacy))

    assert await service.get_verified_job_info(EXECUTOR, HOTKEY_A) == legacy
    assert await _cycle(service, context_factory, hotkey=HOTKEY_A, uuids="gpu-001") is True

    record = await _field(service, verified_job_field(HOTKEY_A, EXECUTOR))
    assert (record["count"], record["failed"], record["uuids"]) == (41, 2, "gpu-001")
    assert await service.redis.hget(VERIFIED_JOB_COUNT_KEY, EXECUTOR) is None
    assert await service.get_verified_job_info(EXECUTOR, HOTKEY_B) == {}
