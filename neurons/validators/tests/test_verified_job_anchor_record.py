"""The executor's verified-job record keeps the GPU anchor and its broken mark across every write (DAH-3457).

Through the real RedisService methods on a fakeredis server, in the order the ResultHandler calls them.
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
from services.redis_service import GPU_ANCHOR_BROKEN_KEY, VERIFIED_JOB_COUNT_KEY, RedisService, verified_job_field

from tests.helpers import build_state

EXECUTOR = "11111111-1111-1111-1111-111111111111"
SETTINGS = "neurons.validators.src.services.task.checks.gpu_fingerprint.settings"


class _Sink:
    async def emit(self, event) -> None:
        return None


def _redis_service() -> RedisService:
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis(server=FakeServer())
    service.lock = asyncio.Lock()
    return service


async def _record(service: RedisService) -> dict:
    return json.loads(await service.redis.hget(VERIFIED_JOB_COUNT_KEY, verified_job_field("hk", EXECUTOR)))


@pytest.mark.asyncio
async def test_anchor_survives_reset_failure_and_success_writes():
    """Regression: a reset, a failed cycle or a later success rewrites `uuids` from the current scrape, so a
    swapped node re-anchors on the new set under the same executor id."""
    service = _redis_service()
    service.publish = AsyncMock()

    await service.set_verified_job_info("hk", EXECUTOR, prev_info={}, success=True, spec="A100:1", uuids="gpu-001")
    info = await _record(service)
    assert info["uuids"] == "gpu-001"

    await service.clear_verified_job_info("hk", EXECUTOR, prev_info=info)
    info = await _record(service)
    assert info == {"count": 0, "failed": 0, "spec": "A100:1", "uuids": "gpu-001"}

    await service.set_verified_job_info("hk", EXECUTOR, prev_info=info, success=False)
    info = await _record(service)
    assert (info["uuids"], info["failed"]) == ("gpu-001", 1)

    await service.set_verified_job_info("hk", EXECUTOR, prev_info=info, success=True, spec="A100:1", uuids="gpu-002")
    info = await _record(service)
    assert info["uuids"] == "gpu-001"
    assert GPU_ANCHOR_BROKEN_KEY not in info


@pytest.mark.asyncio
async def test_broken_mark_is_written_once_and_kept_by_every_later_write():
    """Regression: the reset that breaks the anchor does not store the mark, or a later failed/successful
    write drops it, so the node is re-verified on its next clean scrape."""
    service = _redis_service()
    service.publish = AsyncMock()
    info = {"count": 12, "failed": 1, "spec": "A100:1", "uuids": "gpu-001"}

    await service.clear_verified_job_info("hk", EXECUTOR, prev_info=info, anchor_broken=True)
    info = await _record(service)
    assert info[GPU_ANCHOR_BROKEN_KEY] is True
    assert info["uuids"] == "gpu-001"
    assert (info["count"], info["failed"]) == (0, 0)

    await service.set_verified_job_info("hk", EXECUTOR, prev_info=info, success=False)
    info = await _record(service)
    assert info[GPU_ANCHOR_BROKEN_KEY] is True

    await service.set_verified_job_info("hk", EXECUTOR, prev_info=info, success=True, spec="A100:1", uuids="gpu-001")
    info = await _record(service)
    assert info[GPU_ANCHOR_BROKEN_KEY] is True

    await service.clear_verified_job_info("hk", EXECUTOR, prev_info=info)
    info = await _record(service)
    assert info[GPU_ANCHOR_BROKEN_KEY] is True


@pytest.mark.asyncio
async def test_result_handler_passes_the_broken_mark_to_the_reset(context_factory):
    """Regression: the check sets gpu_anchor_broken on the context and the handler's reset call drops it,
    so the record is reset but never marked."""
    service = _redis_service()
    service.publish = AsyncMock()
    handler = ResultHandler(redis_service=service, dry_run=False)
    ctx = context_factory(
        state=build_state(gpu_uuids="gpu-003"),
        clear_verified_job_info=True,
        gpu_anchor_broken=True,
        clear_verified_job_evidence={"reason_code": "GPU_UUID_CHANGED", "check_id": "gpu.validate.fingerprint"},
    )
    prev = {"count": 3, "failed": 0, "spec": "A100:1", "uuids": "gpu-001"}

    await handler._persist_verification_data(
        miner_hotkey="hk", executor_id=EXECUTOR, verified_job_info=prev, context=ctx, success=False
    )

    info = await _record(service)
    assert info[GPU_ANCHOR_BROKEN_KEY] is True
    assert info["uuids"] == "gpu-001"
    published = service.publish.await_args.args[1]
    assert published["reason_code"] == "GPU_UUID_CHANGED"
    assert published["executor_uuid"] == EXECUTOR


async def _cycle(service, handler, context_factory, *, uuids: str, hard: bool):
    """One validation cycle as the service runs it: record from Redis, the two GPU-set checks through the real
    Pipeline, the outcome persisted by the ResultHandler."""
    verified = await service.get_verified_job_info(EXECUTOR, "hk")
    ctx = context_factory(
        state=build_state(gpu_uuids=uuids, gpu_model_count=f"A100:{len(uuids.split(','))}"), verified=verified
    )
    with patch(SETTINGS) as s:
        s.GPU_ANCHOR_HARD_ENABLED = hard
        ok, events, ctx = await Pipeline([GpuFingerprintCheck(), SpecChangeCheck()], _Sink()).run(ctx)
    await handler._persist_verification_data(
        miner_hotkey="hk", executor_id=EXECUTOR, verified_job_info=verified, context=ctx, success=ok
    )
    return ok, events[-1], await _record(service)


@pytest.mark.asyncio
async def test_a_swap_breaks_the_anchor_through_pipeline_context_and_handler(context_factory):
    """Regression: the check's `gpu_anchor_broken` update and the Context field drift apart (model_copy drops an
    unknown key without error), so the check reports a broken anchor and the record is never marked; or the
    anchored set coming back re-verifies a broken node."""
    service = _redis_service()
    service.publish = AsyncMock()
    handler = ResultHandler(redis_service=service, dry_run=False)

    ok, _, record = await _cycle(service, handler, context_factory, uuids="gpu-001", hard=True)
    assert ok is True and record["uuids"] == "gpu-001" and GPU_ANCHOR_BROKEN_KEY not in record

    ok, event, record = await _cycle(service, handler, context_factory, uuids="gpu-002", hard=True)
    assert ok is False and event.reason_code == "GPU_UUID_CHANGED"
    assert record["uuids"] == "gpu-001" and record[GPU_ANCHOR_BROKEN_KEY] is True
    assert service.publish.await_count == 1

    ok, event, record = await _cycle(service, handler, context_factory, uuids="gpu-001", hard=True)
    assert ok is False and event.what_we_saw["anchor_broken"] is True
    assert record[GPU_ANCHOR_BROKEN_KEY] is True and record["failed"] == 1
    # no second reset for a node already reset on the cycle that broke it
    assert service.publish.await_count == 1


@pytest.mark.asyncio
async def test_a_missing_card_is_gpu_missing_and_the_node_passes_when_it_is_back(context_factory):
    """Regression: SpecChangeCheck runs first again (the count changed, SPEC_CHANGED), or GPU_MISSING marks the
    anchor broken so the node never passes once the card is back."""
    service = _redis_service()
    service.publish = AsyncMock()
    handler = ResultHandler(redis_service=service, dry_run=False)

    ok, _, _ = await _cycle(service, handler, context_factory, uuids="gpu-001,gpu-002", hard=True)
    assert ok is True

    ok, event, record = await _cycle(service, handler, context_factory, uuids="gpu-001", hard=True)
    assert ok is False and event.reason_code == "GPU_MISSING" and event.check_id == "gpu.validate.fingerprint"
    assert record["uuids"] == "gpu-001,gpu-002" and GPU_ANCHOR_BROKEN_KEY not in record

    ok, _, record = await _cycle(service, handler, context_factory, uuids="gpu-001,gpu-002", hard=True)
    assert ok is True and record["count"] == 1
