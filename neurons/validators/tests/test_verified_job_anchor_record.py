"""The executor's verified-job record keeps the GPU anchor across every write, except a full set change (DAH-3457).

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
from services.redis_service import GPU_ANCHOR_BROKEN_KEY, VERIFIED_JOB_COUNT_KEY, RedisService

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
    return json.loads(await service.redis.hget(VERIFIED_JOB_COUNT_KEY, f"hk:{EXECUTOR}"))


async def _cycle(service, handler, context_factory, *, uuids: str, hard: bool):
    """One validation cycle as the service runs it: record from Redis, the two GPU-set checks through the real
    Pipeline, the outcome persisted by the ResultHandler."""
    verified = await service.get_verified_job_info("hk", EXECUTOR)
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
@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.parametrize(
    ("swapped", "passes", "anchored_after"),
    [
        ("gpu-003,gpu-004", True, "gpu-003,gpu-004"),  # every card other (a VM host after a reboot): re-anchored
        ("gpu-001,gpu-003", True, "gpu-001,gpu-003"),  # one card other, same count: re-anchored
        ("gpu-003", False, "gpu-001,gpu-002"),  # fewer cards: a dropout, the anchor stays
    ],
)
async def test_a_full_set_of_other_cards_is_re_anchored_and_verified_the_same_cycle(
    context_factory, hard, swapped, passes, anchored_after
):
    """Regression: a node that came back with the same number of other GPUs failed (reset, broken anchor or a hold)
    instead of being verified on its new cards at once."""
    service = _redis_service()
    service.publish = AsyncMock()
    handler = ResultHandler(redis_service=service, dry_run=False)
    await _cycle(service, handler, context_factory, uuids="gpu-001,gpu-002", hard=hard)

    ok, _, record = await _cycle(service, handler, context_factory, uuids=swapped, hard=hard)
    assert ok is passes and record["uuids"] == anchored_after and GPU_ANCHOR_BROKEN_KEY not in record
    # no verification reset for a full set change
    assert service.publish.await_count == (0 if passes else 1)
    if passes:
        assert record["count"] == 2  # verified on the new cards, not reset

    ok, _, record = await _cycle(service, handler, context_factory, uuids=swapped, hard=hard)
    assert record["uuids"] == anchored_after and ok is passes
