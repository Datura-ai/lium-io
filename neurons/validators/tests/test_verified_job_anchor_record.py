"""The executor's verified-job record keeps the GPU anchor and its broken mark across every write (DAH-3457).

Through the real RedisService methods on a fakeredis server, in the order the ResultHandler calls them.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from neurons.validators.src.services.task.checks.gpu_fingerprint import GpuFingerprintCheck, cards_replaced
from neurons.validators.src.services.task.checks.spec_change import SpecChangeCheck
from neurons.validators.src.services.task.pipeline import Pipeline
from neurons.validators.src.services.task.result_handler import ResultHandler
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.redis_service import GPU_ANCHOR_BROKEN_KEY, VERIFIED_JOB_COUNT_KEY, RedisService

from tests.helpers import build_state, default_executor

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


async def _cycle(service, handler, context_factory, *, uuids: str, hard: bool, spot: bool = False):
    """One validation cycle as the service runs it: record from Redis, the two GPU-set checks through the real
    Pipeline, the outcome persisted by the ResultHandler."""
    verified = await service.get_verified_job_info("hk", EXECUTOR)
    spot_ids = [default_executor().uuid] if spot else []
    ctx = context_factory(
        state=build_state(
            gpu_uuids=uuids,
            gpu_model_count=f"A100:{len(uuids.split(','))}",
            rented_data=RentedExecutorsResponse(executors={}, spot_executor_ids=spot_ids),
        ),
        verified=verified,
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
@pytest.mark.parametrize(
    ("swapped", "anchored_after"),
    [
        ("gpu-003,gpu-004", "gpu-003,gpu-004"),  # every card replaced (a VM host after a reboot): re-anchored
        ("gpu-001,gpu-003", "gpu-001,gpu-002"),  # one card replaced: the anchor stays
        ("gpu-003", "gpu-001,gpu-002"),  # fewer cards: the anchor stays
    ],
)
async def test_an_idle_node_whose_whole_gpu_set_was_replaced_is_re_anchored(context_factory, swapped, anchored_after):
    """Regression: a node whose cards were all replaced failed GPU_UUID_CHANGED on every cycle under its executor id,
    so a VM host had to remove and re-add the node after each reboot that brought other GPUs."""
    service = _redis_service()
    service.publish = AsyncMock()
    handler = ResultHandler(redis_service=service, dry_run=False)
    await _cycle(service, handler, context_factory, uuids="gpu-001,gpu-002", hard=False)

    ok, event, _ = await _cycle(service, handler, context_factory, uuids=swapped, hard=False)
    assert ok is False and event.reason_code == "GPU_UUID_CHANGED"  # the swap cycle is still a reset

    ok, _, record = await _cycle(service, handler, context_factory, uuids=swapped, hard=False)
    assert record["uuids"] == anchored_after
    assert ok is (anchored_after == swapped)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spot", "swapped", "anchored_after"),
    [
        (True, "gpu-001,gpu-003", "gpu-001,gpu-003"),  # Spot, one card replaced: re-anchored, same node id
        (False, "gpu-001,gpu-003", "gpu-001,gpu-002"),  # off Spot (e.g. Secure), one card replaced: the anchor stays
        (True, "gpu-003", "gpu-001,gpu-002"),  # Spot, a card missing: the anchor stays
        (True, "gpu-001,gpu-002,gpu-003", "gpu-001,gpu-002"),  # Spot, a card added: the anchor stays
    ],
)
async def test_replaced_cards_on_an_idle_spot_node_re_verify_in_place(context_factory, spot, swapped, anchored_after):
    """A provider replaced GPUs on a Spot node and was told to remove and re-add it. The swap cycle says it
    re-verifies in place; the next cycle passes with the new set under the same executor id."""
    service = _redis_service()
    service.publish = AsyncMock()
    handler = ResultHandler(redis_service=service, dry_run=False)
    await _cycle(service, handler, context_factory, uuids="gpu-001,gpu-002", hard=False, spot=spot)

    ok, event, _ = await _cycle(service, handler, context_factory, uuids=swapped, hard=False, spot=spot)
    reanchored = anchored_after == swapped
    assert ok is False and event.reason_code == "GPU_UUID_CHANGED"
    assert event.what_we_saw.get("reanchored", False) is reanchored
    assert ("do not remove the node" in event.remediation) is reanchored

    ok, _, record = await _cycle(service, handler, context_factory, uuids=swapped, hard=False, spot=spot)
    assert record["uuids"] == anchored_after
    assert ok is reanchored


def test_a_rented_node_never_takes_a_new_gpu_set_spot_or_not():
    """A rented node never takes a new GPU set, Spot or not: the renter's pod sits on the anchored cards."""
    assert cards_replaced(["a", "b"], ["a", "c"], rented=True, spot=True) is False
    assert cards_replaced(["a", "b"], ["c", "d"], rented=True, spot=True) is False
    assert cards_replaced(["a", "b"], ["a", "c"], rented=False, spot=True) is True

