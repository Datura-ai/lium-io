from __future__ import annotations

from types import SimpleNamespace

import pytest

from neurons.validators.src.services.task.result_handler import ResultHandler

from tests.helpers import build_state


def _miner_info():
    # handle_result only reads .miner_hotkey (logging) and .job_batch_id.
    return SimpleNamespace(miner_hotkey="miner-hotkey", job_batch_id="batch-1")


def _context(context_factory, *, gpu_metrics):
    state = build_state(specs={"gpu": {"count": 1}}, gpu_metrics=gpu_metrics)
    return context_factory(
        state=state,
        tdx_attestation_passed=False,
        score=1.0,
        job_score=1.0,
        collateral_deposited=False,
        ssh_pub_keys=[],
        rented=False,
    )


@pytest.mark.asyncio
async def test_handle_result_nests_gpu_metrics_in_spec(context_factory):
    """When state carries gpu_metrics, the published spec nests them under gpu_metrics."""
    metrics = {"device": "NVIDIA A100", "cc": "8.0", "fp32_tflops": 51.2}
    ctx = _context(context_factory, gpu_metrics=metrics)
    handler = ResultHandler(redis_service=None, dry_run=True)

    result = await handler.handle_result(
        context=ctx,
        miner_info=_miner_info(),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )

    assert result.spec["gpu_metrics"] == metrics
    assert result.spec["gpu_metrics"]["fp32_tflops"] == 51.2


@pytest.mark.asyncio
async def test_handle_result_omits_gpu_metrics_when_absent(context_factory):
    """Fail-safe: with no gpu_metrics in state, the published spec has no gpu_metrics key."""
    ctx = _context(context_factory, gpu_metrics=None)
    handler = ResultHandler(redis_service=None, dry_run=True)

    result = await handler.handle_result(
        context=ctx,
        miner_info=_miner_info(),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )

    assert "gpu_metrics" not in result.spec


@pytest.mark.asyncio
async def test_handle_result_defaults_inspector_outcome_to_skipped(context_factory):
    ctx = context_factory(
        score=1.0,
        job_score=1.0,
        collateral_deposited=False,
        ssh_pub_keys=[],
        rented=False,
    )
    handler = ResultHandler(redis_service=None, dry_run=True)

    result = await handler.handle_result(
        context=ctx,
        miner_info=_miner_info(),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )

    assert result.inspector_outcome == "SKIPPED"


@pytest.mark.asyncio
async def test_handle_result_publishes_inspector_event(context_factory):
    from unittest.mock import AsyncMock

    from services.redis_service import INSPECTOR_EVENT_CHANNEL

    inspector_event = {
        "executor_id": "executor-123",
        "outcome": "CLEAN",
        "reason_code": "INSPECTOR_CLEAN",
        "report": {"canary_ok": True, "findings": []},
        "when": "2026-06-17T12:00:00+00:00",
    }
    state = build_state(inspector_event=inspector_event)
    ctx = context_factory(
        state=state,
        score=1.0,
        job_score=1.0,
        collateral_deposited=False,
        ssh_pub_keys=[],
        rented=True,
    )
    redis = AsyncMock()
    handler = ResultHandler(redis_service=redis, dry_run=False)

    result = await handler.handle_result(
        context=ctx,
        miner_info=_miner_info(),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )

    assert result.inspector_outcome == "CLEAN"
    redis.publish.assert_awaited_once_with(
        INSPECTOR_EVENT_CHANNEL,
        {**inspector_event, "miner_hotkey": "miner-hotkey"},
    )


@pytest.mark.asyncio
async def test_handle_result_skips_inspector_event_in_dry_run(context_factory):
    from unittest.mock import AsyncMock

    state = build_state(
        inspector_event={
            "executor_id": "executor-123",
            "outcome": "CLEAN",
            "reason_code": "INSPECTOR_CLEAN",
        }
    )
    ctx = context_factory(state=state, rented=True)
    redis = AsyncMock()
    handler = ResultHandler(redis_service=redis, dry_run=True)

    await handler.handle_result(
        context=ctx,
        miner_info=_miner_info(),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )

    redis.publish.assert_not_awaited()


# DAH-2265 Plan 2: advisory recommended_image_cached signal rides executor.specs.


def _context_cached(context_factory, *, recommended_image_cached):
    state = build_state(
        specs={"gpu": {"count": 1}},
        recommended_image_cached=recommended_image_cached,
    )
    return context_factory(
        state=state,
        tdx_attestation_passed=False,
        score=1.0,
        job_score=1.0,
        collateral_deposited=False,
        ssh_pub_keys=[],
        rented=False,
    )


@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.asyncio
async def test_handle_result_publishes_recommended_image_cached(context_factory, cached):
    """When measured (True/False), the published spec carries recommended_image_cached."""
    ctx = _context_cached(context_factory, recommended_image_cached=cached)
    handler = ResultHandler(redis_service=None, dry_run=True)

    result = await handler.handle_result(
        context=ctx,
        miner_info=_miner_info(),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )

    assert result.spec["recommended_image_cached"] is cached


@pytest.mark.asyncio
async def test_handle_result_omits_recommended_image_cached_when_not_measured(context_factory):
    """Fail-safe: None (skipped/fail-open) → no recommended_image_cached key published."""
    ctx = _context_cached(context_factory, recommended_image_cached=None)
    handler = ResultHandler(redis_service=None, dry_run=True)

    result = await handler.handle_result(
        context=ctx,
        miner_info=_miner_info(),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )

    assert "recommended_image_cached" not in result.spec
