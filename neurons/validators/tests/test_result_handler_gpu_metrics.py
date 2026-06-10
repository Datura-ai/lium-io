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
