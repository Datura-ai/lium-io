from __future__ import annotations

import pytest

from helpers import build_state
from services.task.checks.gpu_power_limit import GpuPowerLimitCheck


def _state(current_limit, default_limit, max_limit=450, count=1):
    details = [
        {
            "name": "NVIDIA L40S",
            "uuid": "GPU-abc",
            "power_limit": current_limit,
            "power_default_limit": default_limit,
            "power_max_limit": max_limit,
        }
        for _ in range(count)
    ]
    return build_state(
        gpu_model="NVIDIA L40S",
        gpu_count=count,
        gpu_details=details,
    )


@pytest.mark.asyncio
async def test_power_limit_passes_when_current_is_close_to_default(context_factory):
    ctx = context_factory(state=_state(current_limit=320, default_limit=350))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_OK"


@pytest.mark.asyncio
async def test_power_limit_rejects_below_threshold(context_factory):
    ctx = context_factory(state=_state(current_limit=105, default_limit=350))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == "GPU_POWER_LIMIT_BELOW_DEFAULT"
    assert result.updates["score"] == 0.0
    assert result.updates["job_score"] == 0.0
    assert result.updates["persist_zero_score_specs"] is True
    assert result.updates["zero_score_persistence_reason"] == "GPU_POWER_LIMIT_BELOW_DEFAULT"
    assert result.event.what_we_saw["rejected_gpus"][0]["power_limit_ratio"] == 0.3


@pytest.mark.asyncio
async def test_power_limit_allows_exact_threshold(context_factory):
    ctx = context_factory(state=_state(current_limit=280, default_limit=350))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_OK"


@pytest.mark.asyncio
async def test_power_limit_passes_when_default_limit_missing(context_factory):
    ctx = context_factory(state=_state(current_limit=105, default_limit=None))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_BASELINE_INCOMPLETE"
