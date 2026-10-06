"""Unit tests for the GpuVramPrecheck pipeline check.

The model<->VRAM precheck was moved out of CapabilityCheck (which runs after
the rented short-circuit, TenantEnforcementCheck) into this dedicated early
check, so it now gates rented and idle executors alike. It is a pure-data check
reading only ctx.state.gpu_model and ctx.state.gpu_details[0]["capacity"].
"""
from __future__ import annotations

import pytest

from helpers import build_state
from services.task.checks.gpu_vram_precheck import GpuVramPrecheck


def _state(gpu_model, capacity, count=1):
    details = [{"name": gpu_model, "uuid": "GPU-abc", "capacity": capacity}]
    return build_state(gpu_model=gpu_model, gpu_count=count, gpu_details=details)


@pytest.mark.asyncio
async def test_pass_valid_model_and_vram(context_factory):
    ctx = context_factory(state=_state("NVIDIA H100 80GB HBM3", 81920))
    result = await GpuVramPrecheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == "GPU_VRAM_OK"


@pytest.mark.asyncio
async def test_reject_vram_out_of_range(context_factory):
    ctx = context_factory(state=_state("NVIDIA H100 80GB HBM3", 24064))
    result = await GpuVramPrecheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == "GPU_VRAM_MISMATCH"


@pytest.mark.asyncio
async def test_reject_unknown_model(context_factory):
    # In the real pipeline GpuModelValidCheck rejects unknowns first, but the
    # check must still fail closed if it ever sees one.
    ctx = context_factory(state=_state("Totally Unknown GPU", 24064))
    result = await GpuVramPrecheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == "GPU_VRAM_MISMATCH"


@pytest.mark.asyncio
async def test_reject_missing_capacity(context_factory):
    ctx = context_factory(state=_state("NVIDIA H100 80GB HBM3", 0))
    result = await GpuVramPrecheck().run(ctx)
    assert result.passed is False


@pytest.mark.asyncio
async def test_reject_no_gpu_details(context_factory):
    ctx = context_factory(state=build_state(gpu_model=None, gpu_count=0, gpu_details=[]))
    result = await GpuVramPrecheck().run(ctx)
    assert result.passed is False


@pytest.mark.asyncio
async def test_multivariant_intermediate_rejected(context_factory):
    # V100 is 16/32 GB; a 24 GB reading corresponds to no real variant.
    ctx = context_factory(state=_state("NVIDIA Tesla V100 Tensor Core GPU", 24564))
    result = await GpuVramPrecheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == "GPU_VRAM_MISMATCH"


@pytest.mark.asyncio
async def test_multivariant_real_variant_passes(context_factory):
    # RTX PRO 5000 Blackwell ships in 48 and 72 GB; the 72 GB variant must pass.
    ctx = context_factory(state=_state("NVIDIA RTX PRO 5000 Blackwell", 73728))
    result = await GpuVramPrecheck().run(ctx)
    assert result.passed is True
