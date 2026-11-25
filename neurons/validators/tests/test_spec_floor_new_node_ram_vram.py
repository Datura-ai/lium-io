"""SpecFloorCheck: a NEW node whose system RAM is below half its total GPU VRAM is not onboarded.

RED->GREEN: before, such a node passed validation and was listed (only the idle-pay gate saw it). The rule is the
idle-pay gate's own (rental_price), so a node that onboards also clears the idle-pay RAM rule. A node that has
passed validation before (a GPU anchor on its verified record) or is under a live rental is never failed by it.
"""

from __future__ import annotations

import pytest
from neurons.validators.src.services.task.checks.spec_floor import SpecFloorCheck
from neurons.validators.src.services.task.messages import SpecFloorMessages as Msg

from tests.helpers import build_state
from tests.test_gpu_fingerprint_check import _live_rental

KIB_PER_GB = 1024 * 1024


def _uuid(i: int) -> str:
    return f"GPU-{i:08x}-1a2b-3c4d-5e6f-0123456789ab"


def _specs(*, ram_gb: float, gpu_count: int, vram_gb_per_gpu: float):
    return {
        "cpu": {"count": 32},
        "ram": {"total": ram_gb * KIB_PER_GB},
        "gpu": {
            "count": gpu_count,
            "details": [
                {"name": "NVIDIA H100 80GB HBM3", "uuid": _uuid(i + 1), "capacity": vram_gb_per_gpu * 1024}
                for i in range(gpu_count)
            ],
        },
    }


async def _run(context_factory, *, verified=None, rented=None, **kwargs):
    ctx = context_factory(state=build_state(specs=_specs(**kwargs), rented_data=rented), verified=verified or {})
    return await SpecFloorCheck().run(ctx)


@pytest.mark.asyncio
async def test_new_node_with_ram_below_half_vram_is_rejected_with_the_ram_needed(context_factory):
    # 8x 80 GB = 640 GB VRAM; 200 GB RAM is below 320 GB (half), even with the MemTotal tolerance
    result = await _run(context_factory, ram_gb=200, gpu_count=8, vram_gb_per_gpu=80)

    assert (result.passed, result.fatal) == (False, None)
    assert result.event.reason_code == Msg.REJECTED.reason
    assert "ram_below_half_vram" in result.event.what_we_saw["findings"]
    assert result.event.what_we_saw["required_ram_gb"] == 304.8
    assert result.event.what_we_saw["vram_gb"] == 640.0


@pytest.mark.parametrize("ram_gb", [320, 305, 1024])
@pytest.mark.asyncio
async def test_new_node_at_or_above_half_vram_passes(ram_gb, context_factory):
    # 305 GB reads as MemTotal of a 320 GB box: inside the tolerance the idle-pay gate allows
    result = await _run(context_factory, ram_gb=ram_gb, gpu_count=8, vram_gb_per_gpu=80)

    assert result.passed is True


@pytest.mark.asyncio
async def test_a_node_verified_before_is_not_rejected(context_factory):
    verified = {"count": 40, "failed": 0, "spec": "NVIDIA H100 80GB HBM3:8", "uuids": _uuid(1)}
    result = await _run(context_factory, verified=verified, ram_gb=200, gpu_count=8, vram_gb_per_gpu=80)

    assert result.passed is True


@pytest.mark.asyncio
async def test_a_node_under_a_live_rental_is_not_rejected(context_factory):
    result = await _run(context_factory, rented=_live_rental(), ram_gb=200, gpu_count=8, vram_gb_per_gpu=80)

    assert result.passed is True


@pytest.mark.asyncio
async def test_a_scrape_without_vram_fails_open(context_factory):
    result = await _run(context_factory, ram_gb=200, gpu_count=8, vram_gb_per_gpu=0)

    assert result.passed is True


@pytest.mark.asyncio
async def test_the_rejection_tells_the_provider_the_ram_needed(context_factory):
    result = await _run(context_factory, ram_gb=200, gpu_count=8, vram_gb_per_gpu=80)

    assert "at least 304.8 GB of RAM" in result.event.remediation
    assert "640.0 GB of VRAM" in result.event.remediation
    assert "reports 200.0 GB" in result.event.remediation
