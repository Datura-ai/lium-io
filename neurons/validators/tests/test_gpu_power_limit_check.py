from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from helpers import build_state
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.gpu_power_limit import (
    GpuPowerRestoreReadResult,
    GpuPowerRestoreRecord,
)
from services.task.checks import gpu_power_limit as check_module
from services.task.checks.gpu_power_limit import GpuPowerLimitCheck
from services.task.pipeline import ContextState

# Matches default_executor().uuid in helpers.py — the executor the context_factory builds.
_EXECUTOR_UUID = "executor-123"


def _read_result(
    *records: GpuPowerRestoreRecord, read_failed: bool = False
) -> GpuPowerRestoreReadResult:
    return GpuPowerRestoreReadResult(records=list(records), read_failed=read_failed)


@pytest.fixture
def read_records_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    # DAH-2356 safety net: on a below-floor reading the check consults Redis for this validator's
    # own stale filler caps. Tests drive it through this mock (ctx.services.redis is None here).
    mock = AsyncMock(return_value=_read_result())
    monkeypatch.setattr(check_module, "read_gpu_power_restore_records", mock)
    return mock


def _state(
    current_limit: float | None,
    default_limit: float | None,
    max_limit: float = 450,
    count: int = 1,
    default_job_owner: str | None = None,
    rented_data_known: bool = True,
    min_limit: float | None = None,
    gpu_owner_map: dict[str, str] | None = None,
) -> ContextState:
    details = [
        {
            "name": "NVIDIA L40S",
            "uuid": "GPU-abc" if index == 0 else f"GPU-abc-{index}",
            "power_limit": current_limit,
            "power_default_limit": default_limit,
            "power_max_limit": max_limit,
        }
        | ({"power_min_limit": min_limit} if min_limit is not None else {})
        for index in range(count)
    ]
    rented_data = None
    if rented_data_known:
        owner_map = {_EXECUTOR_UUID: default_job_owner} if default_job_owner is not None else {}
        rented_data = RentedExecutorsResponse(
            executors={},
            default_job_owner_by_executor=owner_map,
            default_job_owner_by_gpu=gpu_owner_map or {},
        )
    return build_state(
        gpu_model="NVIDIA L40S",
        gpu_count=count,
        gpu_details=details,
        rented_data=rented_data,
    )


@pytest.mark.asyncio
async def test_power_limit_passes_when_current_is_close_to_default(context_factory, read_records_mock) -> None:
    ctx = context_factory(state=_state(current_limit=320, default_limit=350))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == "GPU_POWER_LIMIT_OK"
    read_records_mock.assert_not_awaited()  # healthy nodes cost zero Redis traffic


@pytest.mark.asyncio
async def test_power_limit_rejects_below_threshold(context_factory, read_records_mock) -> None:
    ctx = context_factory(state=_state(current_limit=105, default_limit=350))
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == "GPU_POWER_LIMIT_BELOW_DEFAULT"
    assert result.updates["score"] == 0.0
    assert result.updates["job_score"] == 0.0
    assert result.event.what_we_saw["rejected_gpus"][0]["power_limit_ratio"] == 0.3


@pytest.mark.asyncio
async def test_power_limit_rejects_capped_gpu_outside_the_lium_gpu_map(
    context_factory, read_records_mock
) -> None:
    # A capped GPU that no Lium filler runs on fails, whatever other GPUs the map lists.
    ctx = context_factory(
        state=_state(current_limit=105, default_limit=350, gpu_owner_map={"GPU-other": "lium"})
    )
    result = await GpuPowerLimitCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == "GPU_POWER_LIMIT_BELOW_DEFAULT"
    assert result.updates["score"] == 0.0


