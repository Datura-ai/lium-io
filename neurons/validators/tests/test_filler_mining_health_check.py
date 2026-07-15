"""Tests for FillerMiningHealthCheck (DAH-2419).

Non-fatal diagnostic: when a Lium filler is RUNNING but not holding the GPU (dead worker / firmware
wedge), snap the worker log. It must NEVER fail the miner and must rate-limit the SSH log capture.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from helpers import build_services, build_state, default_executor, make_context
from neurons.validators.src.services.task.checks import FillerMiningHealthCheck, filler_mining_health

from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

FILLER = "filler_abc123"


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: str) -> None:
        self.store[key] = value


def _ctx(*, gpu_processes=None, has_filler: bool = True):
    executor = default_executor()
    rented = RentedExecutorsResponse(
        executors={},
        filler_containers_by_executor={str(executor.uuid): FILLER} if has_filler else {},
    )
    state = build_state(rented_data=rented, gpu_processes=gpu_processes or [])
    services = build_services(redis=FakeRedis())
    return make_context(executor=executor, services=services, state=state, ssh="ssh-sentinel")


def test_check_is_non_fatal() -> None:
    assert FillerMiningHealthCheck().fatal is False


@pytest.mark.asyncio
async def test_no_filler_passes_without_capture(monkeypatch) -> None:
    capture = AsyncMock()
    monkeypatch.setattr(filler_mining_health, "collect_container_death_diagnostics", capture)
    result = await FillerMiningHealthCheck().run(_ctx(has_filler=False))
    assert result.passed is True
    capture.assert_not_awaited()


@pytest.mark.asyncio
async def test_filler_holding_gpu_is_ok_no_capture(monkeypatch) -> None:
    capture = AsyncMock()
    monkeypatch.setattr(filler_mining_health, "collect_container_death_diagnostics", capture)
    result = await FillerMiningHealthCheck().run(_ctx(gpu_processes=[{"container_name": FILLER}]))
    assert result.passed is True
    capture.assert_not_awaited()


@pytest.mark.asyncio
async def test_filler_not_on_gpu_captures_worker_log(monkeypatch) -> None:
    diag = SimpleNamespace(to_log_fields=lambda: {"container_logs_tail": "OOM during model load"})
    capture = AsyncMock(return_value=diag)
    monkeypatch.setattr(filler_mining_health, "collect_container_death_diagnostics", capture)
    # Nothing from the filler is on the GPU (dead / firmware-wedged) → capture, but still passes.
    result = await FillerMiningHealthCheck().run(_ctx(gpu_processes=[]))
    assert result.passed is True
    capture.assert_awaited_once()


@pytest.mark.asyncio
async def test_capture_is_rate_limited(monkeypatch) -> None:
    capture = AsyncMock(return_value=SimpleNamespace(to_log_fields=lambda: {}))
    monkeypatch.setattr(filler_mining_health, "collect_container_death_diagnostics", capture)
    ctx = _ctx(gpu_processes=[])
    check = FillerMiningHealthCheck()
    await check.run(ctx)
    await check.run(ctx)  # same executor within the cooldown → second capture suppressed
    capture.assert_awaited_once()


@pytest.mark.asyncio
async def test_diagnostics_failure_still_passes(monkeypatch) -> None:
    capture = AsyncMock(side_effect=RuntimeError("ssh dropped"))
    monkeypatch.setattr(filler_mining_health, "collect_container_death_diagnostics", capture)
    result = await FillerMiningHealthCheck().run(_ctx(gpu_processes=[]))
    assert result.passed is True  # a capture failure never fails the miner
