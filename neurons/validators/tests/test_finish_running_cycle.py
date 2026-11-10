"""DAH-6011: on shutdown the running cycle finishes (bounded) instead of being cut off."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from core import validator as validator_module
from core.validator import Validator


class _Lane:
    async def run(self, should_exit) -> None:
        return None


def _validator(monkeypatch: pytest.MonkeyPatch, cycle_seconds: float, cycles: list[str]) -> Validator:
    monkeypatch.setattr(validator_module.settings, "DRY_RUN", False)
    monkeypatch.setattr(validator_module, "SYNC_CYCLE", 0.001)
    monkeypatch.setattr(validator_module, "start_loop_sampler", lambda: asyncio.Event())
    validator = Validator.__new__(Validator)
    validator.default_extra = {}
    validator.should_exit = False
    validator.initiate_services = AsyncMock()
    validator.express_lane = _Lane()

    async def sync() -> None:
        cycles.append("started")
        await asyncio.sleep(cycle_seconds)
        cycles.append("finished")

    validator.sync = sync
    return validator


@pytest.mark.asyncio
async def test_running_cycle_finishes_and_no_new_one_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    cycles: list[str] = []
    validator = _validator(monkeypatch, 0.05, cycles)
    task = asyncio.create_task(validator.start())
    await asyncio.sleep(0.01)  # mid-cycle

    assert await validator.finish_running_cycle(task, wait_seconds=1) is True
    assert cycles == ["started", "finished"]
    assert task.done() and not task.cancelled()


@pytest.mark.asyncio
async def test_cycle_past_the_bound_is_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    cycles: list[str] = []
    validator = _validator(monkeypatch, 10, cycles)
    task = asyncio.create_task(validator.start())
    await asyncio.sleep(0.01)

    assert await validator.finish_running_cycle(task, wait_seconds=0.05) is False
    assert task.cancelled()
    assert cycles == ["started"]
