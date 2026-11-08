"""DAH-6011: on shutdown the running cycle finishes (bounded) instead of being cut off."""

import asyncio

import pytest
from core.validator import Validator


def _validator() -> Validator:
    validator = Validator.__new__(Validator)
    validator.default_extra = {}
    validator.should_exit = False
    return validator


async def _loop(validator: Validator, cycle_seconds: float, cycles: list[str]) -> None:
    # the shape of Validator.start(): a cycle, then a check of should_exit
    while not validator.should_exit:
        cycles.append("started")
        await asyncio.sleep(cycle_seconds)
        cycles.append("finished")


@pytest.mark.asyncio
async def test_running_cycle_finishes_and_no_new_one_starts() -> None:
    validator, cycles = _validator(), []
    task = asyncio.create_task(_loop(validator, 0.05, cycles))
    await asyncio.sleep(0.01)  # mid-cycle

    assert await validator.finish_running_cycle(task, wait_seconds=1) is True
    assert cycles == ["started", "finished"]
    assert task.done() and not task.cancelled()


@pytest.mark.asyncio
async def test_cycle_past_the_bound_is_cancelled() -> None:
    validator, cycles = _validator(), []
    task = asyncio.create_task(_loop(validator, 10, cycles))
    await asyncio.sleep(0.01)

    assert await validator.finish_running_cycle(task, wait_seconds=0.05) is False
    assert task.cancelled()
    assert cycles == ["started"]
