import asyncio

import pytest

from neurons.validators.src.services.task.pipeline import Pipeline
from tests.helpers import build_state
from tests.test_validation_fast_path import _Sink, _StateCheck


@pytest.mark.asyncio
async def test_a_validation_cut_by_the_task_timeout_logs_the_step_it_was_cut_in(context_factory, caplog):
    # Arrange
    ctx = context_factory(state=build_state(specs={"gpu": {"count": 8}}))
    pipeline = Pipeline([_StateCheck("ports"), _StateCheck("verifyx", seconds=10)], _Sink())

    # Act
    with caplog.at_level("WARNING"), pytest.raises(TimeoutError):
        await asyncio.wait_for(pipeline.run(ctx), timeout=0.05)

    # Assert
    [record] = [r for r in caplog.records if r.getMessage() == "Validation cancelled mid-step"]
    assert record.msg.extra["cancelled_check_ids"] == ["verifyx"]
    assert list(record.msg.extra["steps"]) == ["ports"]
