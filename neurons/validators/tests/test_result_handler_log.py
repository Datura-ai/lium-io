import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from neurons.validators.src.services.task.result_handler import ResultHandler
from protocol.vc_protocol.validator_requests import ValidationEvent

from tests.helpers import build_state


@pytest.mark.asyncio
async def test_result_line_carries_the_wave_and_the_node_check_time(context_factory, caplog):
    # Arrange
    ctx = context_factory(state=build_state(specs={"gpu": {"count": 1}})).model_copy(
        update={"started_at_monotonic": time.monotonic() - 42}
    )
    event = ValidationEvent(
        event="x", reason_code="X", severity="info", impact="x", when=datetime(2026, 11, 24, tzinfo=UTC)
    )

    # Act
    with caplog.at_level("INFO"):
        await ResultHandler(redis_service=None, dry_run=True).handle_result(
            context=ctx,
            miner_info=SimpleNamespace(miner_hotkey="miner-hotkey", job_batch_id="2026-11-24 13:00:00"),
            executor_info=ctx.executor,
            verified_job_info={},
            log_text="x",
            success=True,
            validation_event=event,
        )

    # Assert
    [record] = [r for r in caplog.records if r.getMessage() == "Handle task result"]
    assert record.msg.extra["job_batch_id"] == "2026-11-24 13:00:00"
    assert 42 <= record.msg.extra["pipeline_elapsed_s"] < 50
