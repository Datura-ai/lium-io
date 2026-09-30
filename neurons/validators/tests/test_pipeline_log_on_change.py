"""Routine check outcomes log at INFO when they change and at DEBUG when they repeat.

Each check emits one event per executor per cycle, and most repeat the previous cycle. With a
StatusChangeTracker the sink keeps a changed outcome at INFO and moves a repeat to DEBUG; WARNING
and ERROR lines and the run's last event (the step summary) keep their level on every cycle.
A repeat still writes its step duration at INFO as one compact line for the step-duration panels.
"""

import io
import json
import logging
from datetime import UTC, datetime
from functools import partial
from logging import DEBUG, ERROR, INFO, WARNING
from unittest.mock import MagicMock

import pytest

from protocol.vc_protocol.validator_requests import ValidationEvent
from services.task import pipeline_factory as pipeline_factory_module
from services.task.pipeline import (
    STEP_DURATION_LOGGER,
    LoggerSink,
    StatusChangeTracker,
    summarize_steps,
)
from services.task.pipeline_factory import PipelineFactory

LOGGER = "test.sink.on_change"
CHECK = "executor.validate.sysbox_required"
PORTS = "executor.validate.port_connectivity"
FINAL = summarize_steps([("a", 100)], 100)


def _event(
    reason_code: str = "SYSBOX_REQUIRED_OK",
    severity: str = "info",
    *,
    executor_uuid: str | None = "exec-1",
    miner_hotkey: str = "miner-a",
    check_id: str | None = CHECK,
    what_we_saw: dict | None = None,
    ms: int | None = None,
) -> ValidationEvent:
    context = {"executor_uuid": executor_uuid, "miner_hotkey": miner_hotkey} if executor_uuid else {}
    if ms is not None:
        context["execution_time_ms"] = ms
    return ValidationEvent(
        event=f"event {reason_code}",
        reason_code=reason_code,
        severity=severity,
        impact="x",
        check_id=check_id,
        what_we_saw=dict(what_we_saw or {}),
        context=context,
        when=datetime(2026, 9, 28, tzinfo=UTC),
    )


OK, SKIPPED = _event, partial(_event, "SYSBOX_REQUIRED_SKIPPED_RENTED")
MISSING = partial(_event, "SYSBOX_REQUIRED_MISSING", "warning")
PORT_FAILED = partial(_event, "PORT_VERIFICATION_FAILED", "error", check_id=PORTS)
COLLATERAL = partial(_event, "COLLATERAL_MISSING", "warning")
DONE = partial(_event, "VALIDATION_COMPLETED", check_id="x.finalize", what_we_saw=FINAL)


@pytest.fixture
def durations(monkeypatch):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    duration_logger = logging.getLogger(STEP_DURATION_LOGGER)
    duration_logger.addHandler(handler)
    monkeypatch.setattr(duration_logger, "propagate", False)
    yield lambda: [json.loads(line) for line in stream.getvalue().splitlines()]
    duration_logger.removeHandler(handler)


async def _levels(caplog, sink: LoggerSink, events) -> list[int]:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    caplog.clear()
    for event in events:
        await sink.emit(event)
    return [r.levelno for r in caplog.records]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "expected", "tracked"),
    [
        pytest.param([OK(), OK(what_we_saw={"ports": 3}), OK()], [INFO, DEBUG, DEBUG], True, id="repeat"),
        pytest.param([OK(), SKIPPED(), SKIPPED(), OK()], [INFO, INFO, DEBUG, INFO], True, id="change"),
        pytest.param(
            [MISSING(), MISSING(), PORT_FAILED(), PORT_FAILED()],
            [WARNING, WARNING, ERROR, ERROR],
            True, id="warn-error",
        ),
        pytest.param([OK(), MISSING(), OK()], [INFO, WARNING, INFO], True, id="recovery"),
        pytest.param([COLLATERAL(), COLLATERAL()], [INFO, DEBUG], True, id="provider-state"),
        pytest.param(
            [OK(), OK(executor_uuid="exec-2"), OK(check_id="gpu"), OK(executor_uuid="exec-2")],
            [INFO, INFO, INFO, DEBUG],
            True, id="per-executor-and-check",
        ),
        pytest.param([DONE(), DONE()], [INFO, INFO], True, id="run-summary"),
        pytest.param(
            [OK(executor_uuid=None), OK(executor_uuid=None), OK(check_id=None), OK(check_id=None)],
            [INFO] * 4,
            True,
            id="no-identity",
        ),
        pytest.param(
            [OK(), OK(miner_hotkey="miner-b"), OK(miner_hotkey="miner-b"), OK()],
            [INFO, INFO, DEBUG, DEBUG],
            True,
            id="same-uuid-two-miners",
        ),
        pytest.param([OK(), OK()], [INFO, INFO], False, id="no-tracker"),
    ],
)
async def test_sink_levels(caplog, durations, events, expected, tracked):
    sink = LoggerSink(logging.getLogger(LOGGER), tracker=StatusChangeTracker() if tracked else None)

    assert await _levels(caplog, sink, events) == expected
    if events[-1].reason_code == "COLLATERAL_MISSING":
        assert caplog.records[-1].msg.extra["reason"] == "provider_state"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("level", "events", "expected_ms"),
    [
        pytest.param(INFO, [OK(ms=120), OK(ms=95), OK(ms=101)], [95, 101], id="repeat-at-info"),
        pytest.param(
            INFO, [MISSING(ms=120), MISSING(ms=120), OK(ms=120), OK()], [], id="full-line"
        ),
        pytest.param(DEBUG, [OK(ms=120), OK(ms=95)], [], id="debug-on"),
    ],
)
async def test_repeat_keeps_its_step_duration_at_info(
    caplog, durations, level, events, expected_ms
):
    caplog.set_level(level, logger=LOGGER)
    sink = LoggerSink(logging.getLogger(LOGGER), tracker=StatusChangeTracker())
    for event in events:
        await sink.emit(event)

    assert durations() == [
        {
            "level": "INFO",
            "logger": STEP_DURATION_LOGGER,
            "message": "Check step duration",
            "extra": {"check_id": CHECK, "context": {"execution_time_ms": ms}},
        }
        for ms in expected_ms
    ]
    if level == DEBUG:
        assert [r.levelno for r in caplog.records] == [INFO, DEBUG]
        assert caplog.records[-1].msg.extra["context"]["execution_time_ms"] == 95


def test_pipelines_from_one_factory_share_the_tracker(monkeypatch):
    monkeypatch.setattr(pipeline_factory_module, "InspectorValidationService", MagicMock)
    factory = PipelineFactory(*(MagicMock() for _ in range(8)))
    first, second = factory.build_pipeline([]), factory.build_pipeline([])

    assert isinstance(factory.status_tracker, StatusChangeTracker)
    assert first.sink.tracker is second.sink.tracker is factory.status_tracker


def test_tracker_forgets_the_oldest_entry_past_its_bound():
    tracker = StatusChangeTracker(max_entries=2)
    outcome = ("e", "R", "info")

    assert [tracker.changed("miner-a", e, CHECK, outcome) for e in "abcca"] == [True, True, True, False, True]
