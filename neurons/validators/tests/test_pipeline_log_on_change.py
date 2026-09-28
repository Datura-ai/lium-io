"""Routine check outcomes log at INFO when they change and at DEBUG when they repeat.

Each check emits one event per executor per cycle, and most repeat the previous cycle. With a
StatusChangeTracker the sink keeps a changed outcome at INFO and moves a repeat to DEBUG; WARNING
and ERROR lines and the run's last event (the step summary) keep their level on every cycle.
A repeat still writes its step duration at INFO as one compact line for the step-duration panels.
"""

import json
import logging
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from protocol.vc_protocol.validator_requests import ValidationEvent
from services.task import pipeline_factory as pipeline_factory_module
from services.task.pipeline import LoggerSink, StatusChangeTracker
from services.task.pipeline_factory import PipelineFactory

LOGGER = "test.sink.on_change"
CHECK = "executor.validate.sysbox_required"


def _event(
    reason_code: str = "SYSBOX_REQUIRED_OK",
    severity: str = "info",
    *,
    executor_uuid: str | None = "exec-1",
    check_id: str | None = CHECK,
    event: str | None = None,
    what_we_saw: dict | None = None,
) -> ValidationEvent:
    context = {"executor_uuid": executor_uuid} if executor_uuid else {}
    return ValidationEvent(
        event=event or f"event {reason_code}",
        reason_code=reason_code,
        severity=severity,
        impact="x",
        check_id=check_id,
        what_we_saw=what_we_saw or {},
        context=context,
        when=datetime(2026, 9, 28, tzinfo=UTC),
    )


async def _levels(caplog, sink: LoggerSink, *events: ValidationEvent) -> list[int]:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    caplog.clear()
    for event in events:
        await sink.emit(event)
    return [r.levelno for r in caplog.records]


@pytest.fixture
def sink() -> LoggerSink:
    return LoggerSink(logging.getLogger(LOGGER), tracker=StatusChangeTracker())


@pytest.mark.asyncio
async def test_unchanged_status_logs_at_debug(caplog, sink):
    levels = await _levels(caplog, sink, _event(), _event(what_we_saw={"ports": 3}), _event())

    assert levels == [logging.INFO, logging.DEBUG, logging.DEBUG]


@pytest.mark.asyncio
async def test_changed_status_logs_at_info(caplog, sink):
    levels = await _levels(
        caplog,
        sink,
        _event("SYSBOX_REQUIRED_OK"),
        _event("SYSBOX_REQUIRED_SKIPPED_RENTED"),
        _event("SYSBOX_REQUIRED_SKIPPED_RENTED"),
        _event("SYSBOX_REQUIRED_OK"),
    )

    assert levels == [logging.INFO, logging.INFO, logging.DEBUG, logging.INFO]


@pytest.mark.asyncio
async def test_warning_and_error_keep_their_level_when_repeated(caplog, sink):
    levels = await _levels(
        caplog,
        sink,
        _event("SYSBOX_REQUIRED_MISSING", "warning"),
        _event("SYSBOX_REQUIRED_MISSING", "warning"),
        _event("PORT_VERIFICATION_FAILED", "error", check_id="executor.validate.port_connectivity"),
        _event("PORT_VERIFICATION_FAILED", "error", check_id="executor.validate.port_connectivity"),
    )

    assert levels == [logging.WARNING, logging.WARNING, logging.ERROR, logging.ERROR]


@pytest.mark.asyncio
async def test_recovery_from_a_warning_logs_at_info(caplog, sink):
    levels = await _levels(
        caplog,
        sink,
        _event("SYSBOX_REQUIRED_OK"),
        _event("SYSBOX_REQUIRED_MISSING", "warning"),
        _event("SYSBOX_REQUIRED_OK"),
    )

    assert levels == [logging.INFO, logging.WARNING, logging.INFO]


@pytest.mark.asyncio
async def test_repeated_provider_state_verdict_logs_at_debug(caplog, sink):
    levels = await _levels(
        caplog, sink, _event("COLLATERAL_MISSING", "warning"), _event("COLLATERAL_MISSING", "warning")
    )

    assert levels == [logging.INFO, logging.DEBUG]
    assert caplog.records[1].msg.extra["reason"] == "provider_state"


@pytest.mark.asyncio
async def test_executors_and_checks_are_tracked_separately(caplog, sink):
    levels = await _levels(
        caplog,
        sink,
        _event(executor_uuid="exec-1"),
        _event(executor_uuid="exec-2"),
        _event(executor_uuid="exec-1", check_id="executor.validate.gpu_usage"),
        _event(executor_uuid="exec-2"),
    )

    assert levels == [logging.INFO, logging.INFO, logging.INFO, logging.DEBUG]


@pytest.mark.asyncio
async def test_last_event_of_the_run_stays_at_info(caplog, sink):
    summary = {"steps": {"a": 0.1}, "steps_total_s": 0.1}
    final = "executor.validate.finalize"
    levels = await _levels(
        caplog,
        sink,
        _event("VALIDATION_COMPLETED", check_id=final, what_we_saw=dict(summary)),
        _event("VALIDATION_COMPLETED", check_id=final, what_we_saw=dict(summary)),
    )

    assert levels == [logging.INFO, logging.INFO]


@pytest.mark.asyncio
async def test_event_without_executor_or_check_is_never_demoted(caplog, sink):
    levels = await _levels(
        caplog,
        sink,
        _event(executor_uuid=None),
        _event(executor_uuid=None),
        _event(check_id=None),
        _event(check_id=None),
    )

    assert levels == [logging.INFO] * 4


@pytest.mark.asyncio
async def test_sink_without_tracker_logs_every_event_at_info(caplog):
    levels = await _levels(caplog, LoggerSink(logging.getLogger(LOGGER)), _event(), _event())

    assert levels == [logging.INFO, logging.INFO]


@pytest.mark.asyncio
async def test_pipelines_from_one_factory_share_the_tracker(caplog, monkeypatch):
    monkeypatch.setattr(pipeline_factory_module, "InspectorValidationService", MagicMock)
    factory = PipelineFactory(*(MagicMock() for _ in range(8)))
    first, second = factory.build_pipeline([]), factory.build_pipeline([])

    assert first.sink.tracker is second.sink.tracker is factory.status_tracker

    caplog.set_level(logging.DEBUG, logger=pipeline_factory_module.logger.name)
    caplog.clear()
    await first.sink.emit(_event())
    await second.sink.emit(_event())

    assert [r.levelno for r in caplog.records] == [logging.INFO, logging.DEBUG]


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


@pytest.fixture
def durations():
    duration_logger = logging.getLogger("test.sink.step_duration")
    duration_logger.propagate = False
    handler = _Lines()
    handler.setFormatter(logging.Formatter("%(message)s"))
    duration_logger.addHandler(handler)
    yield duration_logger, handler.lines
    duration_logger.removeHandler(handler)


def _timed(ms: int, reason_code: str = "SYSBOX_REQUIRED_OK", severity: str = "info") -> ValidationEvent:
    event = _event(reason_code, severity)
    event.context["execution_time_ms"] = ms
    return event


async def _emit_at_info(sink: LoggerSink, *events: ValidationEvent) -> None:
    sink.logger.setLevel(logging.INFO)
    try:
        for event in events:
            await sink.emit(event)
    finally:
        sink.logger.setLevel(logging.NOTSET)


@pytest.mark.asyncio
async def test_repeat_keeps_its_step_duration_at_info(durations):
    duration_logger, lines = durations
    sink = LoggerSink(logging.getLogger(LOGGER), tracker=StatusChangeTracker(), duration_logger=duration_logger)

    await _emit_at_info(sink, _timed(120), _timed(95), _timed(101))

    assert [json.loads(line) for line in lines] == [
        {
            "level": "INFO",
            "logger": "services.task.step_duration",
            "message": "Check step duration",
            "extra": {"check_id": CHECK, "context": {"execution_time_ms": ms}},
        }
        for ms in (95, 101)
    ]
    assert all("execution_time_ms" in line and len(line) < 200 for line in lines)


@pytest.mark.asyncio
async def test_no_step_duration_line_when_the_full_line_is_logged(durations):
    duration_logger, lines = durations
    sink = LoggerSink(logging.getLogger(LOGGER), tracker=StatusChangeTracker(), duration_logger=duration_logger)

    await _emit_at_info(
        sink,
        _timed(120, "SYSBOX_REQUIRED_MISSING", "warning"),
        _timed(120, "SYSBOX_REQUIRED_MISSING", "warning"),
        _timed(120, "SYSBOX_REQUIRED_OK"),
        _event(),
    )

    assert lines == []


@pytest.mark.asyncio
async def test_no_step_duration_line_when_debug_is_on(caplog, durations):
    duration_logger, lines = durations
    sink = LoggerSink(logging.getLogger(LOGGER), tracker=StatusChangeTracker(), duration_logger=duration_logger)

    levels = await _levels(caplog, sink, _timed(120), _timed(95))

    assert levels == [logging.INFO, logging.DEBUG]
    assert caplog.records[1].msg.extra["context"]["execution_time_ms"] == 95
    assert lines == []


def test_tracker_forgets_the_oldest_entry_past_its_bound():
    tracker = StatusChangeTracker(max_entries=2)
    outcome = ("e", "R", "info")

    assert tracker.changed("a", CHECK, outcome)
    assert tracker.changed("b", CHECK, outcome)
    assert tracker.changed("c", CHECK, outcome)
    assert not tracker.changed("c", CHECK, outcome)
    assert tracker.changed("a", CHECK, outcome)
