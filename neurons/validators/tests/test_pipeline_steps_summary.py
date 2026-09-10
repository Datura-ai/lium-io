"""DAH-3012: the last pipeline event carries a per-step duration summary, so the provider-facing
`last_validation` (backend `log_text` -> portal `what_we_saw`) and one Loki line show what a run
spent its time on and where it stopped."""

from datetime import UTC, datetime

import pytest

from neurons.validators.src.services.task import pipeline as pipeline_module
from neurons.validators.src.services.task.models import ValidationEvent
from neurons.validators.src.services.task.pipeline import (
    STEP_SUMMARY_MIN_MS,
    CheckResult,
    Pipeline,
    summarize_steps,
)


class _Clock:
    """Replaces time.perf_counter so a check can take exactly the seconds it declares."""

    def __init__(self):
        self.now = 1000.0

    def perf_counter(self) -> float:
        return self.now


class _TimedCheck:
    def __init__(self, check_id: str, seconds: float, clock: _Clock, *, passed=True, fatal=True, halt=False):
        self.check_id = check_id
        self.fatal = fatal
        self._seconds = seconds
        self._clock = clock
        self._passed = passed
        self._halt = halt

    async def run(self, ctx) -> CheckResult:
        self._clock.now += self._seconds
        event = ValidationEvent(
            event=f"{self.check_id} ran",
            reason_code="TEST",
            severity="info" if self._passed else "error",
            impact="",
            check_id=self.check_id,
            when=datetime.now(UTC),
        )
        return CheckResult(passed=self._passed, event=event, halt=self._halt)


class _Sink:
    def __init__(self):
        self.emitted: list[ValidationEvent] = []

    async def emit(self, event: ValidationEvent) -> None:
        self.emitted.append(event)


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(pipeline_module.time, "perf_counter", clock.perf_counter)
    return clock


@pytest.mark.asyncio
async def test_passing_run_lists_the_slow_steps_and_the_total_on_the_last_event(clock, context_factory):
    checks = [
        _TimedCheck("prep.start_gpu_monitor", 0.2, clock),
        _TimedCheck("gpu.scrape.machine_spec", 21.04, clock),
        _TimedCheck("gpu.validate.count", 0.0, clock),
        _TimedCheck("gpu.validate.verifyx", 78.5, clock),
        _TimedCheck("gpu.validate.capability", 26.2, clock),
        _TimedCheck("pipeline.finalize", 0.0, clock),
    ]
    sink = _Sink()

    ok, events, _ = await Pipeline(checks, sink).run(context_factory())

    assert ok is True
    last = events[-1].what_we_saw
    assert last["steps"] == {
        "gpu.scrape.machine_spec": 21.0,
        "gpu.validate.verifyx": 78.5,
        "gpu.validate.capability": 26.2,
    }
    assert last["steps_total_s"] == pytest.approx(125.9, abs=0.11)
    assert "steps_failed" not in last
    # Only the last event carries the summary; the others are as before.
    assert all("steps" not in event.what_we_saw for event in events[:-1])
    # The summary is on the event the sink saw, i.e. the Loki line, not added afterwards.
    assert sink.emitted[-1] is events[-1]
    assert "steps" in sink.emitted[-1].what_we_saw
    # Per-event timing fields are unchanged.
    assert events[3].context["execution_time_ms"] == 78500
    assert events[3].context["elapsed_time_ms"] == pytest.approx(99740, abs=10)


@pytest.mark.asyncio
async def test_fatal_stop_names_the_failed_step_on_the_event_the_provider_sees(clock, context_factory):
    checks = [
        _TimedCheck("gpu.scrape.machine_spec", 8.0, clock),
        _TimedCheck("executor.validate.duplicate", 31.0, clock),
        _TimedCheck("gpu.validate.verifyx", 164.6, clock, passed=False),
        _TimedCheck("gpu.validate.capability", 26.0, clock),  # never runs
    ]

    ok, events, _ = await Pipeline(checks, _Sink()).run(context_factory())

    assert ok is False
    assert len(events) == 3
    last = events[-1].what_we_saw
    assert last["steps_failed"] == "gpu.validate.verifyx"
    assert last["steps"] == {
        "gpu.scrape.machine_spec": 8.0,
        "executor.validate.duplicate": 31.0,
        "gpu.validate.verifyx": 164.6,
    }
    assert last["steps_total_s"] == pytest.approx(203.6, abs=0.11)


@pytest.mark.asyncio
async def test_non_fatal_failure_does_not_stop_and_is_not_named(clock, context_factory):
    checks = [
        _TimedCheck("host.validate.cpu_truth", 1.5, clock, passed=False, fatal=False),
        _TimedCheck("pipeline.finalize", 0.0, clock),
    ]

    ok, events, _ = await Pipeline(checks, _Sink()).run(context_factory())

    assert ok is True
    assert "steps_failed" not in events[-1].what_we_saw
    assert events[-1].what_we_saw["steps"] == {"host.validate.cpu_truth": 1.5}


@pytest.mark.asyncio
async def test_halt_is_a_final_event_too(clock, context_factory):
    checks = [
        _TimedCheck("gpu.scrape.machine_spec", 12.0, clock),
        _TimedCheck("executor.validate.rented_state", 2.0, clock, halt=True),
        _TimedCheck("gpu.validate.verifyx", 80.0, clock),  # never runs on a rented box
    ]

    ok, events, _ = await Pipeline(checks, _Sink()).run(context_factory())

    assert ok is True
    assert len(events) == 2
    assert events[-1].what_we_saw["steps"] == {
        "gpu.scrape.machine_spec": 12.0,
        "executor.validate.rented_state": 2.0,
    }
    assert "steps_failed" not in events[-1].what_we_saw


def test_summary_omits_sub_second_steps_and_rounds_to_a_tenth():
    steps = [("a", STEP_SUMMARY_MIN_MS - 1), ("b", STEP_SUMMARY_MIN_MS), ("c", 73_090)]

    summary = summarize_steps(steps, elapsed_time_ms=152_345)

    assert summary == {"steps": {"b": 1.0, "c": 73.1}, "steps_total_s": 152.3}
    assert summarize_steps(steps, 1, failed_check_id="c")["steps_failed"] == "c"
