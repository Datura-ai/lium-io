import selectors
import sys
import threading

from core.loop_sampler import LoopSampler

from core import loop_sampler


def _frame_of(fn):
    """The frame of fn while it runs, taken from inside it."""
    captured = {}

    def grab():
        captured["frame"] = sys._getframe(1)

    fn(grab)
    return captured["frame"]


def _validator_code(grab):
    grab()


def test_a_busy_window_reports_the_hottest_own_code_and_an_idle_one_reports_nothing():
    busy = _frame_of(_validator_code)
    sampler = LoopSampler()
    for _ in range(9):
        sampler.add(busy)
    idle = type("Code", (), {"co_name": "select", "co_filename": selectors.__file__})
    sampler.add(type("Frame", (), {"f_code": idle, "f_lineno": 1, "f_back": None})())

    summary = sampler.report()

    assert summary["busy_fraction"] == 0.9
    assert summary["samples"] == 10
    assert summary["own_code"][0][0].startswith("test_loop_sampler.py:_validator_code:")
    assert summary["own_code"][0][1] == 9
    # the report starts a new window; a window that is mostly idle logs nothing
    sampler.add(type("Frame", (), {"f_code": idle, "f_lineno": 1, "f_back": None})())
    assert sampler.report() is None


def test_the_sampler_thread_stops_when_asked(monkeypatch):
    monkeypatch.setattr(loop_sampler, "SAMPLE_SECONDS", 0.01)
    stop = loop_sampler.start_loop_sampler()
    stop.set()
    for thread in threading.enumerate():
        if thread.name == "loop-sampler":
            thread.join(timeout=2)
            assert not thread.is_alive()
