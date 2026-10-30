"""Names the code that holds the validator's event loop while the loop is saturated.

The validator runs the cycle's wave and the express lane on one event loop. For 4-5 min after each
wave send the process sits at a full CPU, and a new node's express verification started then
waits on every await (a Redis read that takes 0.1 s otherwise takes 15-40 s), which turns a 3-min
listing into 6-8 min. No limit is shared; the loop itself is busy. A daemon thread samples the
loop thread's stack a few times a second and, for each minute the loop was mostly busy, logs the
hottest frames, so the code that saturates it can be found and fixed.
"""

import logging
import os
import sys
import sysconfig
import threading
from collections import Counter
from types import FrameType

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

LOOP_BUSY_EVENT = "[loop] Event loop busy; hottest frames"
SAMPLE_SECONDS = 0.2
REPORT_SECONDS = 60
# a minute below this is the loop's normal load and logs nothing
BUSY_FRACTION_TO_REPORT = 0.5
TOP_FRAMES = 10

_LIBRARY_PREFIXES = tuple(
    {
        sysconfig.get_paths()["stdlib"],
        sysconfig.get_paths()["purelib"],
        sysconfig.get_paths()["platlib"],
    }
)


def _where(frame: FrameType) -> str:
    return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_code.co_name}:{frame.f_lineno}"


def _is_idle(frame: FrameType) -> bool:
    # an idle loop waits for I/O in selectors (epoll); a busy one is anywhere else
    return frame.f_code.co_name == "select" and frame.f_code.co_filename.endswith("selectors.py")


def _own_frame(frame: FrameType) -> FrameType | None:
    """The innermost frame of the validator's own code, below any library frames."""
    current: FrameType | None = frame
    while current is not None:
        filename = current.f_code.co_filename
        if "site-packages" not in filename and not filename.startswith(_LIBRARY_PREFIXES):
            return current
        current = current.f_back
    return None


class LoopSampler:
    def __init__(self) -> None:
        self.samples = 0
        self.busy = 0
        self.innermost: Counter[str] = Counter()
        self.own: Counter[str] = Counter()

    def add(self, frame: FrameType) -> None:
        self.samples += 1
        if _is_idle(frame):
            return
        self.busy += 1
        self.innermost[_where(frame)] += 1
        own = _own_frame(frame)
        if own is not None:
            self.own[_where(own)] += 1

    def report(self) -> dict | None:
        """The window's summary when the loop was mostly busy, else None; starts a new window."""
        summary = None
        if self.samples and self.busy / self.samples >= BUSY_FRACTION_TO_REPORT:
            summary = {
                "busy_fraction": round(self.busy / self.samples, 2),
                "samples": self.samples,
                "innermost": self.innermost.most_common(TOP_FRAMES),
                "own_code": self.own.most_common(TOP_FRAMES),
            }
        self.__init__()
        return summary


def _run(loop_thread_id: int, stop: threading.Event) -> None:
    sampler = LoopSampler()
    per_report = max(1, round(REPORT_SECONDS / SAMPLE_SECONDS))
    while not stop.wait(SAMPLE_SECONDS):
        frame = sys._current_frames().get(loop_thread_id)
        if frame is None:
            return
        sampler.add(frame)
        del frame
        if sampler.samples >= per_report:
            summary = sampler.report()
            if summary is not None:
                logger.warning(_m(LOOP_BUSY_EVENT, extra=get_extra_info(summary)))


def start_loop_sampler() -> threading.Event:
    """Samples the calling thread (the event loop's) until the returned event is set."""
    stop = threading.Event()
    threading.Thread(
        target=_run, args=(threading.get_ident(), stop), name="loop-sampler", daemon=True
    ).start()
    return stop
