"""Replay protection: a persisted nonce set inside the freshness window, plus a startup floor.

A request is accepted only if all three hold:

  1. `|now - timestamp| <= skew`
  2. `timestamp > startup_floor`
  3. `(hotkey, nonce)` has not been seen

**The write ordering is the control.** `check_and_record` fsyncs the nonce before it returns, and
the middleware calls it before dispatching to a handler. Recording afterwards would leave a crash
window in which an already-executed request is still replayable across a restart — the exact hole
this exists to close.

**The startup floor does not advance during a process lifetime.** It is read once at startup from
the highest timestamp the previous life accepted, and stays fixed. That distinction is the whole
design: a floor that advanced per request would be a strict monotonic timestamp, which rejects the
second of any two concurrent requests and every client retry. The platform key does issue
concurrent requests (a `/v1/state` poll interleaved with a `DELETE`), so monotonicity would read
as an intermittent auth bug and the pressure would be to weaken the gate. See test_replay.py,
which asserts the availability property directly.

The two records live in separate files on purpose. If the nonce set is lost or corrupt, the floor
alone still refuses every request the previous life accepted; losing both is a root-only action,
i.e. the host owner, who is outside this layer's threat model by design (arch doc §03).

**The floor can sit ahead of wall-clock time, and that is not a bug.** The skew window is
two-sided, so a timestamp up to `skew_seconds` in the future is valid; accepting one makes it the
high-water mark, and the next start refuses present-time requests until wall-clock passes it —
up to `skew_seconds` of refusal after a restart, self-healing, no intervention. Measured on
hardware during the DAH-2575 acceptance run and pinned by
test_replay.py::TestRestartDurability::test_an_in_skew_future_timestamp_holds_the_floor_ahead_of_wall_clock.

Clamping the recorded floor to `now` would remove that window and is the obvious fix, but it
also republishes the accepted future-dated request: after a restart its timestamp is above the
floor again, leaving only the nonce set to refuse it — and the floor exists precisely to be the
backstop for a lost or corrupt nonce set. Narrowing the window to one side instead would start
rejecting any client whose clock runs fast. Both are protocol changes affecting the DAH-2576 and
DAH-2580 clients, so the behaviour is documented and pinned here rather than changed in passing.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from cvmd.atomic import read_json, write_json_durable

NONCES_FILENAME = "replay_nonces.json"
FLOOR_FILENAME = "replay_floor.json"
SCHEMA_VERSION = 1

NANOSECONDS_PER_SECOND = 1_000_000_000


class ReplayRejected(Exception):
    """The request failed the freshness, floor, or nonce check. The caller turns this into a 401.

    The message carries the specific reason for the journal; it is not sent to the client.
    """


@dataclass(frozen=True)
class _SeenNonce:
    hotkey: str
    nonce: str
    timestamp_ns: int


class ReplayStore:
    def __init__(self, state_dir: Path, skew_seconds: int) -> None:
        self._nonces_path = state_dir / NONCES_FILENAME
        self._floor_path = state_dir / FLOOR_FILENAME
        self._skew_ns = skew_seconds * NANOSECONDS_PER_SECOND
        self._lock = asyncio.Lock()

        self._seen: dict[tuple[str, str], int] = _load_seen(self._nonces_path)
        self._highest_accepted_ns = _load_floor(self._floor_path)

        # Captured once, deliberately never updated below. See the module docstring.
        self._startup_floor_ns = self._highest_accepted_ns

    @property
    def startup_floor_ns(self) -> int:
        return self._startup_floor_ns

    async def check_and_record(self, hotkey: str, nonce: str, timestamp_ns: int) -> None:
        """Accept the request and durably record it, or raise ReplayRejected.

        On return, the nonce is on disk. The caller must not dispatch to a handler before this.
        """
        async with self._lock:
            now_ns = time.time_ns()

            drift_ns = abs(now_ns - timestamp_ns)
            if drift_ns > self._skew_ns:
                raise ReplayRejected(
                    f"timestamp outside the {self._skew_ns // NANOSECONDS_PER_SECOND}s window "
                    f"(drift {drift_ns / NANOSECONDS_PER_SECOND:.3f}s)"
                )

            if timestamp_ns <= self._startup_floor_ns:
                raise ReplayRejected(
                    f"timestamp {timestamp_ns} is at or below the startup floor "
                    f"{self._startup_floor_ns}"
                )

            key = (hotkey, nonce)
            if key in self._seen:
                raise ReplayRejected(f"nonce already used by {hotkey}")

            self._seen[key] = timestamp_ns
            self._prune(now_ns)
            write_json_durable(self._nonces_path, _encode_seen(self._seen))

            # Only persisted, never applied to this process. The next start reads it as its floor.
            if timestamp_ns > self._highest_accepted_ns:
                self._highest_accepted_ns = timestamp_ns
                write_json_durable(
                    self._floor_path,
                    {"version": SCHEMA_VERSION, "highest_accepted_ns": timestamp_ns},
                )

    def _prune(self, now_ns: int) -> None:
        """Drop entries that the freshness window would reject anyway.

        Keeps the set at a few seconds of traffic — it is a window, not a log.
        """
        cutoff = now_ns - self._skew_ns
        self._seen = {key: ts for key, ts in self._seen.items() if ts >= cutoff}


def _load_seen(path: Path) -> dict[tuple[str, str], int]:
    raw = read_json(path)
    if not isinstance(raw, dict) or raw.get("version") != SCHEMA_VERSION:
        return {}

    entries = raw.get("seen")
    if not isinstance(entries, list):
        return {}

    seen: dict[tuple[str, str], int] = {}
    for entry in entries:
        # A malformed entry is skipped rather than fatal: the floor is the backstop, and refusing
        # to start on a corrupt window file would turn a recoverable state into an outage.
        if not isinstance(entry, list) or len(entry) != 3:
            continue
        hotkey, nonce, timestamp_ns = entry
        if isinstance(hotkey, str) and isinstance(nonce, str) and isinstance(timestamp_ns, int):
            seen[(hotkey, nonce)] = timestamp_ns
    return seen


def _encode_seen(seen: dict[tuple[str, str], int]) -> dict:
    return {
        "version": SCHEMA_VERSION,
        "seen": [[hotkey, nonce, ts] for (hotkey, nonce), ts in sorted(seen.items())],
    }


def _load_floor(path: Path) -> int:
    raw = read_json(path)
    if not isinstance(raw, dict) or raw.get("version") != SCHEMA_VERSION:
        return 0
    value = raw.get("highest_accepted_ns")
    return value if isinstance(value, int) and value > 0 else 0
