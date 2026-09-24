from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import shlex
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import RegistryPullMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .outbound_internet import _is_rented

logger = logging.getLogger(__name__)

# library/hello-world's multi-arch index, pinned by digest (linux/amd64: one 2,415-byte layer). A digest
# never changes what it names, so every node pulls the same bytes, and a tag moving upstream cannot turn
# into a fleet-wide manifest_unknown. Unqualified `docker.io`, so dockerd sends it through
# the registry-mirrors in daemon.json first, the way it sends a renter's template image.
REGISTRY_PULL_IMAGE = "docker.io/library/hello-world@sha256:5e23090353324d887c48ad5e5c56d294eab81588df9605b07d1afe895f9cc8f8"
# a healthy pull of this image takes 1-4 s. Behind a mirror whose DNS lookups time out, dockerd waits out
# the lookup before each registry request and then falls back to Docker Hub: measured 51 s and over 60 s
# on two runs of the same setup, so a 60 s bound would pass the broken mirror about half the time
REGISTRY_PULL_TIMEOUT_SECONDS = 30
REGISTRY_PULL_MARKER = "lium_pull"
# the script's own bounds (10 + 10 + 20 + 10 + 30 + 20 s, each plus timeout's 5 s kill grace) and SSH
REGISTRY_PULL_COMMAND_TIMEOUT_SECONDS = 150
# a node fails once this many measured pulls in a row failed; a no-verdict pull neither counts nor resets
REGISTRY_PULL_FAILURES_BEFORE_VERDICT = 2
_REDIS_PREFIX = "registry_pull_probe"
_REDIS_TTL_SECONDS = 7 * 24 * 3600
_TAIL_CHARS = 600

# the validator's own reading of Docker Hub, taken before a failed pull counts: if the validator cannot reach
# it either, the node's failure says nothing about the node. One fetch per window, shared by every node
DOCKER_HUB_CONTROL_URL = "https://registry-1.docker.io/v2/"
# /v2/ answers 401 to an anonymous client and 200 to an authenticated one; anything else is not a working hub
_HUB_REACHABLE_STATUSES = frozenset({200, 401})
_HUB_CONTROL_TIMEOUT_SECONDS = 10
_HUB_CONTROL_KEY = "registry_pull_hub_control"
HUB_CONTROL_TTL_SECONDS = 5 * 60
# this validator's last scheduled pull of each node: {"at": ..., "failed": ..., "miner": <hotkey>}. Entries
# older than one interval plus FLEET_HORIZON_SLACK_SECONDS are pruned; the ones younger are the idle fleet
# this validator has seen
_FLEET_KEY = "registry_pull_fleet"
FLEET_HORIZON_SLACK_SECONDS = 3600
# `required`: this share of the idle nodes seen, or REGISTRY_PULL_FLEET_MIN_PULLS if more
FLEET_MIN_PULLS_SHARE_OF_SEEN = 0.1
# REGISTRY_PULL_FLEET_FAILING is logged at most once per this long
FLEET_LOG_INTERVAL_SECONDS = 3600
_FLEET_LOGGED_KEY = "registry_pull_fleet_logged"
# every Redis read and write of the fleet window and the control runs under this lock, so one fetch comes
# out per window, and a pull recorded while the window is pruned is kept
_FLEET_LOCK = asyncio.Lock()

# why a failed pull did not count
NoVerdict = Literal["docker_hub_down", "fleet_failing", "fleet_state_unreadable", "awaiting_fleet"]

# POSIX sh, run by the executor container's docker CLI against the host's dockerd, so the pull takes the
# node's configured registry path (registry-mirrors, then registry-1.docker.io) exactly as a rental's pull
# does. The image is removed first so the pull reaches the registry or mirror instead of the local store,
# and removed again after. Every docker call runs under `timeout` where the image has it; the pull's
# 30 s is the verdict's bound. Exit 0 always: the marker lines are the answer.
# The first line carries the image store next to the mirrors: `.Driver` is overlayfs on the containerd store
# (Docker 29's default for a new install) and overlay2 on the classic store a host upgraded from 28 keeps.
# The two pay a mirror's DNS wait differently (see REGISTRY_PULL_TIMEOUT_SECONDS and _IMAGE_STORES).
REGISTRY_PULL_SCRIPT = (
    f"img={REGISTRY_PULL_IMAGE}; d=/usr/bin/docker; "
    'bounded() { s=$1; shift; if command -v timeout >/dev/null 2>&1; then timeout -k 5 "$s" "$@"; '
    'else "$@"; fi; }; '
    "info=$(bounded 10 $d info --format 'driver={{.Driver}} mirrors={{json .RegistryConfig.Mirrors}}' "
    "2>/dev/null | head -c 500); "
    f'echo "{REGISTRY_PULL_MARKER} ${{info:-mirrors=unknown}}"; '
    'if bounded 10 $d image inspect "$img" >/dev/null 2>&1; then cached=yes; else cached=no; fi; '
    'bounded 20 $d rmi -f "$img" >/dev/null 2>&1; '
    'if bounded 10 $d image inspect "$img" >/dev/null 2>&1; then '
    f'echo "{REGISTRY_PULL_MARKER} cached=$cached cache=still_present"; exit 0; fi; '
    f'echo "{REGISTRY_PULL_MARKER} cached=$cached cache=removed"; '
    "start=$(date +%s); "
    f'out=$(bounded {REGISTRY_PULL_TIMEOUT_SECONDS} $d pull -q "$img" 2>&1); rc=$?; '
    f'echo "{REGISTRY_PULL_MARKER} exit=$rc seconds=$(( $(date +%s) - start ))"; '
    f"printf '%s\\n' \"$out\" | tail -c {_TAIL_CHARS}; "
    'bounded 20 $d rmi -f "$img" >/dev/null 2>&1; exit 0'
)

Outcome = Literal[
    "ok",
    "timeout",
    "dns_error",
    "unreachable",
    "manifest_unknown",
    "rate_limited",
    "auth_error",
    "other",
    "not_run",
]
# what fails a node (twice in a row): the registry path is broken, the way it broke 14e704ba's rents.
# `unreachable` is a firewall or proxy that rejects the connection rather than dropping it; a renter's
# uncached pull fails on it just the same
FAILING_OUTCOMES = frozenset({"timeout", "dns_error", "unreachable", "manifest_unknown"})
# no verdict either way: a Docker Hub 429 says the IP's anonymous quota is spent, not that the path is
# broken, and an auth or unclassified error, or a probe that did not run, measured nothing about it
UNMEASURED_OUTCOMES = frozenset({"rate_limited", "auth_error", "other", "not_run"})

# timeout(1)'s exit when it ended the command, and its SIGKILL after the grace period
_TIMEOUT_EXITS = frozenset({124, 137})
_RATE_LIMIT_MARKERS = ("toomanyrequests", "too many requests", "rate limit")
# a bare 429 as a word, never the hex of a digest in the error text
_HTTP_429_RX = re.compile(r"(?<![0-9a-f])429(?![0-9a-f])")
_DNS_MARKERS = (
    "no such host",
    "server misbehaving",
    "temporary failure in name resolution",
    "name or service not known",
    "dial udp",
)
_MANIFEST_MARKERS = ("manifest unknown", "manifest_unknown")
# the containerd image store's wording (Docker 29): `failed to resolve reference "<ref>": <ref>: not found`
_NOT_FOUND_RX = re.compile(r"(failed to resolve reference|manifest for) .*not found")
_AUTH_MARKERS = ("unauthorized", "authentication required", "is denied", "access denied")
# a TCP connect to the registry, or to the daemon's proxy (`proxyconnect tcp: dial tcp ...`), that was
# rejected: a REJECT --reject-with tcp-reset, icmp-host-unreachable or icmp-net-unreachable, or no route.
# Only `dial tcp`: the CLI failing to reach dockerd's own socket is `dial unix ...`, which says nothing
# about the registry path
_UNREACHABLE_RX = re.compile(
    r"dial tcp [^\s]+: connect: (connection refused|no route to host|network is unreachable)"
)
_TIMEOUT_MARKERS = (
    "i/o timeout",
    "tls handshake timeout",
    "context deadline exceeded",
    "client.timeout exceeded",
    "timeout exceeded while awaiting headers",
)


# `docker info --format '{{.Driver}}'` → the image store. On the containerd store a mirror whose DNS lookups
# time out costs 10-20 s per registry request (72 s for hello-world unbounded), so the 30 s bound fails it;
# on the classic store it costs about 20 s once per pull, so the probe passes at about 20 s, and that is
# right: a template pulls in about 22 s there too
_IMAGE_STORES = {"overlayfs": "containerd", "overlay2": "classic"}


@dataclass(frozen=True)
class PullReading:
    """What one run of REGISTRY_PULL_SCRIPT said."""

    outcome: Outcome
    mirrors: list[str] | None = None
    exit_code: int | None = None
    seconds: int | None = None
    cached_before: bool | None = None
    detail: str | None = None
    driver: str | None = None

    @property
    def failed(self) -> bool:
        return self.outcome in FAILING_OUTCOMES

    @property
    def image_store(self) -> str | None:
        return _IMAGE_STORES.get(self.driver or "")

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"outcome": self.outcome, "image": REGISTRY_PULL_IMAGE}
        record["mirrors"] = self.mirrors
        if self.exit_code is not None:
            record["exit_code"] = self.exit_code
        if self.seconds is not None:
            record["seconds"] = self.seconds
        record["driver"] = self.driver
        record["image_store"] = self.image_store
        if self.cached_before is not None:
            record["cached_before"] = self.cached_before
        if self.detail:
            record["detail"] = self.detail[-_TAIL_CHARS:]
        return record


def classify_pull_error(exit_code: int, output: str) -> Outcome:
    """The outcome of a finished `docker pull`: ok on exit 0, else read off the daemon's error text.

    A DNS failure is named before a timeout: a mirror whose lookup times out reads
    `lookup docker.m.daocloud.io on 127.0.0.53:53: read udp ...: i/o timeout` (14e704ba).
    """
    if exit_code == 0:
        return "ok"
    text = (output or "").lower()
    if any(marker in text for marker in _RATE_LIMIT_MARKERS) or _HTTP_429_RX.search(text):
        return "rate_limited"
    if "lookup " in text or any(marker in text for marker in _DNS_MARKERS):
        return "dns_error"
    if any(marker in text for marker in _MANIFEST_MARKERS) or _NOT_FOUND_RX.search(text):
        return "manifest_unknown"
    if any(marker in text for marker in _AUTH_MARKERS):
        return "auth_error"
    if _UNREACHABLE_RX.search(text):
        return "unreachable"
    if exit_code in _TIMEOUT_EXITS or any(marker in text for marker in _TIMEOUT_MARKERS):
        return "timeout"
    return "other"


def _parse_mirrors(raw: str) -> list[str] | None:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def parse_pull_probe(stdout: str) -> PullReading:
    fields: dict[str, str] = {}
    other_lines: list[str] = []
    mirrors: list[str] | None = None
    driver: str | None = None
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(REGISTRY_PULL_MARKER) and "mirrors=" in stripped:
            # `lium_pull driver=overlayfs mirrors=[...]`; a daemon the CLI cannot reach still prints the
            # format's literal text, `driver= mirrors=`, which reads as unknown
            head, _, raw_mirrors = stripped.partition("mirrors=")
            mirrors = _parse_mirrors(raw_mirrors)
            for token in head[len(REGISTRY_PULL_MARKER) :].split():
                key, _, value = token.partition("=")
                if key == "driver" and value:
                    driver = value
        elif stripped.startswith(REGISTRY_PULL_MARKER):
            for token in stripped[len(REGISTRY_PULL_MARKER) :].split():
                key, _, value = token.partition("=")
                fields[key] = value
        elif stripped:
            other_lines.append(stripped)
    detail = "\n".join(other_lines)[-_TAIL_CHARS:] or None
    cached_before = {"yes": True, "no": False}.get(fields.get("cached", ""))

    if fields.get("cache") == "still_present":
        return PullReading(
            "not_run",
            mirrors=mirrors,
            cached_before=cached_before,
            detail="the image was still on the node after docker rmi; a pull would not reach the registry",
            driver=driver,
        )
    exit_raw = fields.get("exit")
    if exit_raw is None or not exit_raw.lstrip("-").isdigit():
        return PullReading(
            "not_run", mirrors=mirrors, detail=detail or "no pull output", driver=driver
        )
    exit_code = int(exit_raw)
    seconds_raw = fields.get("seconds", "")
    return PullReading(
        classify_pull_error(exit_code, detail or ""),
        mirrors=mirrors,
        exit_code=exit_code,
        seconds=int(seconds_raw) if seconds_raw.isdigit() else None,
        cached_before=cached_before,
        detail=None if exit_code == 0 else detail,
        driver=driver,
    )


def registry_pull_command() -> str:
    return f"/bin/sh -c {shlex.quote(REGISTRY_PULL_SCRIPT)}"


# the state of a node first seen, before its first pull: the pull waits for the node's phase
_SCHEDULED = "scheduled"


def pull_phase_seconds(uuid: str) -> float:
    """Where in each REGISTRY_PULL_PROBE_INTERVAL_HOURS the node's scheduled pull falls, stable per executor.

    Without it every idle node pulls in the first cycle after deploy and every interval after, so the fleet's
    pulls bunch into one hour in six and an incident in the other five has no fleet pulls to show it.
    """
    interval = settings.REGISTRY_PULL_PROBE_INTERVAL_HOURS * 3600
    fraction = int(hashlib.sha256(uuid.encode()).hexdigest()[:12], 16) / 16**12
    return fraction * interval


def next_scheduled_pull(uuid: str, not_before: float) -> float:
    """The first of the node's phase slots at or after `not_before`."""
    interval = settings.REGISTRY_PULL_PROBE_INTERVAL_HOURS * 3600
    phase = pull_phase_seconds(uuid)
    return phase + math.ceil((not_before - phase) / interval) * interval


@dataclass
class _ProbeState:
    """This validator's last pull reading for the node, kept in Redis between cycles."""

    at: float
    outcome: str
    failures_in_a_row: int = 0
    reading: dict[str, Any] = field(default_factory=dict)
    no_verdict: str | None = None
    guard: dict[str, Any] | None = None
    # when the open streak's first counted failure was pulled; None while no streak is open
    streak_started_at: float | None = None

    @property
    def standing_failure(self) -> bool:
        return self.failures_in_a_row >= REGISTRY_PULL_FAILURES_BEFORE_VERDICT

    def next_due_at(self, uuid: str) -> float:
        # an open streak is re-pulled on the retry even after a no-verdict pull, so a failure never stands
        # on a reading hours old
        if self.failures_in_a_row > 0:
            return self.at + settings.REGISTRY_PULL_PROBE_RETRY_MINUTES * 60
        if self.outcome == _SCHEDULED:
            return next_scheduled_pull(uuid, self.at)
        # the node's next phase slot at least half an interval on, so the gap is one interval once the node
        # is on its phase, and 3-9 h when it joins it (after a first sight or a streak that ended)
        return next_scheduled_pull(
            uuid, self.at + settings.REGISTRY_PULL_PROBE_INTERVAL_HOURS * 3600 / 2
        )

    def as_json(self) -> str:
        return json.dumps(
            {
                "at": self.at,
                "outcome": self.outcome,
                "failures_in_a_row": self.failures_in_a_row,
                "reading": self.reading,
                "no_verdict": self.no_verdict,
                "guard": self.guard,
                "streak_started_at": self.streak_started_at,
            }
        )

    @classmethod
    def from_raw(cls, raw: Any) -> _ProbeState | None:
        try:
            data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            failures = int(data.get("failures_in_a_row", 0))
            started = data.get("streak_started_at")
            return cls(
                at=float(data["at"]),
                outcome=str(data["outcome"]),
                failures_in_a_row=failures,
                reading=dict(data.get("reading") or {}),
                no_verdict=data.get("no_verdict"),
                guard=data.get("guard"),
                # a streak recorded before the field: its last pull is the latest its first failure can be
                streak_started_at=float(started)
                if started is not None
                else (float(data["at"]) if failures > 0 else None),
            )
        except (TypeError, ValueError, KeyError, AttributeError):
            return None


@dataclass(frozen=True)
class _FleetEntry:
    uuid: str
    at: float
    failed: bool
    miner: str


def fleet_wait_min_nodes() -> int:
    """The fewest idle nodes whose phases bring REGISTRY_PULL_FLEET_MIN_PULLS scheduled pulls an hour (30 at the
    defaults). On a smaller fleet a confirming failure does not wait for the fleet: the wait would run for hours."""
    interval = settings.REGISTRY_PULL_PROBE_INTERVAL_HOURS * 3600
    return math.ceil(settings.REGISTRY_PULL_FLEET_MIN_PULLS * interval / 3600)


def fleet_reading(
    seen: list[_FleetEntry], *, uuid: str, miner: str, required: int, since: float | None
) -> dict[str, Any]:
    """The latest `required` scheduled pulls of other providers' idle nodes, as one node's confirming failure
    reads them.

    Other providers: outside the node's own miner, and outside the other miner with the most failed pulls, so
    neither the node's own provider nor one other broken provider (ticket-0361: 27 nodes behind one broken
    mirror) can make its failure look like the fleet's. With `since` (the streak's start), also how many of
    those pulls landed after it and how many must.
    """
    others = [entry for entry in seen if entry.uuid != uuid and entry.miner != miner]
    failed_by_miner = Counter(entry.miner for entry in others if entry.failed)
    excluded = failed_by_miner.most_common(1)[0][0] if failed_by_miner else None
    pool = [entry for entry in others if entry.miner != excluded]
    latest = sorted(pool, key=lambda entry: entry.at, reverse=True)[:required]
    failed = sum(entry.failed for entry in latest)
    reading: dict[str, Any] = {
        "idle_nodes_seen": len(seen),
        "required_pulls": required,
        "pulls": len(latest),
        "failed": failed,
        "share": round(failed / len(latest), 3) if latest else 0.0,
        "excluded_miner": excluded,
    }
    if since is not None:
        reading["pulls_since_streak_began"] = sum(entry.at > since for entry in pool)
        reading["needed"] = min(required, len(pool))
    return reading


class _RedisUnreadable(Exception):
    pass


def _decode(raw: Any) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)


async def probe_docker_hub() -> tuple[bool, str]:
    """The validator's own GET of Docker Hub's registry root: (reachable, what it saw)."""
    try:
        timeout = aiohttp.ClientTimeout(total=_HUB_CONTROL_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(DOCKER_HUB_CONTROL_URL, allow_redirects=False) as response:
                return response.status in _HUB_REACHABLE_STATUSES, f"HTTP {response.status}"
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"[:200]


class RegistryPullCheck:
    """Fail an idle node that cannot pull a Docker Hub image through its own registry path (REGISTRY_PULL_FAILED).

    14e704ba (ticket-0361) failed 16 rents in 24 h, every one of them a template image the node did not
    have cached: its dockerd pulls through the registry mirror docker.m.daocloud.io, whose DNS lookup times
    out, while cached templates started fine. The speed tests and the other checks never pull, so they
    passed it. This check runs a real `docker pull` of a tiny digest-pinned image (REGISTRY_PULL_IMAGE)
    after removing it, under a 30 s bound, and records the outcome and the daemon's registry mirrors.

    Bounded: only on idle nodes, at most once per REGISTRY_PULL_PROBE_INTERVAL_HOURS while no failure is
    open, and once per REGISTRY_PULL_PROBE_RETRY_MINUTES while one is (so the second reading, or a fixed
    node's recovery, comes soon). A Docker Hub 429 (anonymous pulls are counted
    per IP, and a provider's nodes can share one) is no verdict, never a failure. Between pulls the last
    reading stands. A node fails once REGISTRY_PULL_FAILURES_BEFORE_VERDICT measured pulls in a row
    failed (timeout, DNS error, unreachable, manifest unknown), and only under
    REGISTRY_PULL_ENFORCEMENT_ENABLED; without it the finding is logged as REGISTRY_PULL_FAILED_OBSERVED
    and the node passes.

    The guards, in three rules:
    1. A failed pull counts only if the validator's own GET of DOCKER_HUB_CONTROL_URL answers 200 or 401
       (cached for HUB_CONTROL_TTL_SECONDS); else it is REGISTRY_PULL_NO_VERDICT_HUB_DOWN.
    2. The second failure in a row, the one that fails the node, counts only if at most
       REGISTRY_PULL_FLEET_SHARE of the latest `required` scheduled pulls of other providers' idle nodes
       failed (fleet_reading); else it is REGISTRY_PULL_NO_VERDICT_FLEET, the streak stays open and is
       retried, and REGISTRY_PULL_FLEET_FAILING is logged at most hourly. `required` is
       REGISTRY_PULL_FLEET_MIN_PULLS, or FLEET_MIN_PULLS_SHARE_OF_SEEN of the idle nodes seen (last
       scheduled pull within one interval plus an hour) if more. On a validator seeing
       fleet_wait_min_nodes() idle nodes or more (30 at the defaults), those pulls must also all have landed
       after the streak's first failure (or all the other providers' nodes pulled since, if fewer), so it is
       judged on pulls taken since the node started failing; until then it is no verdict (awaiting_fleet).
       On a smaller fleet the 30-minute retry confirms.
    3. A scheduled pull is a node's regular one at its phase (pull_phase_seconds), never the retry of an
       open streak, so a few broken nodes re-pulling every 30 minutes weigh no more than healthy ones.
    The control sees an outage from where the validator stands, at once and on a fleet of any size; the
    fleet reading catches what the validator cannot see (an outage of Docker Hub's CDN in one region, a
    mirror many providers share, an error text a new Docker release words differently). Nothing but each
    node's streak is kept between pulls: a confirming failure is judged on the fleet's latest pulls alone.
    """

    check_id = "executor.validate.registry_pull"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.REGISTRY_PULL_CHECK_ENABLED:
            return self._skipped(ctx, "REGISTRY_PULL_CHECK_ENABLED is off")
        if _is_rented(ctx):
            return self._skipped(ctx, "rented")
        try:
            last = await self._load(ctx)
        except _RedisUnreadable:
            # without the last reading the interval cannot be honoured; pulling every cycle would spend
            # the node's Docker Hub quota, so this cycle has no verdict
            return self._skipped(ctx, "last pull reading unreadable in Redis")

        now = time.time()
        if last is None:
            # first sight: the first pull waits for the node's phase, so a deploy does not pull the fleet at once
            last = _ProbeState(at=now, outcome=_SCHEDULED)
            await self._save(ctx, last)
        if now < last.next_due_at(ctx.executor.uuid):
            return self._verdict(ctx, last, probed=False)

        reading = await self._pull(ctx)
        failures = last.failures_in_a_row
        started = last.streak_started_at
        no_verdict: NoVerdict | None = None
        guard: dict[str, Any] | None = None
        if reading.outcome == "ok" or reading.failed:
            async with _FLEET_LOCK:
                if failures == 0:
                    await self._record_fleet_pull(ctx, now, failed=reading.failed)
                    # reading the window prunes it, which a fleet that never fails would otherwise never do
                    await self._fleet_window(ctx, now)
                if reading.failed:
                    no_verdict, guard, failures, started = await self._guard(
                        ctx, now, failures, started
                    )
        if reading.outcome == "ok":
            failures, started = 0, None
        state = _ProbeState(
            at=now,
            outcome=reading.outcome,
            failures_in_a_row=failures,
            reading=reading.as_record(),
            no_verdict=no_verdict,
            guard=guard,
            streak_started_at=started,
        )
        await self._save(ctx, state)
        return self._verdict(ctx, state, probed=True)

    async def _guard(
        self, ctx: Context, now: float, failures: int, started: float | None
    ) -> tuple[NoVerdict | None, dict[str, Any], int, float | None]:
        """Whether this failed pull counts (no_verdict None) and what the guards saw, with the streak it leaves:
        (no_verdict, guard, failures_in_a_row, streak_started_at). Runs under _FLEET_LOCK."""
        control = await self._hub_control(ctx, now)
        guard: dict[str, Any] = {"docker_hub_control": control}
        if not control["reachable"]:
            return "docker_hub_down", guard, failures, started
        if not 0 < failures < REGISTRY_PULL_FAILURES_BEFORE_VERDICT:
            return None, guard, failures + 1, started if failures > 0 else now
        fleet = await self._fleet_window(ctx, now)
        if fleet is None:
            return "fleet_state_unreadable", guard, failures, started
        seen, required = fleet
        waits = len(seen) >= fleet_wait_min_nodes()
        fleet_guard = fleet_reading(
            seen,
            uuid=ctx.executor.uuid,
            miner=ctx.miner_hotkey or f"executor:{ctx.executor.uuid}",
            required=required,
            since=started if waits else None,
        )
        guard["fleet"] = fleet_guard
        if fleet_guard["failed"] > settings.REGISTRY_PULL_FLEET_SHARE * fleet_guard["pulls"]:
            await self._log_fleet_failing(ctx, now, fleet_guard)
            return "fleet_failing", guard, failures, started
        if waits and fleet_guard["pulls_since_streak_began"] < fleet_guard["needed"]:
            return "awaiting_fleet", guard, failures, started
        return None, guard, failures + 1, started if failures > 0 else now

    async def _hub_control(self, ctx: Context, now: float) -> dict[str, Any]:
        redis = ctx.services.redis
        try:
            raw = await redis.get(_HUB_CONTROL_KEY)
            cached = json.loads(_decode(raw)) if raw is not None else None
            if cached is not None and now - float(cached["at"]) < HUB_CONTROL_TTL_SECONDS:
                return cached
        except Exception:
            logger.warning(
                _m("Registry pull: the cached Docker Hub control is unreadable; fetching it again"),
                exc_info=True,
            )
        reachable, seen = await probe_docker_hub()
        control = {"at": now, "reachable": reachable, "seen": seen}
        try:
            await redis.set(_HUB_CONTROL_KEY, json.dumps(control), ex=HUB_CONTROL_TTL_SECONDS)
        except Exception:
            logger.warning(
                _m("Registry pull: could not cache the Docker Hub control"), exc_info=True
            )
        if not reachable:
            logger.warning(
                _m(
                    "REGISTRY_PULL_NO_VERDICT_HUB_DOWN: the validator cannot reach Docker Hub either; "
                    f"failed pulls are no verdict for the next {HUB_CONTROL_TTL_SECONDS // 60} minutes",
                    extra=get_extra_info({"url": DOCKER_HUB_CONTROL_URL, "seen": seen}),
                )
            )
        return control

    async def _record_fleet_pull(self, ctx: Context, now: float, *, failed: bool) -> None:
        """Runs under _FLEET_LOCK, so a prune in _fleet_window never deletes the entry written here."""
        entry = {"at": now, "failed": failed, "miner": ctx.miner_hotkey}
        try:
            await ctx.services.redis.hset(_FLEET_KEY, ctx.executor.uuid, json.dumps(entry))
        except Exception:
            logger.warning(
                _m(
                    "Registry pull: could not record the pull in the fleet window",
                    extra=get_extra_info(ctx.default_extra),
                ),
                exc_info=True,
            )

    async def _fleet_window(self, ctx: Context, now: float) -> tuple[list[_FleetEntry], int] | None:
        """The idle nodes seen (their last scheduled pull within one interval plus an hour) and `required`.
        Prunes older entries; runs under _FLEET_LOCK."""
        redis = ctx.services.redis
        try:
            raw = await redis.hgetall(_FLEET_KEY) or {}
        except Exception:
            logger.warning(_m("Registry pull: the fleet window is unreadable"), exc_info=True)
            return None
        horizon = settings.REGISTRY_PULL_PROBE_INTERVAL_HOURS * 3600 + FLEET_HORIZON_SLACK_SECONDS
        seen: list[_FleetEntry] = []
        expired: list[str] = []
        for key, value in raw.items():
            uuid = _decode(key)
            try:
                data = json.loads(_decode(value))
                entry = _FleetEntry(
                    uuid=uuid,
                    at=float(data["at"]),
                    failed=bool(data.get("failed")),
                    # an entry without a hotkey stands for a miner of its own
                    miner=str(data.get("miner") or f"executor:{uuid}"),
                )
            except (TypeError, ValueError, KeyError, AttributeError):
                expired.append(uuid)
                continue
            if now - entry.at <= horizon:
                seen.append(entry)
            else:
                expired.append(uuid)
        if expired:
            try:
                await redis.hdel(_FLEET_KEY, *expired)
            except Exception:
                logger.warning(_m("Registry pull: could not prune the fleet window"), exc_info=True)
        required = max(
            settings.REGISTRY_PULL_FLEET_MIN_PULLS,
            math.ceil(FLEET_MIN_PULLS_SHARE_OF_SEEN * len(seen)),
        )
        return seen, required

    async def _log_fleet_failing(self, ctx: Context, now: float, fleet: dict[str, Any]) -> None:
        """REGISTRY_PULL_FLEET_FAILING, at most once per FLEET_LOG_INTERVAL_SECONDS. Runs under _FLEET_LOCK."""
        redis = ctx.services.redis
        try:
            raw = await redis.get(_FLEET_LOGGED_KEY)
            if (
                raw is not None
                and now - float(json.loads(_decode(raw))) < FLEET_LOG_INTERVAL_SECONDS
            ):
                return
            await redis.set(_FLEET_LOGGED_KEY, json.dumps(now), ex=FLEET_LOG_INTERVAL_SECONDS)
        except Exception:
            logger.warning(_m("Registry pull: the fleet log marker is unreadable"), exc_info=True)
        logger.warning(
            _m(
                f"REGISTRY_PULL_FLEET_FAILING: {fleet['failed']} of the latest {fleet['pulls']} scheduled pulls "
                "of other providers' idle nodes failed; a node's second failed pull in a row is no verdict while "
                f"more than {settings.REGISTRY_PULL_FLEET_SHARE:.0%} of them fail",
                extra=get_extra_info({**fleet, "threshold": settings.REGISTRY_PULL_FLEET_SHARE}),
            )
        )

    def _verdict(self, ctx: Context, state: _ProbeState, *, probed: bool) -> CheckResult:
        what: dict[str, Any] = {
            "pull": state.reading,
            "probed_this_cycle": probed,
            "failures_in_a_row": state.failures_in_a_row,
            "enforced": settings.REGISTRY_PULL_ENFORCEMENT_ENABLED,
        }
        if state.streak_started_at is not None:
            what["streak_started_at"] = state.streak_started_at
        if not probed:
            what["read_at"] = state.at
            what["next_pull_at"] = state.next_due_at(ctx.executor.uuid)
        if state.no_verdict is not None:
            what["no_verdict"] = state.no_verdict
        if state.guard is not None:
            what["guard"] = state.guard
        if state.standing_failure:
            template = (
                Msg.REGISTRY_PULL_FAILED
                if settings.REGISTRY_PULL_ENFORCEMENT_ENABLED
                else Msg.REGISTRY_PULL_FAILED_OBSERVED
            )
            event = render_message(template, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=not settings.REGISTRY_PULL_ENFORCEMENT_ENABLED, event=event)
        if not probed:
            return self._skipped(ctx, "not due", last=what)
        if state.no_verdict == "docker_hub_down":
            template = Msg.REGISTRY_PULL_NO_VERDICT_HUB_DOWN
        elif state.no_verdict == "fleet_failing":
            template = Msg.REGISTRY_PULL_NO_VERDICT_FLEET
        elif state.no_verdict is not None:
            template = Msg.REGISTRY_PULL_UNMEASURED
        elif state.outcome in FAILING_OUTCOMES:
            template = Msg.REGISTRY_PULL_FAILED_ONCE
        elif state.outcome == "ok":
            template = Msg.REGISTRY_PULL_OK
        else:
            template = Msg.REGISTRY_PULL_UNMEASURED
        event = render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event)

    async def _pull(self, ctx: Context) -> PullReading:
        result = await ctx.runner.run(
            registry_pull_command(),
            timeout=REGISTRY_PULL_COMMAND_TIMEOUT_SECONDS,
            retryable=False,
        )
        if result.error_type is not None:
            return PullReading("not_run", detail=f"{result.error_type}: {result.error_message}")
        reading = parse_pull_probe(result.stdout)
        if reading.outcome in UNMEASURED_OUTCOMES:
            logger.info(
                _m(
                    "Registry pull probe reached no verdict",
                    extra=get_extra_info({**ctx.default_extra, **reading.as_record()}),
                )
            )
        return reading

    async def _load(self, ctx: Context) -> _ProbeState | None:
        try:
            raw = await ctx.services.redis.get(f"{_REDIS_PREFIX}:{ctx.executor.uuid}")
        except Exception as exc:
            logger.warning(
                _m(
                    "Registry pull probe could not read its last reading from Redis; no pull this cycle",
                    extra=get_extra_info(ctx.default_extra),
                ),
                exc_info=True,
            )
            raise _RedisUnreadable from exc
        return _ProbeState.from_raw(raw) if raw is not None else None

    async def _save(self, ctx: Context, state: _ProbeState) -> None:
        try:
            await ctx.services.redis.set(
                f"{_REDIS_PREFIX}:{ctx.executor.uuid}", state.as_json(), ex=_REDIS_TTL_SECONDS
            )
        except Exception:
            logger.warning(
                _m(
                    "Registry pull probe could not record its reading in Redis",
                    extra=get_extra_info(ctx.default_extra),
                ),
                exc_info=True,
            )

    def _skipped(
        self, ctx: Context, reason: str, *, last: dict[str, Any] | None = None
    ) -> CheckResult:
        what: dict[str, Any] = {"reason": reason}
        if last is not None:
            what["last"] = last
        event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event)
