from __future__ import annotations

import hashlib
import json
import logging
import shlex
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, NamedTuple

import aiohttp

from core.config import settings
from core.docker_utils import ALPINE_HELPER_IMAGE
from core.utils import _m, get_extra_info
from services.rental_docker_sdk import RENTAL_NETWORK_NAME

from ..messages import ContainerDownloadSpeedMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# Two public objects a renter's first minutes fetch: a wheel from PyPI's file CDN and a model file from Hugging Face
# (a pinned revision, so the bytes and the redirect target stay the same). Each read is capped at CAP_BYTES by a
# `head -c` (no Range: busybox wget fails on a 206 answer), under FETCH_TIMEOUT_SECONDS.
ENDPOINTS: tuple[tuple[str, str], ...] = (
    (
        "pypi",
        "https://files.pythonhosted.org/packages/9b/b4/e3c7e6fab0f77fff6194afa173d1f2342073d91b1d3b4b30b17c3fb4407a/"
        "numpy-2.1.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    ),
    (
        "hf",
        "https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors",
    ),
)
CAP_BYTES = 16 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 30
# container start plus both fetches, each with its timeout's grace
COMMAND_TIMEOUT_SECONDS = 120
# a slow reading is confirmed by a second one this soon, not one interval later
RETRY_SECONDS = 30 * 60
READINGS_BEFORE_VERDICT = 2
MARKER = "lium_dl"
_REDIS_PREFIX = "container_download_speed"
_REDIS_TTL_SECONDS = 7 * 24 * 3600
_CONTROL_TIMEOUT_SECONDS = 10
_TAIL_CHARS = 200
# busybox timeout's TERM/KILL exits: the fetch ran out of time, whatever it read by then is the reading
_TIMEOUT_EXITS = frozenset({124, 143, 137})
# the container could not reach the endpoint at all: a renter's pip/hf download fails the same way
NODE_SIDE_ERRORS = frozenset({"dns_failed", "unreachable", "timeout"})

# busybox sh in alpine (the port check's helper image, already on the node). /proc/uptime is the clock: busybox
# date has no sub-second format everywhere. Exit 0 always: the marker lines are the answer.
_FETCH_SCRIPT = (
    "up() { cut -d' ' -f1 /proc/uptime; }; "
    "probe() { s=$(up); "
    f"n=$( {{ timeout {FETCH_TIMEOUT_SECONDS} wget -q -O - \"$2\" "
    f"2>/tmp/e; echo $? >/tmp/rc; }} | head -c {CAP_BYTES} | wc -c); e=$(up); "
    f'echo "{MARKER} ep=$1 bytes=$n start=$s end=$e rc=$(cat /tmp/rc 2>/dev/null)"; '
    f"echo \"{MARKER}_err ep=$1 $(head -c {_TAIL_CHARS} /tmp/e | tr '\\n' ' ')\"; }}; "
    + "; ".join(f"probe {name} {shlex.quote(url)}" for name, url in ENDPOINTS)
    + "; exit 0"
)


def download_speed_command(sysbox: bool, script: str = _FETCH_SCRIPT) -> str:
    """A throwaway container on the rentals bridge, under sysbox-runc where renters get it.

    Pods run on RENTAL_NETWORK_NAME (own DNS and MTU), so that is the path to measure. The node creates it at its
    first rental; until then the probe falls back to the default bridge and says which one it used.
    """
    runtime = "--runtime=sysbox-runc " if sysbox else ""
    return (
        f"net=bridge; /usr/bin/docker network inspect {RENTAL_NETWORK_NAME} >/dev/null 2>&1 && "
        f"net={RENTAL_NETWORK_NAME}; echo \"{MARKER}_net name=$net\"; "
        f"/usr/bin/docker run --rm --network \"$net\" {runtime}--entrypoint /bin/sh {ALPINE_HELPER_IMAGE} "
        f"-c {shlex.quote(script)}"
    )


@dataclass(frozen=True)
class EndpointReading:
    endpoint: str
    bytes: int = 0
    seconds: float | None = None
    mbps: float | None = None
    # None = measured; else why not: dns_failed, unreachable, timeout, http_error, no_output, other
    error: str | None = None
    detail: str | None = None
    network: str | None = None


class Judgement(NamedTuple):
    is_failing: bool
    has_no_verdict: bool


def _error_kind(stderr: str) -> str:
    text = stderr.lower()
    if "bad address" in text or "name or service not known" in text:
        return "dns_failed"
    if "server returned error" in text:
        return "http_error"
    if "refused" in text or "unreachable" in text or "no route" in text:
        return "unreachable"
    return "other"


def parse_download_speed(stdout: str) -> list[EndpointReading]:
    """One reading per endpoint; an endpoint with no marker line is `no_output`."""
    fields: dict[str, dict[str, str]] = {}
    errors: dict[str, str] = {}
    network = None
    for line in (stdout or "").splitlines():
        if line.startswith(f"{MARKER}_net name="):
            network = line[len(f"{MARKER}_net name=") :].strip() or None
        elif line.startswith(f"{MARKER}_err ep="):
            name, _, text = line[len(f"{MARKER}_err ep=") :].partition(" ")
            errors[name] = text.strip()
        elif line.startswith(f"{MARKER} "):
            tokens = dict(t.partition("=")[::2] for t in line[len(MARKER) :].split())
            fields[tokens.get("ep", "")] = tokens
    readings = []
    for name, _ in ENDPOINTS:
        reading = fields.get(name)
        detail = errors.get(name, "")[:_TAIL_CHARS] or None
        if reading is None:
            readings.append(EndpointReading(name, error="no_output"))
            continue
        try:
            got = int(reading.get("bytes", "0"))
            seconds = float(reading["end"]) - float(reading["start"])
            rc = int(reading.get("rc") or -1)
        except (KeyError, ValueError):
            readings.append(EndpointReading(name, error="no_output", detail=detail))
            continue
        timed_out = rc in _TIMEOUT_EXITS
        if got >= CAP_BYTES or (got > 0 and (rc == 0 or timed_out)):
            # a timed-out fetch that read something is the slow reading itself, not an error; a full read is a
            # reading whatever wget's exit (head stops a large file at the cap and wget dies of SIGPIPE)
            mbps = got * 8 / max(seconds, 0.01) / 1e6
            readings.append(EndpointReading(name, got, seconds, mbps))
        elif timed_out:
            readings.append(EndpointReading(name, got, seconds, error="timeout", detail=detail))
        else:
            readings.append(
                EndpointReading(name, got, seconds, error=_error_kind(detail or ""), detail=detail)
            )
    return [replace(r, network=network) for r in readings]


def phase_offset_seconds(uuid: str) -> float:
    """Where in each interval the node's measurement falls, stable per executor, so a deploy doesn't measure the fleet at once."""
    fraction = int(hashlib.sha256(uuid.encode()).hexdigest()[:12], 16) / 16**12
    return fraction * settings.CONTAINER_DOWNLOAD_SPEED_INTERVAL_HOURS * 3600


async def validator_reaches(url: str) -> bool:
    """The validator's own one-byte GET of the endpoint: if it fails too, the node's miss is not the node's."""
    try:
        timeout = aiohttp.ClientTimeout(total=_CONTROL_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers={"Range": "bytes=0-0"}) as response:
                return response.status < 400
    except Exception:  # noqa: BLE001 — any failure is "cannot reach"
        return False


@dataclass
class _State:
    at: float
    failures_in_a_row: int = 0
    readings: list[EndpointReading] = field(default_factory=list)
    measured: bool = False
    # failing readings in a row taken on the rentals bridge. Only these are enforced: a default-bridge reading (a
    # node with no rental yet) restarts the count, and a state saved without the field starts from 0.
    failing_readings_on_rentals_network: int = 0

    def next_due_at(self) -> float:
        if self.failures_in_a_row > 0:
            return self.at + RETRY_SECONDS
        return self.at + settings.CONTAINER_DOWNLOAD_SPEED_INTERVAL_HOURS * 3600

    @classmethod
    def from_raw(cls, raw: Any) -> _State | None:
        try:
            data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            return cls(
                at=float(data["at"]),
                failures_in_a_row=int(data.get("failures_in_a_row", 0)),
                readings=[EndpointReading(**r) for r in data.get("readings") or []],
                measured=bool(data.get("measured")),
                failing_readings_on_rentals_network=int(data.get("failing_readings_on_rentals_network", 0)),
            )
        except (TypeError, ValueError, KeyError, AttributeError):
            return None


class ContainerDownloadSpeedCheck:
    """Measure what a renter's container gets from PyPI and Hugging Face, not what the host gets (CONTAINER_DOWNLOAD_SLOW).

    A node read 297-588 Mbps on the validator's host-side speed test while a renter container on it fetched
    from PyPI at under 1 Mbps: the host probe runs in the executor container on the host network and never
    crosses the docker bridge, NAT and runtime a renter's container does. This check starts a throwaway
    alpine container (sysbox-runc where the node has it, on the rentals bridge pods use) and reads up to
    CAP_BYTES of a pinned PyPI wheel and a pinned Hugging Face file, each under FETCH_TIMEOUT_SECONDS.

    Every measurement is logged as CONTAINER_DOWNLOAD_SPEED. A reading fails when a measured endpoint is below
    CONTAINER_DOWNLOAD_SPEED_MIN_MBPS, or the container cannot reach an endpoint (DNS, refused, nothing read in
    time) that the validator itself reaches. An endpoint the validator cannot reach either, an HTTP error, or a
    container that did not run is no verdict. READINGS_BEFORE_VERDICT failing readings in a row (the second
    RETRY_SECONDS after the first) are the finding: a fail (CONTAINER_DOWNLOAD_SLOW, score 0) that stands
    until a reading passes, or CONTAINER_DOWNLOAD_SLOW_OBSERVED (logged only) when a failing reading in the
    streak was not taken on the rentals network. Idle nodes only (no pod, no filler: both would share the link), once per
    CONTAINER_DOWNLOAD_SPEED_INTERVAL_HOURS at a per-node phase.
    """

    check_id = "executor.validate.container_download_speed"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.CONTAINER_DOWNLOAD_SPEED_CHECK_ENABLED:
            return self._skipped(ctx, "CONTAINER_DOWNLOAD_SPEED_CHECK_ENABLED is off")
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        if rented_executor is not None and rented_executor.pods:
            return self._skipped(ctx, "rented")
        if rented_data is not None and rented_data.get_filler_containers(ctx.executor.uuid):
            return self._skipped(ctx, "filler running")

        key = f"{_REDIS_PREFIX}:{ctx.executor.uuid}"
        redis = ctx.services.redis
        try:
            raw = await redis.get(key)
        except Exception:
            # without the last reading the interval cannot be kept; measuring every cycle is the cost to avoid
            logger.warning(_m("Container download speed: Redis unreadable", extra=get_extra_info(ctx.default_extra)))
            return self._skipped(ctx, "last reading unreadable in Redis")
        state = _State.from_raw(raw) if raw is not None else None
        now = time.time()
        if state is None:
            # first sight: due at the node's phase within the next interval
            state = _State(at=now - phase_offset_seconds(ctx.executor.uuid))
            await self._save(ctx, key, state)
        if now < state.next_due_at():
            return self._verdict(ctx, state, probed=False)

        readings = await self._measure(ctx)
        judgement = await self._judge(readings)
        failing, no_verdict = judgement.is_failing, judgement.has_no_verdict
        on_rentals = all(r.network == RENTAL_NETWORK_NAME for r in readings)
        if failing:
            failures = state.failures_in_a_row + 1
            failing_on_rentals = state.failing_readings_on_rentals_network + 1 if on_rentals else 0
        elif no_verdict:
            failures, failing_on_rentals = state.failures_in_a_row, state.failing_readings_on_rentals_network
        else:
            failures = failing_on_rentals = 0
        state = _State(
            at=now,
            failures_in_a_row=failures,
            readings=readings,
            measured=not no_verdict,
            failing_readings_on_rentals_network=failing_on_rentals,
        )
        logger.info(
            _m(
                "CONTAINER_DOWNLOAD_SPEED",
                extra=get_extra_info(
                    {
                        **ctx.default_extra,
                        "endpoints": [asdict(r) for r in readings],
                        "min_mbps": settings.CONTAINER_DOWNLOAD_SPEED_MIN_MBPS,
                        "failing": failing,
                        "no_verdict": no_verdict,
                        "failures_in_a_row": failures,
                    }
                ),
            )
        )
        await self._save(ctx, key, state)
        return self._verdict(ctx, state, probed=True)

    async def _measure(self, ctx: Context) -> list[EndpointReading]:
        result = await ctx.runner.run(
            download_speed_command(bool(ctx.state.sysbox_runtime)),
            timeout=COMMAND_TIMEOUT_SECONDS,
            retryable=False,
        )
        readings = parse_download_speed(result.stdout or "")
        if result.error_type is not None or all(r.error == "no_output" for r in readings):
            detail = f"{result.error_type or 'exit ' + str(result.exit_code)}: {(result.error_message or result.stderr or '')}"
            return [EndpointReading(name, error="no_output", detail=detail[-_TAIL_CHARS:]) for name, _ in ENDPOINTS]
        return readings

    async def _judge(self, readings: list[EndpointReading]) -> Judgement:
        """Slow on any endpoint fails: a renter needs both."""
        threshold = settings.CONTAINER_DOWNLOAD_SPEED_MIN_MBPS
        if any(r.mbps is not None and r.mbps < threshold for r in readings):
            return Judgement(is_failing=True, has_no_verdict=False)
        urls = dict(ENDPOINTS)
        for r in readings:
            if r.error in NODE_SIDE_ERRORS and await validator_reaches(urls[r.endpoint]):
                return Judgement(is_failing=True, has_no_verdict=False)
        # an endpoint left unmeasured (ours, or the endpoint's) and no failure: no verdict
        return Judgement(is_failing=False, has_no_verdict=any(r.mbps is None for r in readings))

    def _verdict(self, ctx: Context, state: _State, *, probed: bool) -> CheckResult:
        enforced = state.failing_readings_on_rentals_network >= READINGS_BEFORE_VERDICT
        what: dict[str, Any] = {
            "endpoints": [asdict(r) for r in state.readings],
            "min_mbps": settings.CONTAINER_DOWNLOAD_SPEED_MIN_MBPS,
            "probed_this_cycle": probed,
            "failures_in_a_row": state.failures_in_a_row,
            "enforced": enforced,
            "failing_readings_on_rentals_network": state.failing_readings_on_rentals_network,
        }
        if state.failures_in_a_row >= READINGS_BEFORE_VERDICT:
            template = Msg.SLOW if enforced else Msg.SLOW_OBSERVED
            event = render_message(template, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=not enforced, event=event)
        if not probed:
            what["read_at"] = state.at
            what["next_at"] = state.next_due_at()
            return self._skipped(ctx, "not due", last=what)
        if state.failures_in_a_row:
            template = Msg.SLOW_ONCE
        elif state.measured:
            template = Msg.OK
        else:
            template = Msg.UNMEASURED
        return CheckResult(passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what))

    async def _save(self, ctx: Context, key: str, state: _State) -> None:
        payload = {
            "at": state.at,
            "failures_in_a_row": state.failures_in_a_row,
            "readings": [asdict(r) for r in state.readings],
            "measured": state.measured,
            "failing_readings_on_rentals_network": state.failing_readings_on_rentals_network,
        }
        try:
            await ctx.services.redis.set(key, json.dumps(payload), ex=_REDIS_TTL_SECONDS)
        except Exception:
            logger.warning(
                _m("Container download speed: could not record the reading", extra=get_extra_info(ctx.default_extra))
            )

    def _skipped(self, ctx: Context, reason: str, *, last: dict[str, Any] | None = None) -> CheckResult:
        what: dict[str, Any] = {"reason": reason}
        if last is not None:
            what["last"] = last
        return CheckResult(passed=True, event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what=what))
