"""Renter-side reachability of a RUNNING rented pod (DAH-2870, B-71).

A host reboot lets dockerd bring the rental container back on its own. That restart path never
mounts the encrypted volume and sometimes never starts sshd, so the pod's SSH port refuses
(ticket-0326) or accepts and then asks for a password because ``/root/.ssh/authorized_keys`` is
unreadable (ticket-0247). The container is running, so ``TenantEnforcementCheck`` scored the node
1.0 and the renter kept paying for a pod nobody could enter.

This module judges what the renter sees, from outside the container, each cycle:

* a TCP connect to the pod's mapped SSH port (``RentedPod.ssh_port``, sent by the backend) that,
  with ``RENTED_POD_SSH_BANNER_FAULT_ENABLED`` on, must answer with a complete ``SSH-2.0-``
  identification line (RFC 4253 §4.2): docker-proxy accepts on the host while sshd inside is down,
  so an accept alone says nothing, and a 1.x or malformed line is not the sshd the renter can log
  in to. The setting is off by default and is turned on after
  lium-platform#429, which teaches the backend the ``ssh_banner_missing`` fault name, is deployed;
  until then the connect alone decides the port fault, and a report carries only ``tcp_refused``,
  ``tcp_timeout`` or ``authorized_keys_unreadable``;
* the ``authorized_keys`` read the check already does (empty = the mount is missing).

State lives in Redis, one key pair per pod. A pod is judged only after this validator has seen it
healthy once (both signals good), so a template that ships no sshd, or a pod that never came up,
is never reported here. ``RENTED_POD_SSH_PROBE_CYCLES`` consecutive unhealthy cycles (default 2,
about 30 min) after that raise ``RENTED_POD_SSH_UNREACHABLE`` and one POST to the backend per
outage: the POST is repeated each cycle until the backend answers 200 (``recorded`` true or false),
and that answer is kept in the streak so the outage is reported once. The score is not changed by
this module.

The POST is deferred to the end of the cycle and gated by the fleet (Rustam's review, 16 Sep): a
validator whose own network fails sees every mapped port refuse at once, and per-pod reporting
would tell every healthy renter their pod is down. Each probe writes its mapped-port result into a
per-cycle Redis hash; a pod at the threshold queues its report into a second one. At the cycle's
end ``flush_rented_pod_ssh_reports`` reads both: when the share of probed pods failing the
mapped-port check is above ``RENTED_POD_SSH_PROBE_FLEET_FAIL_MAX`` (default 0.5, on a fleet of at
least ``SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE`` pods), or the cycle's executor-SSH check already
judged the validator to be the outage (DAH-2748), the queued reports are logged as
``RENTED_POD_SSH_PROBE_SUPPRESSED_FLEET`` and no report is POSTed, so no renter is told. The
streaks keep ``reported`` False, so the outage is queued again next cycle and reported once the
fleet reads clean. The per-executor ``RENTED_POD_SSH_UNREACHABLE`` events of a suppressed cycle
were rendered before the gate ran and say the renter was told; the sync loop passes the gate to
``silence_rented_pod_ssh_reports_on_our_own_outage`` before the specs publish, which rewrites them
to RENTED with the gate's verdict under ``what_we_saw``, as DAH-2748 rewrites availability errors.

Redis is an input to this signal, never to the check's verdict: when Redis fails, the probe logs
``RENTED_POD_SSH_PROBE_REDIS_UNAVAILABLE`` and returns None for that pod and cycle, exactly as when
the probe is disabled. Every transition the probe makes to a pod's state (ok mark, streak, fleet
mark, queued report) is one MULTI/EXEC through ``RedisWrites``, so a connection lost between two
writes leaves the previous state whole rather than half of the new one; the closed-rental forget
and the flush's key drops are plain deletes, because a half-done delete there costs at most one
TTL or one re-queued cycle. Both keys carry a TTL
(``RENTED_POD_SSH_PROBE_STATE_TTL_SECONDS``, renewed on every probe of the pod) and are deleted by
``forget_rented_pod_ssh`` when the rental has closed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import redis.exceptions
from protocol.vc_protocol.compute_requests import RentedPod

from core.config import settings
from core.utils import _m, get_extra_info

from ...redis_service import RedisWrites
from ..availability import SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE
from ..messages import TenantEnforcementMessages
from ..models import JobResult, build_msg
from ..pipeline import Context

logger = logging.getLogger(__name__)

# Redis keys. `ok` carries the boot_id seen when the pod was last healthy, so a later failure can
# say whether the host rebooted in between. `fail` carries the streak.
RENTED_POD_SSH_OK_KEY_PREFIX = "rented_pod_ssh_ok"
RENTED_POD_SSH_FAIL_KEY_PREFIX = "rented_pod_ssh_fail"
# Per-cycle hashes, keyed by job_batch_id. `fleet`: every probed pod's mapped-port result this
# cycle (FLEET_MARK_OK or the fault). `due`: the reports waiting for the cycle-end fleet gate.
# Both are deleted by the flush; the TTL covers a cycle the validator never finished.
RENTED_POD_SSH_FLEET_KEY_PREFIX = "rented_pod_ssh_fleet"
RENTED_POD_SSH_DUE_KEY_PREFIX = "rented_pod_ssh_due"
FLEET_KEY_TTL_SECONDS = 3600
FLEET_MARK_OK = "ok"
# A cycle whose reports the gate held back: the field every suppressed log line carries.
PROBE_SUPPRESSED_FLEET = "probe_suppressed_fleet"

FAULT_TCP_REFUSED = "tcp_refused"
FAULT_TCP_TIMEOUT = "tcp_timeout"
# The port accepted but nothing that speaks SSH 2.0 is behind it: docker-proxy took the connection
# and closed it (sshd not running in the container), something else answered, or the identification
# line is SSH 1.x / malformed / never completed. The four names below are the backend's
# `PodSshUnreachableRequest.faults` vocabulary (lium-platform#429); a name the backend does not
# know is a 422 and the outage is never recorded, so the two lists move together. This one is sent
# only with RENTED_POD_SSH_BANNER_FAULT_ENABLED on (after lium-platform#429 is deployed).
FAULT_SSH_BANNER_MISSING = "ssh_banner_missing"
FAULT_AUTHORIZED_KEYS_UNREADABLE = "authorized_keys_unreadable"
# RFC 4253 §4.2: `SSH-protoversion-softwareversion SP comments CR LF`, at most 255 bytes including
# CR LF. Rustam's review (16 Sep): only protoversion 2.0 counts. `SSH-1.5-` is a 1.x-only server;
# `SSH-1.99-` (RFC 4253 §5.1) marks a server that also speaks 1.x, and the ask is to refuse both.
SSH_ID_PREFIX = b"SSH-2.0-"
SSH_ID_LINE_MAX = 255
# The same section lets the server send other lines before its identification (each ending in CR LF,
# none starting with `SSH-`) and a client MUST be able to skip them. The scan is bounded: OpenSSH's
# client gives up after 1024 such lines; 64 is more than any pre-banner an sshd is configured to
# print, and a peer that is not sshd at all runs out of lines (or of the shared deadline) long
# before it can hold the probe. Each skipped line is bounded to SSH_ID_LINE_MAX bytes as well.
SSH_PRE_BANNER_LINES_MAX = 64
SSH_ID_ANY_VERSION_PREFIX = b"SSH-"

# What a failing Redis raises through RedisService: the client's own errors (connection, timeout,
# response) and the socket errors under them. Anything else is a bug in this module and propagates.
REDIS_ERRORS: tuple[type[BaseException], ...] = (redis.exceptions.RedisError, OSError)


@dataclass(frozen=True)
class RentedPodSshVerdict:
    """What the probe saw for one pod this cycle."""

    pod_id: str
    container_name: str
    ssh_port: int | None
    healthy: bool
    faults: list[str] = field(default_factory=list)
    consecutive_cycles: int = 0
    first_failed_at: str | None = None
    boot_id_changed: bool | None = None
    # True from the cycle the streak reaches the threshold until the pod is healthy again: the
    # cycle's event names the pod. On every such cycle until the backend answers 200 once
    # (FailStreak.reported) the report is queued for the cycle-end fleet gate; report_queued says
    # whether THIS cycle queued it. Whether it was posted is the flush's log line, not the verdict's.
    report: bool = False
    report_queued: bool = False


@dataclass(frozen=True)
class FleetGate:
    """What the cycle-end gate saw and did with the queued reports."""

    job_batch_id: str
    probed: int
    failed: int
    due: list[str]
    validator_outage: bool
    # None when the reports went out (or there were none); else why they were held back.
    suppressed_by: str | None = None
    posted: list[str] = field(default_factory=list)

    @property
    def fail_share(self) -> float:
        return self.failed / self.probed if self.probed else 0.0


def _ok_key(pod_id: str) -> str:
    return f"{RENTED_POD_SSH_OK_KEY_PREFIX}:{pod_id}"


def _fail_key(pod_id: str) -> str:
    return f"{RENTED_POD_SSH_FAIL_KEY_PREFIX}:{pod_id}"


def _fleet_key(job_batch_id: str) -> str:
    return f"{RENTED_POD_SSH_FLEET_KEY_PREFIX}:{job_batch_id}"


def _due_key(job_batch_id: str) -> str:
    return f"{RENTED_POD_SSH_DUE_KEY_PREFIX}:{job_batch_id}"


def _cycle_id(ctx: Context) -> str:
    # A run outside the sync cycle (the CLI's one-executor verification, the express lane's own
    # batch id) queues under a name no cycle flushes: those entries expire and the streak asks
    # again in the next full cycle.
    return ctx.config.job_batch_id or "no-batch"


def _decode_hash(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        key = key.decode() if isinstance(key, bytes) else key
        value = value.decode() if isinstance(value, bytes) else value
        if isinstance(key, str) and isinstance(value, str):
            out[key] = value
    return out


def _decode(raw: object) -> dict | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class OkMark:
    """The `ok` key: when this validator last saw the pod healthy, and the host's boot_id then."""

    at: str
    boot_id: str | None

    @classmethod
    def load(cls, raw: object) -> OkMark | None:
        """None when the key is absent or unreadable (not JSON, not an object)."""
        value = _decode(raw)
        if value is None:
            return None
        boot_id = value.get("boot_id")
        return cls(
            at=str(value.get("at") or ""), boot_id=boot_id if isinstance(boot_id, str) else None
        )

    def dump(self) -> str:
        return json.dumps({"at": self.at, "boot_id": self.boot_id})


@dataclass(frozen=True)
class FailStreak:
    """The `fail` key: how many consecutive cycles the pod has failed, when the first one was, and
    whether the backend has acknowledged this outage's report (``reported``)."""

    count: int
    first_failed_at: str
    reported: bool = False

    @classmethod
    def load(cls, raw: object, *, now_iso: str) -> FailStreak:
        """The stored streak, or an empty one (count 0, started now) when the key is absent or unreadable.

        A count that is not a non-negative int is treated as 0, so a corrupt value restarts the streak
        instead of raising inside the check.
        """
        value = _decode(raw) or {}
        count = value.get("count", 0)
        first_failed_at = value.get("first_failed_at")
        return cls(
            count=count
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0
            else 0,
            first_failed_at=first_failed_at
            if isinstance(first_failed_at, str) and first_failed_at
            else now_iso,
            reported=value.get("reported") is True,
        )

    def next(self) -> FailStreak:
        return replace(self, count=self.count + 1)

    def dump(self) -> str:
        return json.dumps(
            {
                "count": self.count,
                "first_failed_at": self.first_failed_at,
                "reported": self.reported,
            }
        )


def is_ssh2_identification(line: bytes) -> bool:
    """True for a complete RFC 4253 identification line of protocol version 2.0.

    Complete means terminated by LF (sshd sends CR LF) and no longer than 255 bytes; 2.0 means the
    line starts with ``SSH-2.0-`` and names a software version after it. ``SSH-1.99-``, ``SSH-1.5-``,
    a bare ``SSH-2.0-``, a line cut before its LF, or anything else is not the sshd a renter logs in to.
    """
    if not line.endswith(b"\n") or len(line) > SSH_ID_LINE_MAX:
        return False
    body = line.rstrip(b"\r\n")
    return body.startswith(SSH_ID_PREFIX) and len(body) > len(SSH_ID_PREFIX)


async def read_ssh_identification(reader: asyncio.StreamReader) -> bytes:
    """The server's identification line, or b"" when none arrives within the bounds.

    RFC 4253 §4.2 lets a server send other lines before ``SSH-...``; they are skipped, up to
    ``SSH_PRE_BANNER_LINES_MAX`` of them (Rustam's review, 17 Sep: reading the first line alone
    flagged compliant servers). The reader's ``limit`` bounds every line: a peer whose LF sits past
    byte 255 raises LimitOverrunError instead of growing memory (an LF exactly at index 255 returns
    256 bytes, which ``is_ssh2_identification`` refuses); EOF before the LF raises
    IncompleteReadError (docker-proxy's accept-then-close, or a line cut short). Both are "no
    identification line". The caller holds the deadline.
    """
    for _ in range(SSH_PRE_BANNER_LINES_MAX + 1):
        try:
            line = await reader.readuntil(b"\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, OSError):
            return b""
        if line.startswith(SSH_ID_ANY_VERSION_PREFIX):
            return line
    return b""


async def tcp_connect_fault(
    host: str, port: int, timeout: float, require_ssh2_identification: bool = True
) -> str | None:
    """None when the port accepts AND greets with a complete ``SSH-2.0-`` line; else the fault name.

    The identification line is required because a mapped port is answered by docker-proxy on the
    host: it accepts even when nothing listens inside the container, then closes. sshd sends
    ``SSH-2.0-...CRLF`` first, before the client says anything. The whole line is read (up to the
    LF; the stream buffer is capped at 255 bytes and a line longer than 255 is refused) before it
    is judged: a prefix compared against the first TCP segment alone could call a healthy pod
    unreachable (Rustam's review, 16 Sep).

    ``timeout`` is one deadline for the connect and the read together, so a probe takes at most
    that long (Rustam's review, 17 Sep: two separate timeouts let one probe take twice the value).
    A connect that does not complete by the deadline is ``tcp_timeout``; a peer that accepts and
    then keeps the rest of the deadline without an identification line is ``ssh_banner_missing``.

    With ``require_ssh2_identification`` False (``RENTED_POD_SSH_BANNER_FAULT_ENABLED`` off, the
    default) an accepted connect is health: nothing is read and ``ssh_banner_missing`` is never
    returned, so a validator on this setting sends only the fault names a backend without
    lium-platform#429 knows.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    try:
        async with asyncio.timeout_at(deadline):
            reader, writer = await asyncio.open_connection(host, port, limit=SSH_ID_LINE_MAX)
    except TimeoutError:
        return FAULT_TCP_TIMEOUT
    except OSError:
        return FAULT_TCP_REFUSED
    if not require_ssh2_identification:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
        return None
    try:
        async with asyncio.timeout_at(deadline):
            line = await read_ssh_identification(reader)
    except TimeoutError:
        line = b""
    finally:
        writer.close()
    fault = None if is_ssh2_identification(line) else FAULT_SSH_BANNER_MISSING
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return fault


async def probe_rented_pod_ssh(
    ctx: Context,
    pod: RentedPod,
    ssh_pub_keys: list[str],
) -> RentedPodSshVerdict | None:
    """Judge one RUNNING rented pod from the renter's side and keep the per-pod streak.

    Returns None when the probe is off, and when Redis fails: this is a signal inside a fatal check,
    so a Redis outage skips the signal for the cycle (logged at WARNING) and never reaches the
    verdict. The caller (TenantEnforcementCheck) has already confirmed the container is running
    over the executor's own SSH, so a fault here is the pod, not the host.
    """
    if not settings.RENTED_POD_SSH_PROBE_ENABLED:
        return None

    executor_ip = ctx.executor.address
    faults: list[str] = []
    port_fault: str | None = None
    if pod.ssh_port is not None:
        port_fault = await tcp_connect_fault(
            executor_ip,
            pod.ssh_port,
            settings.RENTED_POD_SSH_PROBE_TIMEOUT_SECONDS,
            require_ssh2_identification=settings.RENTED_POD_SSH_BANNER_FAULT_ENABLED,
        )
        if port_fault:
            faults.append(port_fault)
    if not ssh_pub_keys:
        faults.append(FAULT_AUTHORIZED_KEYS_UNREADABLE)

    try:
        return await _judge_with_streak(ctx, pod, faults, port_fault)
    except REDIS_ERRORS:
        logger.warning(
            _m(
                "RENTED_POD_SSH_PROBE_REDIS_UNAVAILABLE",
                extra=get_extra_info(
                    {**ctx.default_extra, "pod_id": pod.pod_id, "faults": list(faults)}
                ),
            ),
            exc_info=True,
        )
        return None


async def forget_rented_pod_ssh(ctx: Context, pod_id: str) -> None:
    """Drop both marks of a pod whose rental the backend says is closed. Never fatal."""
    store = ctx.services.redis
    try:
        await store.delete(_ok_key(pod_id))
        await store.delete(_fail_key(pod_id))
    except REDIS_ERRORS:
        # The TTL set on every write removes the keys on its own; this only makes it sooner.
        logger.warning(
            _m(
                "RENTED_POD_SSH_PROBE_REDIS_UNAVAILABLE",
                extra=get_extra_info({**ctx.default_extra, "pod_id": pod_id, "on": "forget"}),
            ),
            exc_info=True,
        )


async def _judge_with_streak(
    ctx: Context,
    pod: RentedPod,
    faults: list[str],
    port_fault: str | None,
) -> RentedPodSshVerdict:
    """The Redis-backed part of the probe: the ok mark, the streak, and the one report per outage."""
    store = ctx.services.redis
    ttl = settings.RENTED_POD_SSH_PROBE_STATE_TTL_SECONDS
    boot_id_now = (ctx.state.specs or {}).get("boot_id")
    now_iso = datetime.now(UTC).isoformat()

    # Every transition below is one MULTI/EXEC (Rustam's review, 17 Sep): the ok mark, the streak
    # and the cycle's fleet mark move together or not at all. A connection lost mid-way leaves the
    # previous state whole and the probe skips the cycle (REDIS_UNAVAILABLE, below) instead of
    # leaving a fresh ok mark next to the old streak, or a counted streak next to a stale ok mark.
    if not faults:
        healthy = RedisWrites()
        healthy.set(_ok_key(pod.pod_id), OkMark(at=now_iso, boot_id=boot_id_now).dump(), ex=ttl)
        healthy.delete(_fail_key(pod.pod_id))
        if pod.ssh_port is not None:
            _mark_fleet(ctx, healthy, pod.pod_id, FLEET_MARK_OK)
        await store.write_atomically(healthy)
        return RentedPodSshVerdict(
            pod_id=pod.pod_id,
            container_name=pod.container_name,
            ssh_port=pod.ssh_port,
            healthy=True,
        )

    ok_mark = OkMark.load(await store.get(_ok_key(pod.pod_id)))
    if ok_mark is None:
        # Never seen healthy by this validator: a template without sshd, a pod still coming up, or
        # a deploy that never worked. Not this outage class; nothing is counted, and the pod is not
        # in the fleet share either (three no-sshd templates would otherwise read as an outage).
        return RentedPodSshVerdict(
            pod_id=pod.pod_id,
            container_name=pod.container_name,
            ssh_port=pod.ssh_port,
            healthy=False,
            faults=faults,
        )

    streak = FailStreak.load(await store.get(_fail_key(pod.pod_id)), now_iso=now_iso).next()
    consecutive = streak.count
    first_failed_at = streak.first_failed_at
    unhealthy = RedisWrites()
    if pod.ssh_port is not None:
        # The cycle-end gate reads every counted pod, healthy or not: the share is what tells a
        # validator-side outage (most ports refuse at once) from one pod's. A pod whose port
        # answered but whose authorized_keys is unreadable is a pod fault, not a port fault.
        _mark_fleet(ctx, unhealthy, pod.pod_id, port_fault or FLEET_MARK_OK)
    # The ok mark is what makes the streak count; renew its TTL so an outage longer than the TTL
    # keeps naming the pod in the event instead of silently falling back to RENTED.
    unhealthy.set(_ok_key(pod.pod_id), ok_mark.dump(), ex=ttl)
    unhealthy.set(_fail_key(pod.pod_id), streak.dump(), ex=ttl)

    boot_id_at_ok = ok_mark.boot_id
    boot_id_changed = boot_id_at_ok != boot_id_now if boot_id_at_ok and boot_id_now else None
    threshold = settings.RENTED_POD_SSH_PROBE_CYCLES
    verdict = RentedPodSshVerdict(
        pod_id=pod.pod_id,
        container_name=pod.container_name,
        ssh_port=pod.ssh_port,
        healthy=False,
        faults=faults,
        consecutive_cycles=consecutive,
        first_failed_at=first_failed_at,
        boot_id_changed=boot_id_changed,
        report=consecutive >= threshold,
    )
    if consecutive < threshold or streak.reported or settings.DRY_RUN:
        # Under the threshold, or the backend already acknowledged this outage. DRY_RUN validates
        # without publishing: the event is logged, the backend is not told, and `reported` stays
        # False, so the first live cycle at or past the threshold queues (a dry run consumes nothing).
        await store.write_atomically(unhealthy)
        return verdict

    # Queued, not posted: the cycle-end fleet gate (flush_rented_pod_ssh_reports) decides. No
    # answer from the backend there, or a suppressed cycle, leaves `reported` False and the next
    # cycle queues again. Queued in the same step as the count, so a streak at the threshold is
    # never stored without its report waiting for the gate.
    due_key = _due_key(_cycle_id(ctx))
    unhealthy.hset(
        due_key,
        pod.pod_id,
        json.dumps(
            {
                "ssh_port": verdict.ssh_port,
                "faults": list(verdict.faults),
                "first_failed_at": verdict.first_failed_at or "",
                "consecutive_cycles": verdict.consecutive_cycles,
                "boot_id_changed": verdict.boot_id_changed,
                "boot_id_at_ok": boot_id_at_ok,
                "boot_id_now": boot_id_now,
            }
        ),
    ).expire(due_key, FLEET_KEY_TTL_SECONDS)
    await store.write_atomically(unhealthy)
    return replace(verdict, report_queued=True)


def _mark_fleet(ctx: Context, writes: RedisWrites, pod_id: str, mark: str) -> None:
    key = _fleet_key(_cycle_id(ctx))
    writes.hset(key, pod_id, mark).expire(key, FLEET_KEY_TTL_SECONDS)


async def flush_rented_pod_ssh_reports(
    redis,
    backend,
    job_batch_id: str,
    *,
    validator_outage: bool = False,
) -> FleetGate | None:
    """The cycle-end gate: post the cycle's queued reports only when the fleet says the pods are at fault.

    Called once per cycle from the validator's sync loop, after every executor's task has ended.
    ``validator_outage`` is the cycle's executor-SSH verdict (DAH-2748: most nodes refused the
    validator's own SSH, so its egress is the suspect). The mapped-port share is this module's
    own reading of the same question: above ``RENTED_POD_SSH_PROBE_FLEET_FAIL_MAX`` of the probed
    pods failing at once is our side, not theirs (a fleet smaller than
    ``SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE`` carries no share signal and is gated by
    ``validator_outage`` alone). Either one holds every report back, logged as
    ``RENTED_POD_SSH_PROBE_SUPPRESSED_FLEET`` with the pods it names; nothing is lost, because the
    streaks still read ``reported`` False and queue again next cycle.

    Returns None when the probe is off or Redis failed (logged), else what the gate saw and posted.
    Never raises: a backend or Redis error here is one more cycle of waiting, not a failed cycle.
    """
    if not settings.RENTED_POD_SSH_PROBE_ENABLED:
        return None
    fleet_key, due_key = _fleet_key(job_batch_id), _due_key(job_batch_id)
    extra = {"job_batch_id": job_batch_id}
    try:
        fleet = _decode_hash(await redis.hgetall(fleet_key))
        due = _decode_hash(await redis.hgetall(due_key))
        # Deleted before posting: a crash below costs one cycle (the streaks re-queue), a crash
        # after a post would otherwise post the same outage twice.
        await redis.delete(fleet_key)
        await redis.delete(due_key)
    except REDIS_ERRORS:
        logger.warning(
            _m(
                "RENTED_POD_SSH_PROBE_REDIS_UNAVAILABLE",
                extra=get_extra_info({**extra, "on": "flush"}),
            ),
            exc_info=True,
        )
        return None

    probed = len(fleet)
    failed = sum(1 for mark in fleet.values() if mark != FLEET_MARK_OK)
    gate = FleetGate(
        job_batch_id=job_batch_id,
        probed=probed,
        failed=failed,
        due=sorted(due),
        validator_outage=validator_outage,
    )
    if (
        probed >= SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE
        and gate.fail_share > settings.RENTED_POD_SSH_PROBE_FLEET_FAIL_MAX
    ):
        gate = replace(gate, suppressed_by="mapped_port_share")
    elif validator_outage:
        gate = replace(gate, suppressed_by="validator_outage")

    fleet_fields = {
        **extra,
        "probed": probed,
        "failed": failed,
        "fail_share": round(gate.fail_share, 3),
        "validator_outage": validator_outage,
        "due_pods": gate.due,
    }
    if not due:
        if probed:
            logger.info(_m("RENTED_POD_SSH_PROBE_FLEET", extra=get_extra_info(fleet_fields)))
        return gate
    if gate.suppressed_by:
        logger.warning(
            _m(
                "RENTED_POD_SSH_PROBE_SUPPRESSED_FLEET",
                extra=get_extra_info(
                    {
                        **fleet_fields,
                        "outcome": PROBE_SUPPRESSED_FLEET,
                        "suppressed_by": gate.suppressed_by,
                    }
                ),
            )
        )
        return gate

    # Side by side, as the executor tasks posted them before this gate: this runs on the sync loop
    # ahead of the specs publish, and a backend that is down would otherwise cost one timeout per pod.
    outcomes = await asyncio.gather(
        *(_post_one(redis, backend, pod_id, due[pod_id], extra) for pod_id in gate.due)
    )
    posted = [pod_id for pod_id, ok in zip(gate.due, outcomes, strict=True) if ok]
    if posted:
        logger.info(
            _m(
                "RENTED_POD_SSH_UNREACHABLE_REPORTED",
                extra=get_extra_info({**fleet_fields, "posted_pods": posted}),
            )
        )
    return replace(gate, posted=posted)


def silence_rented_pod_ssh_reports_on_our_own_outage(
    job_results: list[JobResult], gate: FleetGate | None
) -> int:
    """Rewrite to RENTED the cycle's ``RENTED_POD_SSH_UNREACHABLE`` results whose reports the gate held.

    The executor task rendered its event before the cycle-end gate ran, and that event's impact says
    the renter was told. On a suppressed cycle nobody was (Rustam's review, 17 Sep), so before the
    specs publish each such result becomes the RENTED halt it would have been, with the gate's
    verdict and the pods it held under ``what_we_saw[probe_suppressed_fleet]``: the record says the
    validator saw the ports fail and why it did not report them. The same pattern as DAH-2748's
    ``silence_availability_errors_on_our_own_outage``. Score and halt are untouched: the rented
    halt already kept the rented score.

    A result can name several pods of one executor. The pods the gate held move under
    ``probe_suppressed_fleet``; a pod the gate did not hold (its outage was reported in an earlier
    cycle, so ``reported`` is set and it was never due) stays in ``unreachable_pods``, because for
    that renter the impact is true. When nothing stays, the event becomes RENTED; when something
    stays, it keeps its reason and names only the pods whose renters were told (Rustam, round 7).
    Returns how many results were rewritten, for the caller's log line.
    """
    if gate is None or not gate.suppressed_by:
        return 0
    held = set(gate.due)
    unreachable = TenantEnforcementMessages.RENTED_POD_SSH_UNREACHABLE.reason
    rented = TenantEnforcementMessages.ALREADY_RENTED
    rewritten = 0
    for result in job_results:
        event = result.validation_event
        if event is None or event.reason_code != unreachable:
            continue
        pods = [
            pod for pod in event.what_we_saw.get("unreachable_pods") or [] if isinstance(pod, dict)
        ]
        held_pods = [pod for pod in pods if pod.get("pod_id") in held]
        if not held_pods:
            continue
        told_pods = [pod for pod in pods if pod.get("pod_id") not in held]
        what = {key: value for key, value in event.what_we_saw.items() if key != "unreachable_pods"}
        suppressed = {
            "suppressed_by": gate.suppressed_by,
            "probed": gate.probed,
            "failed": gate.failed,
            "fail_share": round(gate.fail_share, 3),
            "unreachable_pods": held_pods,
        }
        if told_pods:
            # Mixed: one pod of this executor was reported in an earlier cycle, another is held now.
            # The event keeps its reason for the pod whose renter was told and stops naming the rest.
            rewritten_event = event.model_copy(
                update={
                    "what_we_saw": {
                        **what,
                        "unreachable_pods": told_pods,
                        PROBE_SUPPRESSED_FLEET: suppressed,
                    }
                }
            )
        else:
            rewritten_event = build_msg(
                event=rented.event,
                reason=rented.reason,
                severity=rented.severity,
                category=rented.category,
                impact=f"Reported rented score={what.get('job_score')} (actual={what.get('actual_score')})",
                remediation="No action needed.",
                what={**what, PROBE_SUPPRESSED_FLEET: suppressed},
                check_id=event.check_id or "",
                pipeline_id=event.pipeline_id,
                ctx=event.context,
            ).model_copy(update={"trace_id": event.trace_id, "when": event.when})
        result.validation_event = rewritten_event
        result.log_text = _m(
            rewritten_event.event, extra=rewritten_event.model_dump()
        ).to_full_string()
        rewritten += 1
    return rewritten


async def _post_one(redis, backend, pod_id: str, raw: str, extra: dict[str, object]) -> bool:
    """POST one queued report; True when the backend answered (the streak is then marked reported)."""
    payload = _decode(raw)
    if payload is None:
        return False
    recorded = await _report_to_backend(backend, pod_id, payload, extra)
    if recorded is None:
        return False
    # The backend answered 200 (recorded or not): this outage is reported. A Redis error on this
    # one write costs one duplicate POST next cycle, which the backend dedupes.
    try:
        stored = await redis.get(_fail_key(pod_id))
        if stored is not None:
            streak = FailStreak.load(stored, now_iso=datetime.now(UTC).isoformat())
            await redis.set(
                _fail_key(pod_id),
                replace(streak, reported=True).dump(),
                ex=settings.RENTED_POD_SSH_PROBE_STATE_TTL_SECONDS,
            )
    except REDIS_ERRORS:
        logger.warning(
            _m(
                "RENTED_POD_SSH_PROBE_REDIS_UNAVAILABLE",
                extra=get_extra_info({**extra, "pod_id": pod_id, "on": "reported_mark"}),
            ),
            exc_info=True,
        )
    return True


async def _report_to_backend(
    backend, pod_id: str, payload: dict, extra: dict[str, object]
) -> bool | None:
    # Never fatal: the verdict is already in the cycle event; a backend that is down or too old
    # (404) must not turn a renter-facing outage report into a validator failure.
    try:
        response = await backend.report_pod_ssh_unreachable(
            pod_id,
            ssh_port=payload.get("ssh_port"),
            faults=list(payload.get("faults") or []),
            first_failed_at=str(payload.get("first_failed_at") or ""),
            consecutive_cycles=int(payload.get("consecutive_cycles") or 0),
            boot_id_changed=payload.get("boot_id_changed"),
            boot_id_at_ok=payload.get("boot_id_at_ok"),
            boot_id_now=payload.get("boot_id_now"),
        )
    except Exception:
        logger.warning(
            _m(
                "RENTED_POD_SSH_UNREACHABLE_REPORT_FAILED",
                extra=get_extra_info({**extra, "pod_id": pod_id}),
            ),
            exc_info=True,
        )
        return None
    return response.recorded if response is not None else None


def verdict_log_fields(verdict: RentedPodSshVerdict) -> dict[str, object]:
    return {
        "pod_id": verdict.pod_id,
        "container_name": verdict.container_name,
        "ssh_port": verdict.ssh_port,
        "faults": list(verdict.faults),
        "consecutive_cycles": verdict.consecutive_cycles,
        "first_failed_at": verdict.first_failed_at,
        "boot_id_changed": verdict.boot_id_changed,
        "report_queued": verdict.report_queued,
    }
