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
healthy once (both signals good; for a banner fault, healthy by the banner rule), so a template
that ships no sshd, or a pod that never came up, is never reported here.
``RENTED_POD_SSH_PROBE_CYCLES`` consecutive unhealthy cycles (default 2, about 30 min) after that
raise ``RENTED_POD_SSH_UNREACHABLE`` and one POST to the backend per
outage: the POST is repeated each cycle until the backend answers 200 (``recorded`` true or false)
with a ``delivery`` other than ``notify_failed`` (lium-platform#429: the renter's mail was refused,
so the next cycle posts again and it is re-sent), and that answer is kept in the streak so the
outage is reported once. The score is not changed by this module: with
``RENTED_POD_SSH_ENFORCEMENT_ENABLED`` on (DAH-2255, off by default) the check reads ``is_enforced``
and fails the cycle itself once the streak reaches ``enforce_after_cycles()``.

The POST is deferred to the end of the cycle and gated by the fleet: a validator whose own network
fails sees every mapped port refuse at once, and per-pod reporting would tell every healthy renter
their pod is down. Each probe writes its mapped-port result into a
per-cycle Redis hash; a pod at the threshold queues its report into a second one. At the cycle's
end ``flush_rented_pod_ssh_reports`` reads both: when the share of probed pods failing the
mapped-port check is above ``RENTED_POD_SSH_PROBE_FLEET_FAIL_MAX`` (default 0.5, on a fleet of at
least ``SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE`` pods), or the cycle's executor-SSH check already
judged the validator to be the outage (DAH-2748), the queued reports are logged as
``RENTED_POD_SSH_PROBE_SUPPRESSED_FLEET`` and no report is POSTed, so no renter is told. The
streaks keep ``reported`` False, so the outage is queued again next cycle and reported once the
fleet reads clean. Enforcement follows the backend accept, not the mail: ``is_enforced`` is true
only when the streak is ``backend_accepted`` (a 200, including ``notify_failed``), so a held cycle
never zeroes the node and a refused mail does not protect the provider. The gate also stores its
verdict (``RENTED_POD_SSH_LAST_GATE_KEY``): while the last gate held the reports as our own outage,
an accepted outage is not enforced either, so our outage zeroes a node for one cycle at most (the
gate runs after the cycle's checks). The per-executor ``RENTED_POD_SSH_UNREACHABLE`` events of a
suppressed cycle were rendered before the gate ran and name a pod outage the gate then judged to be ours; the sync
loop passes the gate to
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
from typing import TYPE_CHECKING

import redis.exceptions
from protocol.vc_protocol.compute_requests import (
    SSH_UNREACHABLE_DELIVERY_NOTIFY_FAILED,
    PodSshUnreachableResponse,
    RentedPod,
)
from pydantic import BaseModel, Field, ValidationError

from core.config import settings
from core.utils import _m, get_extra_info

from ...redis_service import RedisWrites
from ..availability import SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE
from ..messages import TenantEnforcementMessages
from ..models import JobResult, ValidationEvent, build_msg
from ..pipeline import Context
from .ssh_identification import SSH_ID_LINE_MAX, is_ssh2_identification, read_ssh_identification

if TYPE_CHECKING:
    from clients.backend_client import BackendClient

    from ...redis_service import RedisService

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
# The last cycle-end gate's `suppressed_by` ("" when it held nothing). `is_enforced` reads it: a
# validator the last gate judged to be the outage does not zero a pod. Expires with the fleet keys.
RENTED_POD_SSH_LAST_GATE_KEY = "rented_pod_ssh_last_gate"
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
_PORT_FAULTS = frozenset({FAULT_TCP_REFUSED, FAULT_TCP_TIMEOUT, FAULT_SSH_BANNER_MISSING})
# The one pod status the probe judges: the backend lists rebooting, failed and pending pods too, and
# its ssh-unreachable route answers 409 for any of them (lium-platform#429). A backend that predates
# the field sends no status, and every listed pod is judged as before.
POD_STATUS_RUNNING = "RUNNING"
# A host `boot_id` is a 36-char UUID read off the executor's specs (provider-controlled input); the
# backend bounds both boot_id fields to 64 chars and answers 422 above that, so the bound is applied
# here before the value is stored or sent (PR_PROCESS §5 bounded input).
BOOT_ID_MAX = 64

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
    # cycle's event names the pod. On every such cycle until the backend answers 200 with a
    # delivery other than notify_failed (FailStreak.reported) the report is queued for the
    # cycle-end fleet gate; report_queued says whether THIS cycle queued it. Whether it was posted
    # is the flush's log line, not the verdict's.
    report: bool = False
    report_queued: bool = False
    # Backend accepted this outage (HTTP 200), including a ``notify_failed`` delivery.
    # Mail retry still queues; enforcement reads this, not ``reported``.
    backend_accepted: bool = False
    backend_accepted_faults: list[str] = field(default_factory=list)
    # The last cycle-end gate held the reports as our own outage.
    last_gate_suppressed: bool = False


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


def enforce_after_cycles() -> int:
    """The streak at which ``RENTED_POD_SSH_ENFORCEMENT_ENABLED`` fails the rented-state check (DAH-2255).

    ``RENTED_POD_SSH_ENFORCE_AFTER_CYCLES`` when set, else the notify threshold
    ``RENTED_POD_SSH_PROBE_CYCLES``; the settings refuse a value below the notify threshold at startup.
    """
    configured = settings.RENTED_POD_SSH_ENFORCE_AFTER_CYCLES
    return settings.RENTED_POD_SSH_PROBE_CYCLES if configured is None else configured


def is_enforced(verdict: RentedPodSshVerdict) -> bool:
    """True when this verdict fails the rented-state check for the cycle (DAH-2255).

    Only with ``RENTED_POD_SSH_ENFORCEMENT_ENABLED`` on, only for an unhealthy pod, only once its
    streak has reached ``enforce_after_cycles()``, only after the backend accepted the outage report
    (``verdict.backend_accepted``), and not while the last cycle-end gate held the reports as our
    own outage (``verdict.last_gate_suppressed``). Mail delivery is separate: a ``notify_failed``
    200 still accepts, and a cycle that only queued the notice — including a validator-side outage
    the fleet gate holds — does not zero the node. A port-fault accept enforces; a keys-only accept
    enforces only a keys-only cycle, never a later port fault the backend did not accept; a streak
    with no stored accept faults enforces nothing. A renter who deletes ``authorized_keys`` (that
    fault alone, host ``boot_id`` unchanged) is not enforced, whether that is the accepted report's
    fault or this cycle's after a port-fault accept: the provider cannot restore the keys. A pod
    never seen healthy carries no streak (``consecutive_cycles`` 0), a Redis outage yields no
    verdict at all, and the flag off leaves the check with DAH-2870's record-and-report behaviour:
    none of those is enforced.
    """
    if not settings.RENTED_POD_SSH_ENFORCEMENT_ENABLED or verdict.healthy:
        return False
    if verdict.consecutive_cycles < enforce_after_cycles():
        return False
    if not verdict.backend_accepted or verdict.last_gate_suppressed:
        return False
    backend_accepted_faults = set(verdict.backend_accepted_faults)
    current_faults = set(verdict.faults)
    keys_only_now = FAULT_AUTHORIZED_KEYS_UNREADABLE in current_faults and not (
        _PORT_FAULTS & current_faults
    )
    if _PORT_FAULTS & backend_accepted_faults:
        # The boot rule reads this cycle's faults too: keys alone and no reboot is the renter's doing.
        return verdict.boot_id_changed is True if keys_only_now else True
    # A keys-only accept covers a keys-only outage, never a later port fault.
    if FAULT_AUTHORIZED_KEYS_UNREADABLE in backend_accepted_faults and keys_only_now:
        return verdict.boot_id_changed is True
    return False


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


def _decode(raw: object) -> dict[str, object] | None:
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


def bounded_boot_id(value: object) -> str | None:
    """The boot_id as a string of at most BOOT_ID_MAX chars, or None for anything that is not a string."""
    if not isinstance(value, str) or not value:
        return None
    return value[:BOOT_ID_MAX]


@dataclass(frozen=True)
class OkMark:
    """The `ok` key: when this validator last saw the pod healthy, and the host's boot_id then.

    ``banner_seen``: that healthy cycle judged the port by the SSH-2.0 line
    (RENTED_POD_SSH_BANNER_FAULT_ENABLED on), not by the connect alone.
    """

    at: str
    boot_id: str | None
    banner_seen: bool = False

    @classmethod
    def load(cls, raw: object) -> OkMark | None:
        """None when the key is absent or unreadable (not JSON, not an object)."""
        value = _decode(raw)
        if value is None:
            return None
        return cls(
            at=str(value.get("at") or ""),
            boot_id=bounded_boot_id(value.get("boot_id")),
            banner_seen=value.get("banner_seen") is True,
        )

    def dump(self) -> str:
        return json.dumps({"at": self.at, "boot_id": self.boot_id, "banner_seen": self.banner_seen})


@dataclass(frozen=True)
class FailStreak:
    """The `fail` key: how many consecutive cycles the pod has failed, when the first one was,
    whether the backend accepted this outage (``backend_accepted``, any 200), which faults that accept
    named (``backend_accepted_faults``), and whether the renter was told (``reported`` — not set on ``notify_failed``)."""

    count: int
    first_failed_at: str
    reported: bool = False
    backend_accepted: bool = False
    backend_accepted_faults: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, raw: object, *, now_iso: str) -> FailStreak:
        """The stored streak, or an empty one (count 0, started now) when the key is absent or unreadable.

        A count that is not a non-negative int is treated as 0, so a corrupt value restarts the streak
        instead of raising inside the check.
        """
        value = _decode(raw) or {}
        count = value.get("count", 0)
        first_failed_at = value.get("first_failed_at")
        backend_accepted_faults = [
            item for item in (value.get("accepted_faults") or []) if isinstance(item, str)
        ]
        return cls(
            count=count
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0
            else 0,
            first_failed_at=first_failed_at
            if isinstance(first_failed_at, str) and first_failed_at
            else now_iso,
            reported=value.get("reported") is True,
            backend_accepted=value.get("accepted") is True,
            backend_accepted_faults=backend_accepted_faults,
        )

    def plus_one_cycle(self) -> FailStreak:
        return replace(self, count=self.count + 1)

    def dump(self) -> str:
        return json.dumps(
            {
                "count": self.count,
                "first_failed_at": self.first_failed_at,
                "reported": self.reported,
                "accepted": self.backend_accepted,
                "accepted_faults": list(self.backend_accepted_faults),
            }
        )


class DueReport(BaseModel):
    """One queued report in the cycle's `due` hash, validated when read back before the POST, with
    the backend's own bounds, so a corrupt entry is dropped instead of answered with a 422."""

    ssh_port: int | None = Field(ge=1, le=65535)
    faults: list[str] = Field(min_length=1)
    first_failed_at: str = Field(min_length=1)
    consecutive_cycles: int = Field(ge=1)
    boot_id_changed: bool | None
    boot_id_at_ok: str | None = Field(max_length=BOOT_ID_MAX)
    boot_id_now: str | None = Field(max_length=BOOT_ID_MAX)

    @classmethod
    def load(cls, raw: str) -> DueReport | None:
        try:
            return cls.model_validate_json(raw)
        except ValidationError:
            return None


async def tcp_connect_fault(
    host: str, port: int, timeout: float, require_ssh2_identification: bool = True
) -> str | None:
    """None when the port accepts AND greets with a complete ``SSH-2.0-`` line; else the fault name.

    The identification line is required because a mapped port is answered by docker-proxy on the
    host: it accepts even when nothing listens inside the container, then closes. sshd sends
    ``SSH-2.0-...CRLF`` first, before the client says anything. The whole line is read (up to the
    LF; the stream buffer is capped at 255 bytes and a line longer than 255 is refused) before it
    is judged: a prefix compared against the first TCP segment alone could call a healthy pod
    unreachable.

    ``timeout`` is one deadline for the connect and the read together, so a probe takes at most
    that long (two separate timeouts would let one probe take twice the value).
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
    if pod.status is not None and pod.status != POD_STATUS_RUNNING:
        # Rebooting, failed or pending as the backend records it: not this outage class, and a
        # report for it is a 409. Nothing is counted or marked this cycle; the streak, if any,
        # resumes when the pod is RUNNING again.
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
    if not any(key.strip() for key in ssh_pub_keys):
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
    boot_id_now = bounded_boot_id((ctx.state.specs or {}).get("boot_id"))
    now_iso = datetime.now(UTC).isoformat()

    # Every transition below is one MULTI/EXEC: the ok mark, the streak and the cycle's fleet mark
    # move together or not at all. A connection lost mid-way leaves the previous state whole and
    # the probe skips the cycle (REDIS_UNAVAILABLE, below) instead of leaving a fresh ok mark next
    # to the old streak, or a counted streak next to a stale ok mark.
    if not faults:
        return await _record_healthy_cycle(ctx, pod, boot_id_now, now_iso)

    ok_mark = OkMark.load(await store.get(_ok_key(pod.pod_id)))
    if ok_mark is None or (port_fault == FAULT_SSH_BANNER_MISSING and not ok_mark.banner_seen):
        # Never seen healthy by this validator: a template without sshd, a pod still coming up, or
        # a deploy that never worked. Not this outage class; nothing is counted, and the pod is not
        # in the fleet share either (three no-sshd templates would otherwise read as an outage).
        # A mark written by the connect-only rule never saw sshd either: docker-proxy accepts
        # without it, so a banner fault against that mark is the same no-sshd template.
        return RentedPodSshVerdict(
            pod_id=pod.pod_id,
            container_name=pod.container_name,
            ssh_port=pod.ssh_port,
            healthy=False,
            faults=faults,
        )

    streak = FailStreak.load(
        await store.get(_fail_key(pod.pod_id)), now_iso=now_iso
    ).plus_one_cycle()
    consecutive = streak.count
    first_failed_at = streak.first_failed_at
    unhealthy_writes = RedisWrites()
    if pod.ssh_port is not None:
        # The cycle-end gate reads every counted pod, healthy or not: the share is what tells a
        # validator-side outage (most ports refuse at once) from one pod's. A pod whose port
        # answered but whose authorized_keys is unreadable is a pod fault, not a port fault.
        _mark_fleet(ctx, unhealthy_writes, pod.pod_id, port_fault or FLEET_MARK_OK)
    # The ok mark is what makes the streak count; renew its TTL so an outage longer than the TTL
    # keeps naming the pod in the event instead of silently falling back to RENTED.
    unhealthy_writes.set(_ok_key(pod.pod_id), ok_mark.dump(), ex=ttl)
    unhealthy_writes.set(_fail_key(pod.pod_id), streak.dump(), ex=ttl)

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
        backend_accepted=streak.backend_accepted,
        backend_accepted_faults=list(streak.backend_accepted_faults),
        last_gate_suppressed=bool(await store.get(RENTED_POD_SSH_LAST_GATE_KEY)),
    )
    if consecutive < threshold or streak.reported or settings.DRY_RUN:
        # Under the threshold, or the backend already acknowledged this outage. DRY_RUN validates
        # without publishing: the event is logged, the backend is not told, and `reported` stays
        # False, so the first live cycle at or past the threshold queues (a dry run consumes nothing).
        await store.write_atomically(unhealthy_writes)
        return verdict

    _queue_report_for_fleet_gate(ctx, unhealthy_writes, verdict, boot_id_at_ok, boot_id_now)
    await store.write_atomically(unhealthy_writes)
    return replace(verdict, report_queued=True)


async def _record_healthy_cycle(
    ctx: Context, pod: RentedPod, boot_id_now: str | None, now_iso: str
) -> RentedPodSshVerdict:
    """Renew the ok mark and end any streak: the pod counts from here on."""
    healthy_mark = OkMark(
        at=now_iso,
        boot_id=boot_id_now,
        banner_seen=pod.ssh_port is not None and settings.RENTED_POD_SSH_BANNER_FAULT_ENABLED,
    )
    healthy_writes = RedisWrites()
    healthy_writes.set(
        _ok_key(pod.pod_id), healthy_mark.dump(), ex=settings.RENTED_POD_SSH_PROBE_STATE_TTL_SECONDS
    )
    healthy_writes.delete(_fail_key(pod.pod_id))
    if pod.ssh_port is not None:
        _mark_fleet(ctx, healthy_writes, pod.pod_id, FLEET_MARK_OK)
    await ctx.services.redis.write_atomically(healthy_writes)
    return RentedPodSshVerdict(
        pod_id=pod.pod_id,
        container_name=pod.container_name,
        ssh_port=pod.ssh_port,
        healthy=True,
    )


def _queue_report_for_fleet_gate(
    ctx: Context,
    writes: RedisWrites,
    verdict: RentedPodSshVerdict,
    boot_id_at_ok: str | None,
    boot_id_now: str | None,
) -> None:
    """Queue the outage report for the cycle-end fleet gate (flush_rented_pod_ssh_reports).

    Queued, not posted: the gate decides. No answer from the backend there, or a suppressed cycle,
    leaves ``reported`` False and the next cycle queues again. Queued in the same step as the
    count, so a streak at the threshold is never stored without its report waiting for the gate.
    """
    due_key = _due_key(_cycle_id(ctx))
    report = DueReport(
        ssh_port=verdict.ssh_port,
        faults=list(verdict.faults),
        first_failed_at=verdict.first_failed_at or "",
        consecutive_cycles=verdict.consecutive_cycles,
        boot_id_changed=verdict.boot_id_changed,
        boot_id_at_ok=boot_id_at_ok,
        boot_id_now=boot_id_now,
    )
    writes.hset(due_key, verdict.pod_id, report.model_dump_json()).expire(
        due_key, FLEET_KEY_TTL_SECONDS
    )


def _mark_fleet(ctx: Context, writes: RedisWrites, pod_id: str, mark: str) -> None:
    key = _fleet_key(_cycle_id(ctx))
    writes.hset(key, pod_id, mark).expire(key, FLEET_KEY_TTL_SECONDS)


def judge_fleet_gate(
    fleet: dict[str, str], due: dict[str, str], job_batch_id: str, *, validator_outage: bool
) -> FleetGate:
    """The gate's verdict on one cycle: who was probed, who failed, and whether the reports are held.

    ``fleet`` is the cycle's per-pod mapped-port marks, ``due`` its queued reports. The share of
    probed pods failing above ``RENTED_POD_SSH_PROBE_FLEET_FAIL_MAX`` is our side, not theirs
    (a fleet under ``SMALLEST_FLEET_THAT_CAN_SHOW_AN_OUTAGE`` carries no share signal); else the
    cycle's executor-SSH verdict decides. Pure: no Redis, no backend.
    """
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
        return replace(gate, suppressed_by="mapped_port_share")
    if validator_outage:
        return replace(gate, suppressed_by="validator_outage")
    return gate


async def flush_rented_pod_ssh_reports(
    redis: RedisService,
    backend: BackendClient,
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

    The verdict is also stored under ``RENTED_POD_SSH_LAST_GATE_KEY`` for the next cycle's
    ``is_enforced``. Returns None when the probe is off or Redis failed (logged), else what the gate
    saw and posted.
    Never raises: a backend or Redis error here is one more cycle of waiting, not a failed cycle.
    """
    if not settings.RENTED_POD_SSH_PROBE_ENABLED:
        return None
    fleet_key, due_key = _fleet_key(job_batch_id), _due_key(job_batch_id)
    extra = {"job_batch_id": job_batch_id}
    try:
        fleet = _decode_hash(await redis.hgetall(fleet_key))
        due = _decode_hash(await redis.hgetall(due_key))
        gate = judge_fleet_gate(fleet, due, job_batch_id, validator_outage=validator_outage)
        await redis.set(
            RENTED_POD_SSH_LAST_GATE_KEY, gate.suppressed_by or "", ex=FLEET_KEY_TTL_SECONDS
        )
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

    gate_log_fields = {
        **extra,
        "probed": gate.probed,
        "failed": gate.failed,
        "fail_share": round(gate.fail_share, 3),
        "validator_outage": validator_outage,
        "due_pods": gate.due,
    }
    if not due:
        if gate.probed:
            logger.info(_m("RENTED_POD_SSH_PROBE_FLEET", extra=get_extra_info(gate_log_fields)))
        return gate
    if gate.suppressed_by:
        logger.warning(
            _m(
                "RENTED_POD_SSH_PROBE_SUPPRESSED_FLEET",
                extra=get_extra_info(
                    {
                        **gate_log_fields,
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
                extra=get_extra_info({**gate_log_fields, "posted_pods": posted}),
            )
        )
    return replace(gate, posted=posted)


def silence_rented_pod_ssh_reports_on_our_own_outage(
    job_results: list[JobResult], gate: FleetGate | None
) -> int:
    """Rewrite to RENTED the cycle's ``RENTED_POD_SSH_UNREACHABLE`` results whose reports the gate held.

    The executor task rendered its event before the cycle-end gate ran, and that event names a pod
    outage with its notice queued. On a suppressed cycle the outage was ours and nobody is told,
    so before the specs publish each such result becomes the RENTED halt it would have been, with the gate's
    verdict and the pods it held under ``what_we_saw[probe_suppressed_fleet]``: the record says the
    validator saw the ports fail and why it did not report them. The same pattern as DAH-2748's
    ``silence_availability_errors_on_our_own_outage``. Score and halt are untouched: the rented
    halt already kept the rented score.

    A result can name several pods of one executor. The pods the gate held move under
    ``probe_suppressed_fleet``; a pod the gate did not hold (its outage was reported in an earlier
    cycle, so ``reported`` is set and it was never due) stays in ``unreachable_pods``, because that
    pod's outage is real and already on record. When nothing stays, the event becomes RENTED; when
    something stays, it keeps its reason and names only the pods whose outage stands.
    An enforced event (DAH-2255, ``what_we_saw.enforced``) is a later cycle whose backend already
    accepted the report: it failed at score 0 and is never rewritten to RENTED. It keeps reason,
    impact and pods and gains the gate's verdict under ``probe_suppressed_fleet``. The stored gate
    verdict stops enforcement from the next cycle on, so this is at most the first cycle of our
    outage. A first-threshold cycle the gate holds is not enforced (``is_enforced`` needs the
    accept), so it takes the RENTED rewrite.
    Returns how many results were rewritten, for the caller's log line.
    """
    if gate is None or not gate.suppressed_by:
        return 0
    held_pod_ids = set(gate.due)
    rewritten = 0
    for result in job_results:
        rewritten_event = _event_without_held_pods(result.validation_event, gate, held_pod_ids)
        if rewritten_event is None:
            continue
        result.validation_event = rewritten_event
        result.log_text = _m(
            rewritten_event.event, extra=rewritten_event.model_dump()
        ).to_full_string()
        rewritten += 1
    return rewritten


def _event_without_held_pods(
    event: ValidationEvent | None, gate: FleetGate, held_pod_ids: set[str]
) -> ValidationEvent | None:
    """The RENTED_POD_SSH_UNREACHABLE event with the gate's held pods moved under
    ``probe_suppressed_fleet``, or None when the event names no held pod."""
    if (
        event is None
        or event.reason_code != TenantEnforcementMessages.RENTED_POD_SSH_UNREACHABLE.reason
    ):
        return None
    pods = [pod for pod in event.what_we_saw.get("unreachable_pods") or [] if isinstance(pod, dict)]
    held_pods = [pod for pod in pods if pod.get("pod_id") in held_pod_ids]
    if not held_pods:
        return None
    already_reported_pods = [pod for pod in pods if pod.get("pod_id") not in held_pod_ids]
    what = {key: value for key, value in event.what_we_saw.items() if key != "unreachable_pods"}
    gate_verdict = {
        "suppressed_by": gate.suppressed_by,
        "probed": gate.probed,
        "failed": gate.failed,
        "fail_share": round(gate.fail_share, 3),
        "unreachable_pods": held_pods,
    }
    if event.what_we_saw.get("enforced") is True:
        # DAH-2255: the check failed this cycle (score 0, verified job cleared) — that is not the
        # rented halt, so the event keeps its reason, impact and pods; the gate's verdict rides
        # along so the record says the zero fell in a cycle whose renter notice was held. The
        # score and the job reset were applied before the gate ran, so the record keeps them.
        return event.model_copy(
            update={"what_we_saw": {**event.what_we_saw, PROBE_SUPPRESSED_FLEET: gate_verdict}}
        )
    if already_reported_pods:
        # Mixed: one pod of this executor was reported in an earlier cycle, another is held now.
        # The event keeps its reason for the pod whose outage stands and stops naming the rest;
        # nothing was queued for that pod (it was never due), so the impact says so too.
        return event.model_copy(
            update={
                "impact": TenantEnforcementMessages.RENTED_POD_SSH_UNREACHABLE_NOT_QUEUED_IMPACT,
                "what_we_saw": {
                    **what,
                    "unreachable_pods": already_reported_pods,
                    PROBE_SUPPRESSED_FLEET: gate_verdict,
                },
            }
        )
    rented = TenantEnforcementMessages.ALREADY_RENTED
    return build_msg(
        event=rented.event,
        reason=rented.reason,
        severity=rented.severity,
        category=rented.category,
        impact=f"Reported rented score={what.get('job_score')} (actual={what.get('actual_score')})",
        remediation="No action needed.",
        what={**what, PROBE_SUPPRESSED_FLEET: gate_verdict},
        check_id=event.check_id or "",
        pipeline_id=event.pipeline_id,
        ctx=event.context,
    ).model_copy(update={"trace_id": event.trace_id, "when": event.when})


async def _post_one(
    redis: RedisService, backend: BackendClient, pod_id: str, raw: str, extra: dict[str, object]
) -> bool:
    """POST one queued report; True when the renter was told (or nothing was due).

    A 200 marks the streak accepted (enforcement can proceed) and stores the faults that
    accept named. A ``notify_failed`` answer (lium-platform#429: the mail was refused) leaves
    ``reported`` False so the next cycle posts again and the mail is re-sent.
    """
    report = DueReport.load(raw)
    if report is None:
        return False
    response = await _report_to_backend(backend, pod_id, report, extra)
    if response is None:
        return False
    mail_failed = response.delivery == SSH_UNREACHABLE_DELIVERY_NOTIFY_FAILED
    if mail_failed:
        logger.warning(
            _m(
                "RENTED_POD_SSH_UNREACHABLE_NOTIFY_FAILED",
                extra=get_extra_info({**extra, "pod_id": pod_id, "recorded": response.recorded}),
            )
        )
    # Backend accepted (200). A Redis error on this write costs one duplicate POST next cycle,
    # which the backend dedupes; ``backend_accepted`` then stays off until that retry lands.
    try:
        stored = await redis.get(_fail_key(pod_id))
        if stored is not None:
            streak = FailStreak.load(stored, now_iso=datetime.now(UTC).isoformat())
            await redis.set(
                _fail_key(pod_id),
                replace(
                    streak,
                    backend_accepted=True,
                    backend_accepted_faults=list(report.faults),
                    reported=not mail_failed,
                ).dump(),
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
    return not mail_failed


async def _report_to_backend(
    backend: BackendClient, pod_id: str, report: DueReport, extra: dict[str, object]
) -> PodSshUnreachableResponse | None:
    # Never fatal: the verdict is already in the cycle event; a backend that is down or too old
    # (404) must not turn a renter-facing outage report into a validator failure.
    try:
        response = await backend.report_pod_ssh_unreachable(
            pod_id,
            ssh_port=report.ssh_port,
            faults=report.faults,
            first_failed_at=report.first_failed_at,
            consecutive_cycles=report.consecutive_cycles,
            boot_id_changed=report.boot_id_changed,
            boot_id_at_ok=report.boot_id_at_ok,
            boot_id_now=report.boot_id_now,
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
    return response


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
