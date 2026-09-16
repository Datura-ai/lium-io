"""Renter-side reachability of a RUNNING rented pod (DAH-2870, B-71).

A host reboot lets dockerd bring the rental container back on its own. That restart path never
mounts the encrypted volume and sometimes never starts sshd, so the pod's SSH port refuses
(ticket-0326) or accepts and then asks for a password because ``/root/.ssh/authorized_keys`` is
unreadable (ticket-0247). The container is running, so ``TenantEnforcementCheck`` scored the node
1.0 and the renter kept paying for a pod nobody could enter.

This module judges what the renter sees, from outside the container, each cycle:

* a TCP connect to the pod's mapped SSH port (``RentedPod.ssh_port``, sent by the backend) that
  must answer with an ``SSH-`` banner: docker-proxy accepts on the host while sshd inside is down,
  so an accept alone says nothing (ticket-0326 is exactly that case);
* the ``authorized_keys`` read the check already does (empty = the mount is missing).

State lives in Redis, one key pair per pod. A pod is judged only after this validator has seen it
healthy once (both signals good), so a template that ships no sshd, or a pod that never came up,
is never reported here. ``RENTED_POD_SSH_PROBE_CYCLES`` consecutive unhealthy cycles (default 2,
about 30 min) after that raise ``RENTED_POD_SSH_UNREACHABLE`` and one POST to the backend per
outage: the POST is repeated each cycle until the backend answers 200 (``recorded`` true or false),
and that answer is kept in the streak so the outage is reported once. The score is not changed by
this module.

Redis is an input to this signal, never to the check's verdict: when Redis fails, the probe logs
``RENTED_POD_SSH_PROBE_REDIS_UNAVAILABLE`` and returns None for that pod and cycle, exactly as when
the probe is disabled. Both keys carry a TTL (``RENTED_POD_SSH_PROBE_STATE_TTL_SECONDS``, renewed on
every probe of the pod) and are deleted by ``forget_rented_pod_ssh`` when the rental has closed.
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

from ..pipeline import Context

logger = logging.getLogger(__name__)

# Redis keys. `ok` carries the boot_id seen when the pod was last healthy, so a later failure can
# say whether the host rebooted in between. `fail` carries the streak.
RENTED_POD_SSH_OK_KEY_PREFIX = "rented_pod_ssh_ok"
RENTED_POD_SSH_FAIL_KEY_PREFIX = "rented_pod_ssh_fail"

FAULT_TCP_REFUSED = "tcp_refused"
FAULT_TCP_TIMEOUT = "tcp_timeout"
# The port accepted but nothing that speaks SSH is behind it: docker-proxy took the connection and
# closed it (sshd not running in the container), or something else answered.
FAULT_SSH_BANNER_MISSING = "ssh_banner_missing"
FAULT_AUTHORIZED_KEYS_UNREADABLE = "authorized_keys_unreadable"
SSH_BANNER_PREFIX = b"SSH-"

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
    # cycle's event names the pod. The backend is POSTed on every such cycle until it answers 200
    # once (FailStreak.reported); reported_to_backend says whether THIS cycle posted.
    report: bool = False
    reported_to_backend: bool = False
    backend_recorded: bool | None = None


def _ok_key(pod_id: str) -> str:
    return f"{RENTED_POD_SSH_OK_KEY_PREFIX}:{pod_id}"


def _fail_key(pod_id: str) -> str:
    return f"{RENTED_POD_SSH_FAIL_KEY_PREFIX}:{pod_id}"


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


async def tcp_connect_fault(host: str, port: int, timeout: float) -> str | None:
    """None when the port accepts a TCP connection AND greets with an SSH banner; else the fault name.

    The banner is required because a mapped port is answered by docker-proxy on the host: it
    accepts even when nothing listens inside the container, then closes. sshd sends
    ``SSH-2.0-...`` first, before the client says anything, so one read tells the two apart.
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
    except TimeoutError:
        return FAULT_TCP_TIMEOUT
    except OSError:
        return FAULT_TCP_REFUSED
    try:
        # RFC 4253 caps the version line at 255 bytes; a bounded read never grows a buffer on what a
        # peer chooses to send, and sshd sends the line first, before the client says anything.
        banner = await asyncio.wait_for(reader.read(255), timeout=timeout)
    except (TimeoutError, OSError):
        banner = b""
    finally:
        writer.close()
    fault = None if banner.startswith(SSH_BANNER_PREFIX) else FAULT_SSH_BANNER_MISSING
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
    if pod.ssh_port is not None:
        fault = await tcp_connect_fault(
            executor_ip, pod.ssh_port, settings.RENTED_POD_SSH_PROBE_TIMEOUT_SECONDS
        )
        if fault:
            faults.append(fault)
    if not ssh_pub_keys:
        faults.append(FAULT_AUTHORIZED_KEYS_UNREADABLE)

    try:
        return await _judge_with_streak(ctx, pod, faults)
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
) -> RentedPodSshVerdict:
    """The Redis-backed part of the probe: the ok mark, the streak, and the one report per outage."""
    store = ctx.services.redis
    ttl = settings.RENTED_POD_SSH_PROBE_STATE_TTL_SECONDS
    boot_id_now = (ctx.state.specs or {}).get("boot_id")
    now_iso = datetime.now(UTC).isoformat()

    if not faults:
        await store.set(_ok_key(pod.pod_id), OkMark(at=now_iso, boot_id=boot_id_now).dump(), ex=ttl)
        await store.delete(_fail_key(pod.pod_id))
        return RentedPodSshVerdict(
            pod_id=pod.pod_id,
            container_name=pod.container_name,
            ssh_port=pod.ssh_port,
            healthy=True,
        )

    ok_mark = OkMark.load(await store.get(_ok_key(pod.pod_id)))
    if ok_mark is None:
        # Never seen healthy by this validator: a template without sshd, a pod still coming up, or
        # a deploy that never worked. Not this outage class; nothing is counted.
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
    # The ok mark is what makes the streak count; renew its TTL so an outage longer than the TTL
    # keeps naming the pod in the event instead of silently falling back to RENTED.
    await store.set(_ok_key(pod.pod_id), ok_mark.dump(), ex=ttl)
    await store.set(_fail_key(pod.pod_id), streak.dump(), ex=ttl)

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
        # False, so the first live cycle at or past the threshold posts (a dry run consumes nothing).
        return verdict

    recorded = await _report_to_backend(ctx, verdict, boot_id_at_ok, boot_id_now)
    if recorded is not None:
        # The backend answered 200 (recorded or not): this outage is reported. No answer (down,
        # non-200 such as a 404 from a backend too old, timeout) leaves `reported` False and the
        # next cycle posts again. A Redis error on this one write must not drop the verdict the
        # POST already went out for: it costs one duplicate POST next cycle, which the backend dedupes.
        try:
            await store.set(_fail_key(pod.pod_id), replace(streak, reported=True).dump(), ex=ttl)
        except REDIS_ERRORS:
            logger.warning(
                _m(
                    "RENTED_POD_SSH_PROBE_REDIS_UNAVAILABLE",
                    extra=get_extra_info(
                        {**ctx.default_extra, "pod_id": pod.pod_id, "on": "reported_mark"}
                    ),
                ),
                exc_info=True,
            )
    return replace(verdict, reported_to_backend=True, backend_recorded=recorded)


async def _report_to_backend(
    ctx: Context,
    verdict: RentedPodSshVerdict,
    boot_id_at_ok: str | None,
    boot_id_now: str | None,
) -> bool | None:
    # Never fatal: the verdict is already in the cycle event; a backend that is down or too old
    # (404) must not turn a renter-facing outage report into a validator failure.
    try:
        response = await ctx.services.backend.report_pod_ssh_unreachable(
            verdict.pod_id,
            ssh_port=verdict.ssh_port,
            faults=verdict.faults,
            first_failed_at=verdict.first_failed_at or "",
            consecutive_cycles=verdict.consecutive_cycles,
            boot_id_changed=verdict.boot_id_changed,
            boot_id_at_ok=boot_id_at_ok,
            boot_id_now=boot_id_now,
        )
    except Exception:
        logger.warning(
            _m(
                "RENTED_POD_SSH_UNREACHABLE_REPORT_FAILED",
                extra=get_extra_info({**ctx.default_extra, "pod_id": verdict.pod_id}),
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
        "reported_to_backend": verdict.reported_to_backend,
        "backend_recorded": verdict.backend_recorded,
    }
