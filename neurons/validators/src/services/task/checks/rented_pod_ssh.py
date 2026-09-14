"""Renter-side reachability of a RUNNING rented pod (DAH-2870, B-71).

A host reboot lets dockerd bring the rental container back on its own. That restart path never
mounts the encrypted volume and sometimes never starts sshd, so the pod's SSH port refuses
(ticket-0326) or accepts and then asks for a password because ``/root/.ssh/authorized_keys`` is
unreadable (ticket-0247). The container is running, so ``TenantEnforcementCheck`` scored the node
1.0 and the renter kept paying for a pod nobody could enter.

This module judges what the renter sees, from outside the container, each cycle:

* a TCP connect to the pod's mapped SSH port (``RentedPod.ssh_port``, sent by the backend);
* the ``authorized_keys`` read the check already does (empty = the mount is missing).

State lives in Redis, one key pair per pod. A pod is judged only after this validator has seen it
healthy once (both signals good), so a template that ships no sshd, or a pod that never came up,
is never reported here. ``RENTED_POD_SSH_PROBE_CYCLES`` consecutive unhealthy cycles (default 2,
about 30 min) after that raise ``RENTED_POD_SSH_UNREACHABLE`` and one POST to the backend per
outage. The score is not changed by this module.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

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
FAULT_AUTHORIZED_KEYS_UNREADABLE = "authorized_keys_unreadable"


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
    # cycle's event names the pod. The backend is told once, on the threshold cycle only.
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


async def tcp_connect_fault(host: str, port: int, timeout: float) -> str | None:
    """None when the port accepts a TCP connection; else the fault name."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
    except TimeoutError:
        return FAULT_TCP_TIMEOUT
    except OSError:
        return FAULT_TCP_REFUSED
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return None


async def probe_rented_pod_ssh(
    ctx: Context,
    pod: RentedPod,
    ssh_pub_keys: list[str],
) -> RentedPodSshVerdict | None:
    """Judge one RUNNING rented pod from the renter's side and keep the per-pod streak.

    Returns None when the probe is off. The caller (TenantEnforcementCheck) has already confirmed
    the container is running over the executor's own SSH, so a fault here is the pod, not the host.
    """
    if not settings.RENTED_POD_SSH_PROBE_ENABLED:
        return None

    redis = ctx.services.redis
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

    boot_id_now = (ctx.state.specs or {}).get("boot_id")
    now_iso = datetime.now(UTC).isoformat()

    if not faults:
        await redis.set(_ok_key(pod.pod_id), json.dumps({"at": now_iso, "boot_id": boot_id_now}))
        await redis.delete(_fail_key(pod.pod_id))
        return RentedPodSshVerdict(
            pod_id=pod.pod_id,
            container_name=pod.container_name,
            ssh_port=pod.ssh_port,
            healthy=True,
        )

    ok_mark = _decode(await redis.get(_ok_key(pod.pod_id)))
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

    streak = _decode(await redis.get(_fail_key(pod.pod_id))) or {}
    consecutive = int(streak.get("count", 0)) + 1
    first_failed_at = streak.get("first_failed_at") or now_iso
    await redis.set(
        _fail_key(pod.pod_id),
        json.dumps({"count": consecutive, "first_failed_at": first_failed_at}),
    )

    boot_id_at_ok = ok_mark.get("boot_id")
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
    if consecutive != threshold:
        return verdict

    recorded = await _report_to_backend(ctx, verdict, boot_id_at_ok, boot_id_now)
    return RentedPodSshVerdict(
        **{**verdict.__dict__, "reported_to_backend": True, "backend_recorded": recorded}
    )


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
