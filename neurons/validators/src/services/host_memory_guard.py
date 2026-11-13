"""Keep one runaway rental from taking its host down (DAH-6379).

Two cgroup v2 settings, applied from the executor (root, host PID namespace) on every rental create and
again on every rented-pod resources check, so a restarted pod or a rebooted host gets them back:

- memory.oom.group=1 on the pod's container cgroup: an OOM kill of any pod process (the pod's own limit
  or a host-wide OOM) kills the whole pod at once, which docker then restarts, instead of the kernel
  killing one task at a time while the host thrashes.
- MemoryMin (memory.min) on the host services the node needs to stay reachable and manageable (sshd,
  dockerd, containerd, sysbox, the executor's own container) and on system.slice, so their pages are
  not reclaimed when a pod drives the host out of memory. `--runtime`: gone at reboot, set again here.

Best-effort and silent on hosts without cgroup v2, systemd or a host view: nothing is changed there.
"""

from __future__ import annotations

import logging
import shlex

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 30
MIB = 1024 * 1024
# protection per host unit, MiB. Only what a unit really uses is protected, so a generous value costs
# nothing; the sum stays under the pod's minimum host RAM reserve (4 GiB).
UNIT_MEMORY_MIN_MIB = {
    "ssh.service": 128,
    "sshd.service": 128,
    "docker.service": 1024,
    "containerd.service": 512,
    "sysbox-mgr.service": 256,
    "sysbox-fs.service": 256,
}
EXECUTOR_MEMORY_MIN_MIB = 1536

_OOM_GROUP_SCRIPT = (
    "for id in $(/usr/bin/docker inspect --format '{{.Id}}' %(names)s 2>/dev/null); do "
    "for d in /proc/1/root/sys/fs/cgroup/system.slice/docker-$id.scope /proc/1/root/sys/fs/cgroup/docker/$id; do "
    '[ -f "$d/memory.oom.group" ] || continue; '
    '[ "$(cat "$d/memory.oom.group")" = 1 ] || { echo 1 > "$d/memory.oom.group" && echo "oom_group_set $id"; }; '
    "done; done"
)


def _host_memory_min_script() -> str:
    units = " ".join(f"{unit}={mib * MIB}" for unit, mib in UNIT_MEMORY_MIN_MIB.items())
    # sshd is ssh.service on Debian/Ubuntu and sshd.service elsewhere: count it once
    total_mib = sum(UNIT_MEMORY_MIN_MIB.values()) - UNIT_MEMORY_MIN_MIB["ssh.service"] + EXECUTOR_MEMORY_MIN_MIB
    inner = (
        "[ -d /run/systemd/system ] && [ -f /sys/fs/cgroup/cgroup.controllers ] || exit 0; "
        "set_min() { [ \"$(systemctl show -p MemoryMin --value \"$1\")\" = \"$2\" ] && return; "
        'systemctl set-property --runtime "$1" MemoryMin="$2" && echo "memory_min_set $1"; }; '
        f"for kv in {units}; do u=${{kv%=*}}; systemctl is-active -q \"$u\" && set_min \"$u\" \"${{kv#*=}}\"; done; "
        # this shell sits in the executor's cgroup; with the host cgroup namespace its path is absolute
        "ex=$(sed -n 's#^0::/system.slice/\\(docker-[0-9a-f]*\\.scope\\)$#\\1#p' /proc/self/cgroup); "
        f'[ -n "$ex" ] && set_min "$ex" {EXECUTOR_MEMORY_MIN_MIB * MIB}; '
        # a child's protection never exceeds its parent's
        f"set_min system.slice {total_mib * MIB}; true"
    )
    return "command -v nsenter >/dev/null 2>&1 || exit 0; nsenter -t 1 -m -C -- sh -c " + shlex.quote(inner)


def host_memory_guard_script(container_names: list[str]) -> str:
    names = " ".join(shlex.quote(n) for n in container_names)
    oom_group = _OOM_GROUP_SCRIPT % {"names": names} if names else "true"
    return f"{oom_group}; {_host_memory_min_script()}; true"


async def apply_host_memory_guard(ssh, container_names: list[str], log_extra: dict | None = None) -> None:
    """Never raises: a host this cannot reach keeps running as before."""
    try:
        res = await ssh.run(host_memory_guard_script(container_names), timeout=TIMEOUT_SECONDS)
        changed = (getattr(res, "stdout", "") or "").split()
        if changed:
            logger.info(_m("HOST_MEMORY_GUARD_APPLIED", extra=get_extra_info({**(log_extra or {}), "changed": changed})))
    except Exception as exc:  # noqa: BLE001
        logger.warning(_m("HOST_MEMORY_GUARD_FAILED", extra=get_extra_info({**(log_extra or {}), "error": repr(exc)})))
