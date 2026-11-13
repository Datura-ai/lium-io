"""Keep one runaway rental from taking its host down (DAH-6379).

Two cgroup v2 settings, applied from the executor (root, host PID namespace) on every rental create and
again on every rented-pod resources check, so a restarted pod or a rebooted host gets them back:

- memory.oom.group=1 on the pod's container cgroup: an OOM kill of any pod process (the pod's own limit
  or a host-wide OOM) kills the whole pod at once, which docker then restarts, instead of the kernel
  killing one task at a time while the host thrashes.
- MemoryMin (memory.min) on the host services the node needs to stay reachable and manageable (sshd,
  dockerd, containerd, sysbox, the executor's own container) and on system.slice, so their pages are
  not reclaimed when a pod drives the host out of memory. `--runtime`: gone at reboot, set again here.

Best-effort: never fails the caller. Every setting it cannot apply (no cgroup v2, systemd or host view, a
denied write) is printed as a reason, logged as a WARNING with the node id and returned, so coverage is visible.
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
    "for id in $(/usr/bin/docker inspect --format '{{.Id}}' %(names)s 2>/dev/null); do f=; "
    "for d in /proc/1/root/sys/fs/cgroup/system.slice/docker-$id.scope /proc/1/root/sys/fs/cgroup/docker/$id; do "
    '[ -f "$d/memory.oom.group" ] || continue; f=1; '
    '[ "$(cat "$d/memory.oom.group")" = 1 ] || { { echo 1 > "$d/memory.oom.group"; } 2>/dev/null '
    '&& echo "oom_group_set $id" || echo "oom_group_failed $id write_denied"; }; '
    'done; [ -n "$f" ] || echo "oom_group_failed $id no_cgroup_v2_scope"; done'
)


def _host_memory_min_script() -> str:
    units = " ".join(f"{unit}={mib * MIB}" for unit, mib in UNIT_MEMORY_MIN_MIB.items())
    # sshd is ssh.service on Debian/Ubuntu and sshd.service elsewhere: count it once
    units_mib = sum(UNIT_MEMORY_MIN_MIB.values()) - UNIT_MEMORY_MIN_MIB["ssh.service"]
    ex_min = EXECUTOR_MEMORY_MIN_MIB * MIB
    inner = (
        "[ -d /run/systemd/system ] || { echo memory_min_skipped no_systemd; exit 0; }; "
        "[ -f /sys/fs/cgroup/cgroup.controllers ] || { echo memory_min_skipped no_cgroup_v2; exit 0; }; "
        "set_min() { [ \"$(systemctl show -p MemoryMin --value \"$1\")\" = \"$2\" ] && return; "
        'systemctl set-property --runtime "$1" MemoryMin="$2" >/dev/null 2>&1 && echo "memory_min_set $1" '
        '|| echo "memory_min_failed $1 set_property_failed"; }; '
        f"for kv in {units}; do u=${{kv%=*}}; systemctl is-active -q \"$u\" && set_min \"$u\" \"${{kv#*=}}\"; done; "
        # cgroupfs driver (the documented daemon.json): not a unit, so memory.min is written directly; a
        # child's protection never exceeds its parent's, so /docker gets at least the same
        'set_cg_min() { f="/sys/fs/cgroup$1/memory.min"; [ "$(cat "$f" 2>/dev/null)" -ge "$2" ] 2>/dev/null && return; '
        '{ echo "$2" > "$f"; } 2>/dev/null && echo "memory_min_set $1" || echo "memory_min_failed $1 write_denied"; }; '
        # this shell sits in the executor's cgroup; with the host cgroup namespace its path is absolute
        f"ex=$(sed -n 's#^0::##p' /proc/self/cgroup); sl={units_mib * MIB}; "
        'case "$ex" in '
        f'/system.slice/docker-*.scope) set_min "${{ex#/system.slice/}}" {ex_min}; sl=$((sl + {ex_min}));; '
        f'/docker/*) set_cg_min /docker {ex_min}; set_cg_min "$ex" {ex_min};; '
        '*) echo "memory_min_failed executor no_host_cgroup_path";; esac; '
        # a child's protection never exceeds its parent's
        'set_min system.slice "$sl"; true'
    )
    return (
        "if ! command -v nsenter >/dev/null 2>&1; then echo memory_min_skipped no_nsenter; "
        "else nsenter -t 1 -m -C -- sh -c " + shlex.quote(inner) + " 2>/dev/null || echo memory_min_skipped nsenter_failed; fi"
    )


def host_memory_guard_script(container_names: list[str]) -> str:
    names = " ".join(shlex.quote(n) for n in container_names)
    oom_group = _OOM_GROUP_SCRIPT % {"names": names} if names else "true"
    return f"{oom_group}; {_host_memory_min_script()}; true"


def parse_guard_output(stdout: str) -> dict:
    """{"set": [...], "failed": [...]}: each item is the script's line without its verb."""
    result: dict = {"set": [], "failed": []}
    for line in stdout.splitlines():
        verb, _, rest = line.strip().partition(" ")
        if verb.endswith("_set"):
            result["set"].append(f"{verb.removesuffix('_set')} {rest}")
        elif verb.endswith(("_failed", "_skipped")):
            result["failed"].append(f"{verb.rsplit('_', 1)[0]} {rest}")
    return result


async def apply_host_memory_guard(ssh, container_names: list[str], log_extra: dict | None = None) -> dict:
    """Never raises: a host this cannot reach keeps running as before. Returns parse_guard_output's dict."""
    extra = log_extra or {}
    try:
        res = await ssh.run(host_memory_guard_script(container_names), timeout=TIMEOUT_SECONDS)
        result = parse_guard_output(getattr(res, "stdout", "") or "")
    except Exception as exc:  # noqa: BLE001
        result = {"set": [], "failed": [f"ssh {exc!r}"]}
    if result["set"]:
        logger.info(_m("HOST_MEMORY_GUARD_APPLIED", extra=get_extra_info({**extra, "changed": result["set"]})))
    if result["failed"]:
        logger.warning(_m("HOST_MEMORY_GUARD_NOT_APPLIED", extra=get_extra_info({**extra, "failed": result["failed"]})))
    return result
