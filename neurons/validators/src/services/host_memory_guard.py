"""Keep one runaway rental from taking its host down (DAH-6379).

Applied from the executor (root, host PID namespace) on every rental create and again on every rented-pod
resources check, so a rebooted host gets it back: MemoryMin (memory.min) on the host services the node needs
to stay reachable and manageable (sshd, dockerd, containerd, sysbox, the executor's own container) and on
system.slice, so their pages are not reclaimed when a pod drives the host out of memory. `--runtime`: gone at
reboot, set again here.

Without systemd (another init, or a container as PID 1) the same services are found by process name and their
cgroups written directly. Best-effort: never fails the caller. Every setting it cannot apply (no cgroup v2 or host
view, a service or the executor in the root cgroup, a denied write) is printed as a reason, logged as a WARNING with the node id and returned, so coverage is visible.
"""

from __future__ import annotations

import logging
import shlex

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 30
MIB = 1024 * 1024
# protection per host unit, MiB. Only what a unit really uses is protected, so a generous value costs
# nothing; the sum stays under the pod's minimum host RAM reserve (8 GiB).
UNIT_MEMORY_MIN_MIB = {
    "ssh.service": 128,
    "sshd.service": 128,
    "docker.service": 1024,
    "containerd.service": 512,
    "sysbox-mgr.service": 256,
    "sysbox-fs.service": 256,
}
EXECUTOR_MEMORY_MIN_MIB = 1536
SERVICE_PROCESS = {"docker.service": "dockerd"}

def _host_memory_min_script() -> str:
    units = " ".join(f"{unit}={mib * MIB}" for unit, mib in UNIT_MEMORY_MIN_MIB.items())
    # sshd is ssh.service on Debian/Ubuntu and sshd.service elsewhere: count it once
    units_mib = sum(UNIT_MEMORY_MIN_MIB.values()) - UNIT_MEMORY_MIN_MIB["ssh.service"]
    ex_min = EXECUTOR_MEMORY_MIN_MIB * MIB
    # without systemd the same services are found by process name (sshd for both ssh units)
    procs = " ".join(
        f"{SERVICE_PROCESS.get(unit, unit.removesuffix('.service'))}={mib * MIB}"
        for unit, mib in UNIT_MEMORY_MIN_MIB.items()
        if unit != "ssh.service"
    )
    inner = (
        "[ -f /sys/fs/cgroup/cgroup.controllers ] || { echo memory_min_skipped no_cgroup_v2; exit 0; }; "
        # never lower a cgroup that already has more protection, as set_min below
        'set_cg_min() { f="/sys/fs/cgroup$1/memory.min"; [ "$(cat "$f" 2>/dev/null)" -ge "$2" ] 2>/dev/null && return; '
        '{ echo "$2" > "$f"; } 2>/dev/null && echo "memory_min_set $1" || echo "memory_min_failed $1 write_denied"; }; '
        # cgroups written directly (not systemd units): a child's protection never exceeds its parent's, so every
        # ancestor gets the sum of what is protected below it, parents first; /system.slice stays systemd's
        'acc=; add() { acc="$acc $1=$2"; }; '
        "flush() { for kv in $acc; do echo \"${kv%=*} ${kv#*=}\"; done | awk '{n=split($1,a,\"/\"); p=\"\"; "
        "for(i=2;i<=n;i++){p=p \"/\" a[i]; if(p!=\"/system.slice\") s[p]+=$2}} END{for(p in s) print p, s[p]}' "
        '| sort | while read -r p v; do set_cg_min "$p" "$v"; done; }; '
        # this shell sits in the executor's cgroup; with the host cgroup namespace its path is absolute
        "ex=$(sed -n 's#^0::##p' /proc/self/cgroup); "
        'case "$ex" in /|"") ex=; echo "memory_min_failed executor root_cgroup";; '
        '/..*) ex=; echo "memory_min_failed executor outside_host_cgroup_namespace";; esac; '
        "if [ -d /run/systemd/system ]; then "
        # never lower a unit that already has more protection (infinity included)
        'set_min() { v=$(systemctl show -p MemoryMin --value "$1"); '
        '{ [ "$v" = infinity ] || [ "$v" -ge "$2" ]; } 2>/dev/null && return; '
        'systemctl set-property --runtime "$1" MemoryMin="$2" >/dev/null 2>&1 && echo "memory_min_set $1" '
        '|| echo "memory_min_failed $1 set_property_failed"; }; '
        f"for kv in {units}; do u=${{kv%=*}}; systemctl is-active -q \"$u\" && set_min \"$u\" \"${{kv#*=}}\"; done; "
        f"sl={units_mib * MIB}; "
        'case "$ex" in '
        f'/system.slice/docker-*.scope) set_min "${{ex#/system.slice/}}" {ex_min}; sl=$((sl + {ex_min}));; '
        f'/system.slice/*) add "$ex" {ex_min}; sl=$((sl + {ex_min}));; '
        f'?*) add "$ex" {ex_min};; esac; '
        'flush; set_min system.slice "$sl"; '
        "else "
        # no systemd (another init, a container as PID 1): the services' own cgroups, found from /proc
        f"seen=; for d in /proc/[0-9]*; do n=$(cat \"$d/comm\" 2>/dev/null); v=; for kv in {procs}; do "
        '[ "${kv%=*}" = "$n" ] && v=${kv#*=}; done; [ -n "$v" ] || continue; case " $seen " in *" $n "*) continue;; esac; '
        'seen="$seen $n"; p=$(sed -n \'s#^0::##p\' "$d/cgroup" 2>/dev/null); '
        # memory.min does not exist on the root cgroup: a service there cannot be protected without moving it
        'case "$p" in /|"") echo "memory_min_failed $n root_cgroup";; /..*) ;; *) add "$p" "$v";; esac; done; '
        f'[ -n "$ex" ] && add "$ex" {ex_min}; flush; '
        "fi; true"
    )
    return (
        "if ! command -v nsenter >/dev/null 2>&1; then echo memory_min_skipped no_nsenter; "
        "else nsenter -t 1 -m -C -- sh -c " + shlex.quote(inner) + " 2>/dev/null || echo memory_min_skipped nsenter_failed; fi"
    )


def host_memory_guard_script() -> str:
    return f"{_host_memory_min_script()}; true"


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


async def apply_host_memory_guard(ssh, log_extra: dict | None = None) -> dict:
    """Never raises: a host this cannot reach keeps running as before. Returns parse_guard_output's dict."""
    extra = log_extra or {}
    try:
        res = await ssh.run(host_memory_guard_script(), timeout=TIMEOUT_SECONDS)
        result = parse_guard_output(getattr(res, "stdout", "") or "")
    except Exception as exc:  # noqa: BLE001
        result = {"set": [], "failed": [f"ssh {exc!r}"]}
    if result["set"]:
        logger.info(_m("HOST_MEMORY_GUARD_APPLIED", extra=get_extra_info({**extra, "changed": result["set"]})))
    if result["failed"]:
        logger.warning(_m("HOST_MEMORY_GUARD_NOT_APPLIED", extra=get_extra_info({**extra, "failed": result["failed"]})))
    return result
