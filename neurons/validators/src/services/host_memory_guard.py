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
from collections.abc import Awaitable, Callable
from typing import Any

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
# the host's /run as the executor sees it (host PID namespace, as nsenter -t 1 needs); /run is cleared each boot
LOCK = "/proc/1/root/run/lium-memory-guard.lock"
# awk shared by the written parents (tree): rd reads a cgroup's memory.min
_AWK_RD = 'function rd(p, c,f){c=""; f="/sys/fs/cgroup" p "/memory.min"; if((getline c < f)<=0) c=0; close(f); return c} '


def _host_memory_min_script() -> str:
    units = " ".join(f"{unit}={mib * MIB}" for unit, mib in UNIT_MEMORY_MIN_MIB.items())
    unit_names = " ".join(UNIT_MEMORY_MIN_MIB)
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
        # never lower a cgroup that already has more protection ("max" included), as set_min below
        'set_cg_min() { f="/sys/fs/cgroup$1/memory.min"; c=$(cat "$f" 2>/dev/null); '
        '{ [ "$c" = max ] || [ "$c" -ge "$2" ]; } 2>/dev/null && return; '
        '{ echo "$2" > "$f"; } 2>/dev/null && echo "memory_min_set $1" || echo "memory_min_failed $1 write_denied"; }; '
        # nothing is written once a calculation fails
        'fail() { echo "memory_min_failed $1"; exit 0; }; '
        # a cgroup's children, dot-named ones included, one per line with prefix $2
        'kids() { for c in "/sys/fs/cgroup$1"/*/ "/sys/fs/cgroup$1"/.[!.]*/ "/sys/fs/cgroup$1"/..?*/; do '
        '[ -f "${c}memory.min" ] || continue; c=${c%/}; printf \'%s%s\\n\' "$2" "${c#/sys/fs/cgroup}"; done; }; '
        # cgroups written directly (not systemd units): one "value path" record per line, so a path may hold spaces.
        # A child's protection never exceeds its parent's, so bottom-up each cgroup gets the larger of its existing
        # memory.min and its own request plus its children's totals ("max" absorbs), parents written first. tree's
        # argument names a path left to systemd (/system.slice when systemd sets it); its direct children's total is
        # the record "<total> =".
        # A parent's protection is shared among all its children (the kernel divides it by claim), so each written
        # parent also covers its other children ("s path" records from sib) with their full memory.min: they keep all
        # they claim and the cgroups this guard protects keep theirs (lium-io#2385, option A). No cgroup this guard
        # does not write is changed; a child claiming "max" makes its parent "max".
        'acc=; add() { acc="$acc$2 $1\n"; }; '
        "sib() { printf '%s' \"$acc\" | while IFS= read -r r; do p=${r#* }; "
        'while [ -n "$p" ] && [ "$p" != "$1" ]; do u=${p%/*}; [ -n "$u" ] && [ "$u" != "$1" ] && kids "$u" "s "; '
        "p=$u; done; done; }; "
        'tree() { { printf \'%s\' "$acc"; sib "$1"; } | awk -v k="$1" \''
        'function sum(a,b){return (a=="max"||b=="max")?"max":a+b} '
        'function big(a,b){return (a=="max"||b=="max")?"max":(a+0>b+0?a:b)} ' + _AWK_RD +
        # byte totals always print as digits: a 32-bit mawk prints 2 GiB as 2.14748e+09 otherwise
        'BEGIN{OFMT=CONVFMT="%.0f"} '
        '{v=$1; q=substr($0,length(v)+2); if(v!="s") own[q]=sum(own[q],v); n=split(q,a,"/"); p=""; '
        'for(i=2;i<=n;i++){p=p "/" a[i]; if(p!=k){d[p]=i; if(i>m) m=i; if(v!="s") w[p]=1}}} '
        'END{for(i=m;i>=2;i--) for(p in d) if(d[p]==i){u=p; sub(/\\/[^\\/]*$/,"",u); '
        "if(!(p in w)){ss[u]=sum(ss[u],rd(p)); continue} "
        "t[p]=big(rd(p),sum(own[p],sum(t[p],(p in ss)?ss[p]:0))); "
        "if(u in d) t[u]=sum(t[u],t[p]); else if(u==k) s=sum(s,t[p]); print t[p], p} "
        'print (s==""?0:s), "="}\'; }; '
        # applies $1 to each "value path" record of $recs, parents first (record paths are read whole: one may end in
        # a space)
        "wr() { printf '%s\\n' \"$recs\" | LC_ALL=C sort -k2 | while IFS= read -r r; do v=${r%% *}; p=${r#* }; "
        'case "$v" in max|[0-9]*) [ "$p" = = ] || "$1" "$p" "$v";; esac; done; }; '
        # a parent's protection is shared among its children: when their claims still add up to more (a write was
        # denied), each gets less than it claims, so that parent is reported
        'over() { kids "$1" | { g=$(cat "/sys/fs/cgroup$1/memory.min" 2>/dev/null); n=0; while IFS= read -r c; do '
        'v=$(cat "/sys/fs/cgroup$c/memory.min" 2>/dev/null); case "$v" in max) n=max;; ""|*[!0-9]*) ;; '
        '*) [ "$n" = max ] || n=$((n + v));; esac; done; [ "$g" != max ] && { [ "$n" = max ] || [ "$n" -gt "$g" ]; } '
        '2>/dev/null && echo "memory_min_failed $1 overcommitted"; }; }; '
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
        # a unit's effective share is its larger existing MemoryMin when it has one; the slice sum and a unit holding the
        # executor (summed with the executor below) count that share. An "infinity" unit makes both unlimited too
        # (max for a directly written cgroup, infinity for system.slice): a finite parent would cap its protection.
        # Units are set (todo) after the slice total is known
        f"sl={units_mib * MIB}; inf=; hs=; todo=; ids=' '; "
        f'for kv in {units}; do u=${{kv%=*}}; d=${{kv#*=}}; e=$d; systemctl is-active -q "$u" || continue; '
        # an alias (sshd.service for ssh.service) is the same unit: counted once
        'i=$(systemctl show -p Id --value "$u"); i=${i:-$u}; case "$ids" in *" $i "*) continue;; esac; ids="$ids$i "; '
        'v=$(systemctl show -p MemoryMin --value "$u"); '
        'if [ "$v" = infinity ]; then e=max; inf=1; elif [ "$v" -gt "$d" ] 2>/dev/null; then e=$v; sl=$((sl + v - d)); fi; '
        'todo="$todo $u=$d"; '
        # the executor in the unit's own cgroup or below it (wherever Slice= put the unit) shares that cgroup's total
        'c=$(systemctl show -p ControlGroup --value "$u"); '
        '[ -n "$c" ] && case "$ex/" in "$c/"*) add "$c" "$e"; hs=$e;; esac; done; '
        # the executor's scope counts with its larger existing MemoryMin ("infinity" included); an executor directly
        # in system.slice adds its own share to the slice; one nested deeper (even under a docker scope) is written directly
        'case "$ex" in '
        f"/system.slice) sl=$((sl + {ex_min}));; "
        f'/system.slice/*/*) add "$ex" {ex_min};; '
        '/system.slice/docker-*.scope) s=${ex#/system.slice/}; v=$(systemctl show -p MemoryMin --value "$s"); '
        f'if [ "$v" = infinity ]; then inf=1; elif [ "$v" -gt {ex_min} ] 2>/dev/null; then sl=$((sl + v)); '
        f'else sl=$((sl + {ex_min})); fi; todo="$todo $s={ex_min}";; '
        f'?*) add "$ex" {ex_min};; esac; '
        # the cgroups under system.slice written here count with their full totals (an existing larger memory.min
        # included), in place of the share already counted for the unit holding the executor
        'recs=$(tree /system.slice) || fail "cgroups calculation_failed"; '
        "ts=$(printf '%s\\n' \"$recs\" | sed -n 's/ =$//p'); "
        'case "$ts" in max) ;; ""|*[!0-9]*) fail "cgroups calculation_failed";; esac; '
        'if [ "$ts" = max ] || [ "$hs" = max ]; then inf=1; else sl=$((sl - ${hs:-0} + ts)); fi; '
        # the slice's other children (services this guard does not set, rentals' scopes) count with their full
        # memory.min, as for a written parent's children; one claiming "max" makes the slice infinity
        "ss=0; "
        "for c in /sys/fs/cgroup/system.slice/*/ /sys/fs/cgroup/system.slice/.[!.]*/ /sys/fs/cgroup/system.slice/..?*/; do "
        '[ -f "${c}memory.min" ] || continue; n=${c%/}; n=${n##*/}; '
        f'case " {unit_names} " in *" $n "*) continue;; esac; case "$ex/" in "/system.slice/$n/"*) continue;; esac; '
        'v=$(cat "${c}memory.min" 2>/dev/null); '
        'if [ "$v" = max ]; then ss=max; elif [ "$ss" != max ] && [ "$v" -gt 0 ] 2>/dev/null; then ss=$((ss + v)); fi; done; '
        'if [ "$ss" = max ]; then inf=1; else sl=$((sl + ss)); fi; '
        'for kv in $todo; do set_min "${kv%=*}" "${kv#*=}"; done; wr set_cg_min; '
        '[ -n "$inf" ] && sl=infinity; set_min system.slice "$sl"; wr over; over /system.slice; '
        "else "
        # no systemd (another init, a container as PID 1): the services' own cgroups, found from /proc. A host service
        # shares PID 1's PID namespace (its mount namespace may be its own); a rental's processes have their own PID
        # namespace, and the executor's (host PID namespace) sit in its cgroup, so both are left out
        "hp=$(readlink /proc/1/ns/pid); "
        f'seen=; for d in /proc/[0-9]*; do n=$(cat "$d/comm" 2>/dev/null); v=; for kv in {procs}; do '
        '[ "${kv%=*}" = "$n" ] && v=${kv#*=}; done; [ -n "$v" ] || continue; '
        'pn=$(readlink "$d/ns/pid"); case " $seen " in *" $n "*) continue;; esac; '
        '[ -n "$pn" ] && [ -n "$hp" ] || { echo "memory_min_failed $n pid_namespace_unreadable"; continue; }; '
        '[ "$pn" = "$hp" ] || continue; p=$(sed -n \'s#^0::##p\' "$d/cgroup" 2>/dev/null); '
        '[ -n "$ex" ] && case "$p/" in "$ex/"*) continue;; esac; seen="$seen $n"; '
        # memory.min does not exist on the root cgroup: a service there cannot be protected without moving it
        'case "$p" in /|"") echo "memory_min_failed $n root_cgroup";; /..*) ;; *) add "$p" "$v";; esac; done; '
        f'[ -n "$ex" ] && add "$ex" {ex_min}; '
        'recs=$(tree) || fail "cgroups calculation_failed"; wr set_cg_min; wr over; '
        "fi; true"
    )
    # a rental create and a resources check can run the guard at once: one run at a time, so two never write the same
    # cgroups from stale reads. The lock is taken here, in the executor (whose image
    # ships flock with nsenter, both util-linux), on a file in the host's /run, so the host needs no flock of its own
    return (
        "if ! command -v nsenter >/dev/null 2>&1; then echo memory_min_skipped no_nsenter; "
        "elif ! command -v flock >/dev/null 2>&1; then echo memory_min_skipped no_flock; "
        f"elif ! (: >>{LOCK}) 2>/dev/null; then echo memory_min_skipped lock_unavailable; "
        # the host part inherits descriptor 9, so the lock stays held until it exits even if this shell dies first
        "else { if flock -w 25 9; then nsenter -t 1 -m -C -- sh -c "
        + shlex.quote(inner)
        + " 2>/dev/null || echo memory_min_skipped nsenter_failed; else echo memory_min_skipped lock_busy; fi; "
        f"}} 9>>{LOCK}; fi"
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


async def apply_host_memory_guard(
    ssh,
    log_extra: dict | None = None,
    run: Callable[[str], Awaitable[Any]] | None = None,
    reraise: tuple[type[BaseException], ...] = (),
) -> dict:
    """Never raises, except an error in `reraise`: a host this cannot reach keeps running as before. `run` replaces
    ssh.run(script, timeout=TIMEOUT_SECONDS), e.g. with a run that frees the channel on a timeout. Returns
    parse_guard_output's dict."""
    extra = log_extra or {}
    try:
        script = host_memory_guard_script()
        res = await (run(script) if run is not None else ssh.run(script, timeout=TIMEOUT_SECONDS))
        result = parse_guard_output(getattr(res, "stdout", "") or "")
    except reraise:
        raise
    except Exception as exc:  # noqa: BLE001
        result = {"set": [], "failed": [f"ssh {exc!r}"]}
    if result["set"]:
        logger.info(
            _m(
                "HOST_MEMORY_GUARD_APPLIED",
                extra=get_extra_info({**extra, "changed": result["set"]}),
            )
        )
    if result["failed"]:
        logger.warning(
            _m(
                "HOST_MEMORY_GUARD_NOT_APPLIED",
                extra=get_extra_info({**extra, "failed": result["failed"]}),
            )
        )
    return result
