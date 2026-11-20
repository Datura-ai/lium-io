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
        # cgroups written directly (not systemd units): one "value path" record per line, so a path may hold spaces.
        # A child's protection never exceeds its parent's, so bottom-up each cgroup gets the larger of its existing
        # memory.min and its own request plus its children's totals ("max" absorbs), parents written first. tree's
        # argument names a path left to systemd (/system.slice when systemd sets it); its direct children's total is
        # the record "<total> =", which flush skips.
        # A parent's protection is shared among all its children, so each written parent also counts its other
        # children ("s path" records from sib) with the protection they already hold: their memory.min summed, never
        # more in total than the parent granted before this guard first raised it (a sibling may be a rental; its own
        # claim must not raise the host's reserve). That baseline is kept per boot in $B ("value path" lines), so the
        # guard's own earlier increment is never counted as a sibling's grant on a later run
        'acc=; add() { acc="$acc$2 $1\n"; }; '
        "sib() { printf '%s' \"$acc\" | while IFS= read -r r; do p=${r#* }; "
        'while u=${p%/*}; [ -n "$u" ] && [ "$u" != "$1" ]; do for c in "/sys/fs/cgroup$u"/*/; do '
        '[ -f "${c}memory.min" ] || continue; c=${c%/}; printf \'s %s\\n\' "${c#/sys/fs/cgroup}"; done; p=$u; done; done; }; '
        "tree() { { printf '%s' \"$acc\"; sib \"$1\"; } | awk -v k=\"$1\" -v B=\"$B\" '"
        'function sum(a,b){return (a=="max"||b=="max")?"max":a+b} '
        'function big(a,b){return (a=="max"||b=="max")?"max":(a+0>b+0?a:b)} '
        'function small(a,b){return a=="max"?b:(b=="max"?a:(a+0<b+0?a:b))} '
        'function rd(p, c,f){c=""; f="/sys/fs/cgroup" p "/memory.min"; if((getline c < f)<=0) c=0; close(f); return c} '
        'function base(p){if(!(p in bs)){bs[p]=rd(p); nb[p]=1} return bs[p]} '
        'BEGIN{while((getline l < B)>0){v=l; sub(/ .*/,"",v); q=substr(l,length(v)+2); '
        'if(q!="" && !(q in bs) && v ~ /^([0-9]+|max)$/) bs[q]=v} close(B)} '
        '{v=$1; q=substr($0,length(v)+2); if(v!="s") own[q]=sum(own[q],v); n=split(q,a,"/"); p=""; '
        "for(i=2;i<=n;i++){p=p \"/\" a[i]; if(p!=k){d[p]=i; if(i>m) m=i; if(v!=\"s\") w[p]=1}}} "
        'END{for(i=m;i>=2;i--) for(p in d) if(d[p]==i){u=p; sub(/\\/[^\\/]*$/,"",u); '
        'if(!(p in w)){ss[u]=sum(ss[u],rd(p)); continue} '
        # every written parent's baseline is taken before its first write, siblings or not, so one that appears
        # later is never credited with the guard's own increment
        "b=base(p); t[p]=big(rd(p),sum(own[p],sum(t[p],(p in ss)?small(ss[p],b):0))); "
        'if(u in d) t[u]=sum(t[u],t[p]); else if(u==k) s=sum(s,t[p]); print t[p], p} '
        'for(p in nb) print bs[p], p >> B; print (s==""?0:s), "="}\'; }; '
        'flush() { tree "$1" | LC_ALL=C sort -k2 | while read -r v p; do [ "$p" = = ] || set_cg_min "$p" "$v"; done; }; '
        # this shell sits in the executor's cgroup; with the host cgroup namespace its path is absolute
        "ex=$(sed -n 's#^0::##p' /proc/self/cgroup); "
        "B=/run/lium-host-memory-guard.$(cat /proc/sys/kernel/random/boot_id 2>/dev/null); "
        # a rental create and a resources check can run the guard at once: one run at a time, so a baseline is
        # recorded once and always before this guard first raises that cgroup
        'command -v flock >/dev/null 2>&1 || { echo memory_min_skipped no_flock; exit 0; }; '
        'exec 9>>"$B.lock"; flock -w 25 9 || { echo memory_min_skipped lock_busy; exit 0; }; '
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
        # (max for a directly written cgroup, infinity for system.slice): a finite parent would cap its protection
        f"sl={units_mib * MIB}; inf=; hs=; "
        f"for kv in {units}; do u=${{kv%=*}}; d=${{kv#*=}}; e=$d; "
        'if systemctl is-active -q "$u"; then v=$(systemctl show -p MemoryMin --value "$u"); '
        'if [ "$v" = infinity ]; then e=max; inf=1; elif [ "$v" -gt "$d" ] 2>/dev/null; then e=$v; sl=$((sl + v - d)); fi; '
        'set_min "$u" "$d"; fi; '
        'case "$ex" in "/system.slice/$u/"*) add "/system.slice/$u" "$e"; hs=$e;; esac; done; '
        # the executor's scope counts with its larger existing MemoryMin ("infinity" included); an executor directly
        # in system.slice adds its own share to the slice; one nested deeper (even under a docker scope) is written directly
        'case "$ex" in '
        f'/system.slice) sl=$((sl + {ex_min}));; '
        f'/system.slice/*/*) add "$ex" {ex_min};; '
        '/system.slice/docker-*.scope) s=${ex#/system.slice/}; v=$(systemctl show -p MemoryMin --value "$s"); '
        f'if [ "$v" = infinity ]; then inf=1; elif [ "$v" -gt {ex_min} ] 2>/dev/null; then sl=$((sl + v)); '
        f'else sl=$((sl + {ex_min})); fi; set_min "$s" {ex_min};; '
        f'?*) add "$ex" {ex_min};; esac; '
        # the cgroups under system.slice written here count with their full totals (an existing larger memory.min
        # included), in place of the share already counted for the unit holding the executor
        'ts=$(tree /system.slice | sed -n "s/ =\$//p"); '
        'if [ "$ts" = max ] || [ "$hs" = max ]; then inf=1; else sl=$((sl - ${hs:-0} + ${ts:-0})); fi; '
        # the slice's other children (services this guard does not set, rentals' scopes) keep the protection they
        # already hold: their memory.min summed, never more in total than the slice granted before this guard first
        # raised it (the per-boot baseline in $B), as for a written parent's children
        "bs=$(awk '$2==\"/system.slice\" && $1 ~ /^([0-9]+|max)$/ {print $1; exit}' \"$B\" 2>/dev/null); "
        'if [ -z "$bs" ]; then bs=$(systemctl show -p MemoryMin --value system.slice); [ "$bs" = infinity ] && bs=max; '
        'bs=${bs:-0}; echo "$bs /system.slice" >> "$B"; fi; ss=0; '
        'for c in /sys/fs/cgroup/system.slice/*/; do [ -f "${c}memory.min" ] || continue; n=${c%/}; n=${n##*/}; '
        f'case " {unit_names} " in *" $n "*) continue;; esac; case "$ex/" in "/system.slice/$n/"*) continue;; esac; '
        'v=$(cat "${c}memory.min" 2>/dev/null); '
        'if [ "$v" = max ]; then ss=max; elif [ "$ss" != max ] && [ "$v" -gt 0 ] 2>/dev/null; then ss=$((ss + v)); fi; done; '
        '[ "$ss" = max ] && ss=$bs; [ "$bs" != max ] && [ "$ss" -gt "$bs" ] 2>/dev/null && ss=$bs; '
        'if [ "$ss" = max ]; then inf=1; else sl=$((sl + ss)); fi; '
        'flush /system.slice; [ -n "$inf" ] && sl=infinity; set_min system.slice "$sl"; '
        "else "
        # no systemd (another init, a container as PID 1): the services' own cgroups, found from /proc. A host service
        # shares PID 1's PID namespace (its mount namespace may be its own); a rental's processes have their own PID
        # namespace, and the executor's (host PID namespace) sit in its cgroup, so both are left out
        "hp=$(readlink /proc/1/ns/pid); "
        f"seen=; for d in /proc/[0-9]*; do n=$(cat \"$d/comm\" 2>/dev/null); v=; for kv in {procs}; do "
        '[ "${kv%=*}" = "$n" ] && v=${kv#*=}; done; [ -n "$v" ] || continue; '
        'pn=$(readlink "$d/ns/pid"); case " $seen " in *" $n "*) continue;; esac; '
        '[ -n "$pn" ] && [ -n "$hp" ] || { echo "memory_min_failed $n pid_namespace_unreadable"; continue; }; '
        '[ "$pn" = "$hp" ] || continue; p=$(sed -n \'s#^0::##p\' "$d/cgroup" 2>/dev/null); '
        '[ -n "$ex" ] && case "$p/" in "$ex/"*) continue;; esac; seen="$seen $n"; '
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
