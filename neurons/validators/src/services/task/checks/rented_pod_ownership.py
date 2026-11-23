from __future__ import annotations

import asyncio
import shlex
from datetime import datetime, timezone

import asyncssh
import redis.exceptions
from core.config import settings
from services.redis_service import RedisWrites

from ..messages import RentedPodOwnershipMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rented_pod_resources import running_rented_container_names

READ_TIMEOUT_SECONDS = 30
# docker starts a restart-policy container within a few minutes of boot; one started later was restarted alone
CONTAINER_RESTART_AFTER_BOOT_SECONDS = 600
# a creation time up to this long after the boot (whole seconds on both sides, clock slew) is read as before it: no
# container is created seconds after the kernel boots (dockerd and the validator come up later), so it is one
# created just before the stop
BOOT_TIME_UNCERTAINTY_SECONDS = 5


# a restart of these under a running pod breaks it while it looks healthy: sysbox-fs serves the pod's /proc
# (ENOTCONN after it restarts), sysbox-mgr holds its uid shift, nv-fabricmanager its NVLink multicast (CUDA 401)
RUNTIME_PROCESSES = ("sysbox-mgr", "sysbox-fs", "nv-fabricmanager")

# sudo refuses to run when any of these is not owned by the pod's root, so a shift that spares / and /root
# still breaks it; a partial shift can also spare those and leave the package dirs to nobody, so apt and dpkg
# fail
PATHS = (
    "/",
    "/root",
    "/etc/sudoers",
    "/etc/sudoers.d",
    "/etc/sudoers.d/README",
    "/usr/bin/sudo",
    "/usr",
    "/usr/lib",
    "/usr/share",
    "/var/lib/dpkg",
)


# Sysbox hands a stopping pod's files back to their host owners from the top of the tree: in 11 of 11 reads taken
# just before a host reboot (21-22 Nov) these were back at host root, the sudo files still in the range. The two
# package dirs came back in either order, so they are a set that may follow the top dirs, not a fixed sequence
REVERTED_FIRST = ("/", "/usr", "/usr/lib")
MAY_BE_REVERTED_NEXT = ("/usr/share", "/var/lib/dpkg")
# a container seen mid-revert is remembered until a clean read: any later read still showing the shape, on the same init
# pid or a new one (a pod restarted into it has finished stopping), is a lasting state, not a stop
MID_REVERT_KEY_PREFIX = "rented_pod_ownership:mid_revert"
MID_REVERT_TTL_SECONDS = 24 * 3600
# the marks' one read and one write together: Redis lets a call wait a minute for a pooled connection, and an
# observe-only check must not spend the validation deadline on it. A candidate not read back in time is SHIFTED
MID_REVERT_MARKS_BUDGET_SECONDS = 10


def read_command(pids: list[int], proc: str = "/proc") -> str:
    """Per container init pid: `<pid>|<first uid_map line>|<host owner of each of PATHS>`, read on the host."""
    return (
        f"for p in {' '.join(str(int(p)) for p in pids)}; do "
        f"u=$(head -n1 {proc}/$p/uid_map 2>/dev/null); o=; "
        f'for f in {" ".join(PATHS)}; do o="$o|$(stat -c %u {proc}/$p/root$f 2>/dev/null)"; done; '
        'echo "$p|$u$o"; done'
    )


def _num(text: str) -> int | None:
    # the provider controls the SSH endpoint: accept only plain ASCII digits ("²".isdigit() is True, int() fails)
    # of at most 10 (a uid or pid fits in 32 bits; a huge string hits int()'s digit limit and raises)
    text = text.strip()
    return int(text) if len(text) <= 10 and text.isascii() and text.isdigit() else None


def _epoch(stamp: str) -> int | None:
    """Seconds since the epoch of docker's `.Created` (UTC, `2026-11-17T19:44:26.123456789Z`); None when it is not
    one (the provider controls the SSH endpoint)."""
    try:
        return int(
            datetime.strptime(stamp.strip()[:19], "%Y-%m-%dT%H:%M:%S")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    except ValueError:
        return None


def _shift_cause(pod: dict) -> str | None:
    """"host" when the pod's container was on the host when it stopped (created before the boot) or came back right
    after the boot; "container" when it was created after the boot, or started long after it with its creation time
    unknown (created or restarted on the running host); None when neither is read."""
    before = pod.get("created_before_boot")
    if before is not None:
        return "host" if before else "container"
    after = pod.get("started_after_boot_s")
    if after is None:
        return None
    return "container" if after >= CONTAINER_RESTART_AFTER_BOOT_SECONDS else "host"


def judge(line: str) -> dict | None:
    """The pod's user namespace and the host owners of its PATHS; None when the line is unreadable.
    A runc pod (0 maps to the host's 0) has nothing outside its namespace."""
    pid, uid_map, *owner_fields = (line.split("|") + [""] * (len(PATHS) + 1))[: len(PATHS) + 2]
    pid_num, fields = _num(pid), [_num(f) for f in uid_map.split()]
    if pid_num is None or len(fields) != 3 or fields[0] != 0 or None in fields:
        return None
    base, length = fields[1], fields[2]
    if base == 0:
        return {"pid": pid_num, "uid_map_base": 0, "host_owners": {}, "outside_namespace": []}
    read = {path: _num(field) for path, field in zip(PATHS, owner_fields)}
    owners = {path: uid for path, uid in read.items() if uid is not None}
    shifted = sorted(path for path, uid in owners.items() if not base <= uid < base + length)
    # one shifted path proves the fault; without one, an unread / is unknown, not healthy. Other unread paths
    # (an image without /root or sudo) are judged from the rest
    if not shifted and read["/"] is None:
        return None
    pod = {"pid": pid_num, "uid_map_base": base, "host_owners": owners, "outside_namespace": shifted}
    # the shape a stopping pod's hand-back leaves: the top dirs (and maybe the package dirs) at host root, everything
    # else read in the range. It is only a candidate: run() reports it unread once, and SHIFTED on any later read of the
    # same container before a clean one (a hand-back cut short by a power loss stays like this on a running pod)
    if (
        set(REVERTED_FIRST) <= set(shifted) <= set(REVERTED_FIRST + MAY_BE_REVERTED_NEXT)
        and all(owners[path] == 0 for path in shifted)
        and len(owners) > len(shifted)
    ):
        pod.update(outside_namespace=[], being_reverted=shifted)
    return pod


class RentedPodOwnershipCheck:
    """Warn when a rented sysbox pod's key system files are owned by a host uid outside its user namespace.

    Sysbox without idmapped mounts chowns a container's files into its uid range on start and back on a clean
    stop. A host that loses power skips the revert, and the container docker restarts on boot gets a new range
    its files are not in: inside it they belong to nobody, so root login (sshd StrictModes) and sudo fail while
    the pod still looks running and healthy. Read-only (one stat per path from the host, through the init pid's
    /proc root); never fails or scores. Repairing it means a recursive chown of the renter's live filesystem
    (setuid bits and file capabilities included), so it is left to the provider.
    """

    check_id = "host.observe.rented_pod_ownership"
    # safe to rerun on a fresh session: a mark carries the run's pipeline id, so a rerun reads its own mark as the same sight
    replays_after_session_drop = True
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        names = running_rented_container_names(ctx)
        if not names:
            return self._event(ctx, Msg.NOT_RENTED, {})
        # /proc/<pid> of a pod is visible only when the executor shares the host PID namespace
        if (ctx.state.specs or {}).get("executor_pid_mode") != "host":
            return self._event(
                ctx, Msg.UNREAD, {"reason": "executor not in the host PID namespace"}
            )
        try:
            res = await asyncio.wait_for(
                ctx.ssh.run(
                    "/usr/bin/docker inspect --format '{{.Name}} {{.State.Pid}}' "
                    + " ".join(shlex.quote(n) for n in names),
                    check=False,
                ),
                timeout=READ_TIMEOUT_SECONDS,
            )
            pid_by_name = {}
            # docker inspect exits 1 on a missing name but still prints the others; a name not asked for is ignored
            # (the provider controls the reply, and each pod read can move a Redis mark)
            for line in (res.stdout or "").splitlines():
                name, _, pid = line.partition(" ")
                name = name.lstrip("/")
                if name in names and (_num(pid) or 0) > 0:
                    pid_by_name[name] = _num(pid)
            if not pid_by_name:
                return self._event(
                    ctx, Msg.UNREAD, {"reason": "no running rented container", "pods": names}
                )
            res = await asyncio.wait_for(
                ctx.ssh.run(read_command(sorted(pid_by_name.values())), check=False),
                timeout=READ_TIMEOUT_SECONDS,
            )
        except (asyncssh.Error, OSError, TimeoutError) as exc:
            return self._event(ctx, Msg.UNREAD, {"reason": "ssh error", "error": repr(exc)})
        name_by_pid = {pid: name for name, pid in pid_by_name.items()}
        pods, empty_reads = {}, {}
        for line in (res.stdout or "").splitlines():
            pod = judge(line)
            if pod and pod["pid"] in name_by_pid:
                pods[pod["pid"]] = {"container": name_by_pid[pod["pid"]], **pod}
            elif pod is None and _num(line.split("|")[0]) in name_by_pid:
                # which read came back empty: tells a pod with no user namespace from a /proc root not stat-able here
                fields = (line.split("|") + ["", "", ""])[1:4]
                empty_reads[name_by_pid[_num(line.split("|")[0])]] = [
                    name for name, field in zip(("uid_map", "/", "/root"), fields) if not field.strip()
                ]
        # a requested pod not read back (missing from inspect, restarted before the read, or a garbled line)
        # is unknown, not healthy
        unread = sorted(set(names) - {p["container"] for p in pods.values()})
        what = {"pods": list(pods.values())}
        if pods:
            restarted, started_at = await self._runtime_restarted_under(ctx, list(pods.values()))
            if restarted:
                what["runtime_restarted_under_pods"] = restarted
                # when each flagged runtime started: a notice must date the restart, not the day the check first saw it
                what["runtime_started_at"] = started_at
        candidates = [p for p in pods.values() if p.get("being_reverted")]
        first_sights = await self._settle_mid_revert_marks(
            ctx, candidates, [p for p in pods.values() if not p.get("being_reverted") and not p["outside_namespace"]]
        )
        for pod in candidates:
            if pod["container"] not in first_sights:
                pod["outside_namespace"] = pod.pop("being_reverted")
        shifted = [p for p in pods.values() if p["outside_namespace"]]
        reverting = sorted(p["container"] for p in pods.values() if p.get("being_reverted"))
        if shifted:
            template = await self._shift_template(ctx, shifted)
        elif unread:
            template = Msg.UNREAD
            what.update(reason="pod not read back", unread=unread, exit_status=res.exit_status)
            if empty_reads:
                what["empty_reads"] = empty_reads
        elif reverting:
            # first sight of the hand-back shape a stopping pod leaves (the host going down): unknown until the next
            # cycle shows whether the pod stopped or keeps it
            template = Msg.UNREAD
            what.update(reason="ownership being handed back, as when the pod stops", being_reverted=reverting)
        elif "runtime_restarted_under_pods" in what:
            # own outcome, not OK: a repeated OK is logged at DEBUG, so the restart would never surface
            template = Msg.RUNTIME_RESTARTED
        else:
            template = Msg.OK
        if unread:
            what["unread"] = unread
        return self._event(ctx, template, what)

    async def _settle_mid_revert_marks(self, ctx: Context, candidates: list[dict], clean: list[dict]) -> set[str]:
        """The candidates read in the hand-back shape for the first time since their last clean read (any init pid).
        One MGET and one MULTI/EXEC within MID_REVERT_MARKS_BUDGET_SECONDS: each candidate's mark is set to this run's
        pipeline id (a mark of the same run, left by the check before its session dropped and it ran again, is the same
        sight; the TTL only ends marks of containers never read again; each sight renews it) and each clean pod's is
        cleared, as a clean read ends the stop the mark stood for. When Redis cannot say in time, no candidate is a first sight: the shape
        is reported, never hidden. A dry run shares Redis with the live validator: it reads the marks, never writes."""
        if not candidates and (not clean or settings.DRY_RUN):
            return set()

        def key(pod: dict) -> str:
            return f"{MID_REVERT_KEY_PREFIX}:{ctx.executor.uuid}:{pod['container']}"

        async def settle() -> set[str]:
            seen = await ctx.services.redis.mget([key(p) for p in candidates])
            run = ctx.pipeline_id.encode()
            first = {
                p["container"]
                for p, mark in zip(candidates, seen)
                if mark is None or (mark.encode() if isinstance(mark, str) else mark) == run
            }
            if not settings.DRY_RUN:
                writes = RedisWrites()
                for pod in candidates:
                    if pod["container"] in first:
                        writes.set(key(pod), ctx.pipeline_id, ex=MID_REVERT_TTL_SECONDS)
                    else:
                        # a later sight keeps the earlier run's id: a rerun of this run must still read it as later
                        writes.expire(key(pod), MID_REVERT_TTL_SECONDS)
                for pod in clean:
                    writes.delete(key(pod))
                await ctx.services.redis.write_atomically(writes)
            return first

        try:
            return await asyncio.wait_for(settle(), timeout=MID_REVERT_MARKS_BUDGET_SECONDS)
        except (redis.exceptions.RedisError, OSError, TimeoutError):
            return set()

    async def _shift_template(self, ctx: Context, shifted: list[dict]):
        """A container restarted long after the host booted was not brought back by a host reboot: say so,
        instead of blaming an unclean host shutdown. One created before this boot was already there when the host
        stopped, however late it was started again (the validator's own stale-mount recovery starts it minutes
        after boot). Sets `started_after_boot_s` and `created_before_boot` on each pod read."""
        try:
            res = await asyncio.wait_for(
                ctx.ssh.run(
                    # pid beside its age: a pid gone since the read prints nothing, so ages cannot pair by line
                    "cut -d' ' -f1 /proc/uptime; ps -o pid=,etimes= -p "
                    + ",".join(str(int(p["pid"])) for p in shifted)
                    # btime: the kernel's boot epoch, taken in one read (not a clock and an uptime sampled apart)
                    + "; echo --; awk '/^btime /{print $2}' /proc/stat; /usr/bin/docker inspect --format '{{.Name}} {{.Created}}' "
                    + " ".join(shlex.quote(p["container"]) for p in shifted),
                    check=False,
                ),
                timeout=READ_TIMEOUT_SECONDS,
            )
        except (asyncssh.Error, OSError, TimeoutError):
            return Msg.SHIFTED
        head, _, tail = (res.stdout or "").partition("\n--\n")
        uptime, *lines = head.splitlines() or [""]
        uptime_s = _num(uptime.partition(".")[0])
        ages = {}
        for line in lines:
            pid, _, age = line.strip().partition(" ")
            ages[_num(pid)] = _num(age.strip())
        btime, *created_lines = tail.splitlines() or [""]
        boot_s = _num(btime)
        created = {}
        for line in created_lines:
            name, _, stamp = line.strip().partition(" ")
            created[name.lstrip("/")] = _epoch(stamp)
        for pod in shifted:
            age_s = ages.get(int(pod["pid"]))
            if uptime_s is not None and age_s is not None:
                pod["started_after_boot_s"] = uptime_s - age_s
            created_s = created.get(pod["container"])
            if boot_s is not None and created_s is not None:
                pod["created_before_boot"] = created_s <= boot_s + BOOT_TIME_UNCERTAINTY_SECONDS
        causes = [_shift_cause(p) for p in shifted]
        if all(c == "container" for c in causes):
            return Msg.SHIFTED_AFTER_CONTAINER_RESTART
        if None not in causes and "container" in causes:
            # some pods were created or restarted on a running host, others came back with it
            return Msg.SHIFTED_MIXED
        return Msg.SHIFTED

    async def _runtime_restarted_under(
        self, ctx: Context, pods: list[dict]
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Each runtime process that started after a pod's container: {process: [containers]}, and when each of
        those started ({process: UTC time, or "unknown"}). Both come from the kernel's start time of each process
        (/proc/<pid>/stat field 22, ticks since boot, plus the boot epoch), not from ages and a clock sampled
        apart, so a slow read cannot move a date. Observe-only, for the support sweep; an unread start flags
        nothing."""
        start = "$(sed 's/.*) //' /proc/$p/stat 2>/dev/null | cut -d' ' -f20)"  # after "pid (comm) ": field 22
        try:
            res = await asyncio.wait_for(
                ctx.ssh.run(
                    # comm is cut to 15 characters (nv-fabricmanage)
                    "ps -eo pid=,ppid=,comm= | grep -E ' ("
                    + "|".join(n[:15] for n in RUNTIME_PROCESSES)
                    + ")$' | while read -r p pp c; do echo \"$p $pp $c "
                    + start
                    + '"; done; echo --; for p in '
                    + " ".join(str(int(p["pid"])) for p in pods)
                    + '; do echo "$p '
                    + start
                    + "\"; done; echo --; awk '/^btime /{print $2}' /proc/stat; getconf CLK_TCK",
                    check=False,
                ),
                timeout=READ_TIMEOUT_SECONDS,
            )
        except (asyncssh.Error, OSError, TimeoutError):
            return {}, {}
        services, _, rest = (res.stdout or "").partition("--")
        pod_lines, _, clock = rest.partition("--")
        boot_s, ticks_per_s = ([_num(x) for x in clock.split()] + [None, None])[:2]
        procs = []
        for line in services.splitlines():
            fields = line.split()
            if len(fields) != 4:
                continue
            pid, ppid, comm, ticks = fields
            name = next((n for n in RUNTIME_PROCESSES if n[:15] == comm), None)
            if name and None not in (_num(pid), _num(ppid), _num(ticks)):
                procs.append((name, _num(pid), _num(ppid), _num(ticks)))
        pids = {(name, pid) for name, pid, _, _ in procs}
        latest: dict[str, int] = {}
        for name, _, ppid, ticks in procs:
            # sysbox-fs and sysbox-mgr re-exec themselves for short-lived helpers (e.g. while serving
            # a pod's /proc): a child of the same daemon is not a restart of it
            if (name, ppid) not in pids:
                latest[name] = max(ticks, latest.get(name, ticks))
        pod_start = {}
        for line in pod_lines.splitlines():
            pid, _, ticks = line.strip().partition(" ")
            pod_start[_num(pid)] = _num(ticks.strip())
        restarted, started_at = {}, {}
        for name, ticks in sorted(latest.items()):
            under = sorted(
                p["container"] for p in pods if pod_start.get(p["pid"]) is not None and pod_start[p["pid"]] < ticks
            )
            if under:
                restarted[name] = under
                # stated, not left out: a reader must not take a missing time for "today"
                started_at[name] = (
                    datetime.fromtimestamp(boot_s + ticks // ticks_per_s, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    if boot_s and ticks_per_s
                    else "unknown"
                )
        return restarted, started_at

    def _event(self, ctx: Context, template, what: dict) -> CheckResult:
        what = {"executor_uuid": ctx.executor.uuid, **what}
        return CheckResult(
            passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )
