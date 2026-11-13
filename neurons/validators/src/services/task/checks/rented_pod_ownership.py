from __future__ import annotations

import asyncio
import shlex

import asyncssh

from ..messages import RentedPodOwnershipMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rented_pod_resources import running_rented_container_names

READ_TIMEOUT_SECONDS = 30


# sudo refuses to run when any of these is not owned by the pod's root, so a shift that spares / and /root
# still breaks it; a partial shift can also spare those and leave the package dirs to nobody, so apt and dpkg
# fail (pod 71fc3916, 13 Nov)
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
    return {
        "pid": pid_num,
        "uid_map_base": base,
        "host_owners": owners,
        "outside_namespace": shifted,
    }


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
            # docker inspect exits 1 on a missing name but still prints the others
            for line in (res.stdout or "").splitlines():
                name, _, pid = line.partition(" ")
                if (_num(pid) or 0) > 0:
                    pid_by_name[name.lstrip("/")] = _num(pid)
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
        if any(p["outside_namespace"] for p in pods.values()):
            template = Msg.SHIFTED
        elif unread:
            template = Msg.UNREAD
            what.update(reason="pod not read back", unread=unread, exit_status=res.exit_status)
            if empty_reads:
                what["empty_reads"] = empty_reads
        else:
            template = Msg.OK
        if unread:
            what["unread"] = unread
        return self._event(ctx, template, what)

    def _event(self, ctx: Context, template, what: dict) -> CheckResult:
        what = {"executor_uuid": ctx.executor.uuid, **what}
        return CheckResult(
            passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )
