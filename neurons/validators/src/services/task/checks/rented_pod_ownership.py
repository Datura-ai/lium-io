from __future__ import annotations

import asyncio
import shlex

import asyncssh

from ..messages import RentedPodOwnershipMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rented_pod_resources import running_rented_container_names

READ_TIMEOUT_SECONDS = 30


def read_command(pids: list[int], proc: str = "/proc") -> str:
    """Per container init pid: `<pid>|<first uid_map line>|<owner of />|<owner of /root>`, read on the host."""
    return (
        f"for p in {' '.join(str(int(p)) for p in pids)}; do "
        f"u=$(head -n1 {proc}/$p/uid_map 2>/dev/null); "
        f"r=$(stat -c %u {proc}/$p/root/ 2>/dev/null); h=$(stat -c %u {proc}/$p/root/root 2>/dev/null); "
        'echo "$p|$u|$r|$h"; done'
    )


def _num(text: str) -> int | None:
    # the provider controls the SSH endpoint: accept only plain ASCII digits ("²".isdigit() is True, int() fails)
    text = text.strip()
    return int(text) if text.isascii() and text.isdigit() else None


def judge(line: str) -> dict | None:
    """The pod's user namespace and the host owners of its / and /root; None when the line is unreadable.
    A runc pod (0 maps to the host's 0) has nothing outside its namespace."""
    pid, uid_map, root_uid, home_uid = (line.split("|") + ["", "", ""])[:4]
    pid_num, fields = _num(pid), [_num(f) for f in uid_map.split()]
    if pid_num is None or len(fields) != 3 or fields[0] != 0 or None in fields:
        return None
    base, length = fields[1], fields[2]
    if base == 0:
        return {"pid": pid_num, "uid_map_base": 0, "host_owners": {}, "outside_namespace": []}
    if _num(root_uid) is None:
        return None
    owners = {"/": _num(root_uid)}
    if _num(home_uid) is not None:
        owners["/root"] = _num(home_uid)
    shifted = sorted(path for path, uid in owners.items() if not base <= uid < base + length)
    return {
        "pid": pid_num,
        "uid_map_base": base,
        "host_owners": owners,
        "outside_namespace": shifted,
    }


class RentedPodOwnershipCheck:
    """Warn when a rented sysbox pod's / or /root is owned by a host uid outside the pod's own user namespace.

    Sysbox without idmapped mounts chowns a container's files into its uid range on start and back on a clean
    stop. A host that loses power skips the revert, and the container docker restarts on boot gets a new range
    its files are not in: inside it they belong to nobody, so root login (sshd StrictModes) and sudo fail while
    the pod still looks running and healthy. Read-only (one stat per path from the host, through the init pid's
    /proc root); never fails or scores. Repairing it means a recursive chown of the renter's live filesystem
    (setuid bits and file capabilities included), so it is left to the provider.
    """

    check_id = "host.observe.rented_pod_ownership"
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
        pods = {}
        for line in (res.stdout or "").splitlines():
            pod = judge(line)
            if pod and pod["pid"] in name_by_pid:
                pods[pod["pid"]] = {"container": name_by_pid[pod["pid"]], **pod}
        # a pod not read back (restarted between inspect and the read, or a garbled line) is unknown, not healthy
        unread = sorted(name for pid, name in name_by_pid.items() if pid not in pods)
        what = {"pods": list(pods.values())}
        if any(p["outside_namespace"] for p in pods.values()):
            template = Msg.SHIFTED
        elif unread:
            template = Msg.UNREAD
            what.update(reason="pod not read back", unread=unread, exit_status=res.exit_status)
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
