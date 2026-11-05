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


def judge(line: str) -> dict | None:
    """The pod's user namespace and the host owners of its / and /root; None when unreadable or not
    in a user namespace (a runc pod maps 0 to the host's 0, so nothing can be outside it)."""
    pid, uid_map, root_uid, home_uid = (line.split("|") + ["", "", ""])[:4]
    fields = uid_map.split()
    if len(fields) != 3 or fields[0] != "0" or not all(f.isdigit() for f in fields):
        return None
    base, length = int(fields[1]), int(fields[2])
    if base == 0 or not root_uid.isdigit():
        return None
    owners = {"/": int(root_uid)}
    if home_uid.isdigit():
        owners["/root"] = int(home_uid)
    shifted = sorted(path for path, uid in owners.items() if not base <= uid < base + length)
    return {
        "pid": int(pid),
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
                if pid.strip().isdigit() and int(pid) > 0:
                    pid_by_name[name.lstrip("/")] = int(pid)
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
        pods = []
        for line in (res.stdout or "").splitlines():
            pod = judge(line)
            if pod and pod["pid"] in name_by_pid:
                pods.append({"container": name_by_pid[pod["pid"]], **pod})
        template = Msg.SHIFTED if any(p["outside_namespace"] for p in pods) else Msg.OK
        return self._event(ctx, template, {"pods": pods})

    def _event(self, ctx: Context, template, what: dict) -> CheckResult:
        what = {"executor_uuid": ctx.executor.uuid, **what}
        return CheckResult(
            passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )
