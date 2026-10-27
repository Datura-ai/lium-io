"""Restart the executor container when a stale containerd task keeps it down (DAH-5101).

If the executor dies while its containerd shim is still alive (a host IO stall can leave the shim blocked
for minutes), every restart fails with `failed to create task for container: AlreadyExists: task <id>:
already exists`. Docker's restart policy then stops retrying and autoheal only restarts running containers,
so the node stays offline until someone kills the shim by hand. The monitor container (pid: host, docker
socket) runs this check every minute: it kills only the shim that carries this container's own id, and its
children, then starts the container. A tree with a process in uninterruptible IO (D state) is left for the
next pass, and the container starts only once every killed process is gone: a survivor would run beside
the new executor on the same rootfs and volumes.
"""

import os
import re
import signal
import time

EXECUTOR_SERVICE_LABEL = "com.docker.compose.service=executor"
_ALREADY_EXISTS = re.compile(r"AlreadyExists: task ([0-9a-f]{64})")


def stale_task_id(container_id: str, status: str, error: str) -> str | None:
    """The container's own id when it is down because its old task still exists, else None."""
    match = _ALREADY_EXISTS.search(error or "")
    if status not in ("exited", "created") or not match or match.group(1) != container_id:
        return None
    return container_id


def _cmdline(proc_root: str, pid: str) -> list[str]:
    try:
        with open(os.path.join(proc_root, pid, "cmdline"), "rb") as file:
            return [arg.decode(errors="replace") for arg in file.read().split(b"\0") if arg]
    except OSError:
        return []


def _stat_field(proc_root: str, pid: str, index: int) -> str | None:
    try:
        with open(os.path.join(proc_root, pid, "stat")) as file:
            # the comm field may hold spaces; state and ppid are the 1st and 2nd fields after its ")"
            return file.read().rpartition(")")[2].split()[index]
    except (OSError, IndexError):
        return None


def _cgroup(proc_root: str, pid: str) -> str:
    try:
        with open(os.path.join(proc_root, pid, "cgroup")) as file:
            return file.read()
    except OSError:
        return ""


def _parent_pid(proc_root: str, pid: str) -> str | None:
    return _stat_field(proc_root, pid, 1)


def _state(proc_root: str, pid: int) -> str | None:
    """The process state letter, None once the pid is gone."""
    return _stat_field(proc_root, str(pid), 0)


def shim_pids(task_id: str, proc_root: str = "/proc") -> list[int]:
    """The containerd shim holding `task_id`, the task's own processes and all their descendants."""
    pids = [pid for pid in os.listdir(proc_root) if pid.isdigit()]
    shims = []
    for pid in pids:
        argv = _cmdline(proc_root, pid)
        if (
            argv
            and os.path.basename(argv[0]).startswith("containerd-shim")
            and task_id in _id_args(argv)
        ) or task_id in _cgroup(proc_root, pid):
            # the cgroup still names a process that outlived its shim and was reparented to init
            shims.append(pid)
    found = list(shims)
    parent_of = {pid: _parent_pid(proc_root, pid) for pid in pids}
    for parent in found:  # grows while iterating: breadth-first over the shims' descendants
        found.extend(pid for pid, ppid in parent_of.items() if ppid == parent and pid not in found)
    return [int(pid) for pid in found]


def _id_args(argv: list[str]) -> set[str]:
    return {argv[index + 1] for index, arg in enumerate(argv[:-1]) if arg == "-id"}


def _stale_task_id(container) -> str | None:
    state = container.attrs.get("State", {})
    return stale_task_id(container.id, state.get("Status", ""), state.get("Error", ""))


def recover_stale_executor_task(
    client, proc_root: str = "/proc", kill=os.kill, sleep=time.sleep, wait_seconds: int = 30
) -> str | None:
    """One pass: start an executor container held down by its own stale task. Returns what was done."""
    for container in client.containers.list(all=True, filters={"label": EXECUTOR_SERVICE_LABEL}):
        task_id = _stale_task_id(container)
        if task_id is None:
            continue
        pids = shim_pids(task_id, proc_root)
        if any(_state(proc_root, pid) == "D" for pid in pids):
            return f"stale containerd task {task_id[:12]} is in uninterruptible IO, retrying next pass"
        # the /proc scan takes a while: a restart that won meanwhile must not lose its fresh shim
        container.reload()
        if _stale_task_id(container) != task_id:
            continue
        for pid in pids:
            try:
                kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for _ in range(wait_seconds):
            if all(_state(proc_root, pid) in (None, "Z") for pid in pids):
                break
            sleep(1)
        else:
            return f"stale containerd task {task_id[:12]} survived SIGKILL, {container.name} not started"
        container.start()
        return f"killed stale containerd task {task_id[:12]} and started {container.name}"
    return None
