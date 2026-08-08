"""Start, recognise and stop the process that holds a CVM.

The supervisor is `cvmd.dstack.child` in its own session. Everything here is about the two
questions cvmd has to answer honestly about it: *is it still ours*, and *is it really gone*.

Measured on au11 (2026-08-08): `run/vms/my-executor/runtime.json` still named pid 56920 hours
after that process died, because `lium-cvm.sh stop` gave up waiting and left the file behind.
So a recorded pid is a claim, never evidence. `is_supervisor` reads `/proc/<pid>/cmdline` and
requires both the child module name and this CVM's directory, which also rules out a recycled
pid — a real risk on a host that has been up for weeks.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

CHILD_MODULE = "cvmd.dstack.child"
CONSOLE_LOG = "console.log"

# How long a signalled process group gets before the next signal. A TDX guest returns its
# memory to the host as it exits and that is not instant, but this is the *post-signal* wait —
# the graceful window is the caller's `timeout`.
TERMINATION_GRACE_SECONDS = 10

# What a running CVM's QEMU always carries on its command line. dstack builds every TDX guest
# with `-object tdx-guest,id=tdx`, so this identifies one whether cvmd or `lium-cvm.sh` started
# it — which is the only way "one CVM per node" can hold while both launch paths exist.
TDX_GUEST_MARKER = "tdx-guest"

# `run_instance` writes this when the VM is about to start and removes it when QEMU exits, so
# its presence is dstack's own liveness marker. cvmd reads it, but never trusts it alone.
RUNTIME_FILE = "runtime.json"

# The file dstack creates for the guest's disk. Its existence is the fail-closed "a CVM lives
# in this directory" predicate: a stopped CVM still owns the node's GPUs and ports until its
# directory is removed, so a launch must not step over one just because no process is running.
DISK_IMAGE = "hda.img"


class SupervisorError(Exception):
    """The supervisor could not be started."""


def _cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return [part for part in raw.decode(errors="replace").split("\0") if part]


def is_supervisor(pid: int, vm_dir: Path) -> bool:
    """Is `pid` alive *and* the supervisor for this VM directory?

    Both halves matter. Liveness alone accepts a recycled pid; identity alone accepts a record
    of something long dead.
    """
    argv = _cmdline(pid)
    if argv is None:
        return False
    joined = " ".join(argv)
    return CHILD_MODULE in joined and str(vm_dir) in joined


def running_cvms() -> list[tuple[int, str]]:
    """Every TDX guest running on this host right now, as (pid, argv0-ish description).

    This is what makes "one CVM per node" true rather than "one cvmd-managed CVM per node".
    GPU passthrough is exclusive and so is the host's TDX capacity, so a CVM started by
    `lium-cvm.sh` blocks a cvmd launch exactly as much as one cvmd started — and during the
    CVM v2 rollout both launch paths exist on the same hosts. Reading `/proc` finds both;
    cvmd's own records find only its own.
    """
    found: list[tuple[int, str]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found

    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        argv = _cmdline(int(entry.name))
        if not argv:
            continue
        # The marker has to appear in an argument, not merely somewhere in the joined string,
        # so a path or a log filename containing it cannot be mistaken for a running guest.
        if any(TDX_GUEST_MARKER in arg for arg in argv) and "qemu" in argv[0].lower():
            found.append((int(entry.name), argv[0]))
    return sorted(found)


def console_log_path(vm_dir: Path) -> Path:
    return vm_dir / CONSOLE_LOG


def vm_directories(run_dir: Path) -> list[Path]:
    """Every directory under `run_dir` that holds a CVM's disk.

    Keyed on the disk image rather than on the directory existing, so a half-written directory
    from a launch that failed before `qemu-img create` does not look like a live CVM.
    """
    if not run_dir.is_dir():
        return []
    return sorted(child for child in run_dir.iterdir() if (child / DISK_IMAGE).exists())


def runtime_pid(vm_dir: Path) -> int | None:
    """The pid dstack recorded for this VM, if it recorded one. A claim, not evidence."""
    from cvmd.atomic import read_json

    raw = read_json(vm_dir / RUNTIME_FILE)
    if not isinstance(raw, dict):
        return None
    try:
        return int(raw["pid"])
    except (KeyError, TypeError, ValueError):
        return None


def spawn(*, scripts_dir: Path, vm_dir: Path, kp_port: int) -> int:
    """Start the supervisor detached from cvmd and return its pid.

    `start_new_session=True` puts it in its own session and process group, so it survives a
    cvmd restart and never receives a signal aimed at the daemon. Both streams go to
    `console.log`: QEMU runs `-nographic` with its serial line on stdio, so that file is the
    guest's boot console, and `run_instance` prints the full QEMU command line into it first.
    A pipe would be worse than useless here — nothing would drain it, and the guest would block
    on a full buffer partway through booting.
    """
    log_path = console_log_path(vm_dir)
    try:
        log_handle = log_path.open("ab", buffering=0)
    except OSError as exc:
        raise SupervisorError(f"cannot open the console log at {log_path}: {exc}") from exc

    try:
        with open(os.devnull, "rb") as devnull:
            process = subprocess.Popen(  # noqa: S603 - argv is built here, never from a request
                [sys.executable, "-m", CHILD_MODULE, str(scripts_dir), str(vm_dir), str(kp_port)],
                stdin=devnull,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except OSError as exc:
        raise SupervisorError(f"cannot start the supervisor: {exc}") from exc
    finally:
        # cvmd holds no handle on the console log; the child owns it now. Keeping one open would
        # pin the file after a teardown removed it.
        log_handle.close()

    logger.info("supervisor for %s started as pid %d", vm_dir, process.pid)
    return process.pid


def _process_group_gone(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _wait_for_group(pid: int, seconds: int) -> bool:
    for _ in range(max(seconds, 1)):
        if _process_group_gone(pid):
            return True
        time.sleep(1)
    return _process_group_gone(pid)


def shutdown(dstack: ModuleType, vm_dir: Path, pid: int, *, timeout: int) -> str:
    """Stop the CVM and return what it took. Raises nothing — it reports.

    Three escalating steps, and the middle one is the reason this is not simply a call to
    `shutdown_instance`:

    1. `shutdown_instance` asks the guest to power off over vsock. This is dstack's own path
       and the only graceful one.
    2. If the process group is still there, SIGTERM the **group**. `shutdown_instance`'s own
       `--force` kills the pid in `runtime.json`, which is the *supervisor's* pid — QEMU is its
       child, so killing that pid alone orphans a running VM still holding the guest's RAM and
       the GPUs. Signalling the group reaches both, and the group exists precisely because
       `spawn` started a new session.
    3. SIGKILL the group.

    Confirming the process group is gone is the floor, not the ceiling. DAH-2577 adds the real
    predicate — VFIO file descriptors closed, guest RAM returned, forwarded ports bindable —
    because a reaped QEMU is necessary for those and nowhere near sufficient.
    """
    dstack.shutdown_instance(str(vm_dir), timeout=timeout, force=False)
    if _wait_for_group(pid, timeout):
        return "guest powered off on request"

    logger.warning(
        "CVM at %s did not power off in %ds; signalling process group %d", vm_dir, timeout, pid
    )
    return shutdown_by_signal(pid, timeout=timeout)


def shutdown_by_signal(pid: int, *, timeout: int) -> str:
    """SIGTERM then SIGKILL the supervisor's whole process group.

    The **group**, not the pid. `shutdown_instance`'s own `--force` kills the pid recorded in
    `runtime.json`, which is the supervisor's — QEMU is its child, so killing that pid alone
    leaves a running VM still holding the guest's RAM and the GPUs while every layer above
    reports a successful stop. `spawn` starts a new session precisely so this group exists.
    """
    if _process_group_gone(pid):
        return "process group was already gone"

    for signal_number, label in ((signal.SIGTERM, "SIGTERM"), (signal.SIGKILL, "SIGKILL")):
        try:
            os.killpg(pid, signal_number)
        except ProcessLookupError:
            return f"process group exited before {label}"
        except PermissionError:
            return f"no permission to send {label} to process group {pid}"
        if _wait_for_group(pid, min(timeout, TERMINATION_GRACE_SECONDS)):
            return f"process group stopped by {label}"

    return f"process group {pid} is still present after SIGKILL"
