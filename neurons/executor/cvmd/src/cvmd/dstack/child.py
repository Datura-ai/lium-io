"""The supervisor process: one CVM, held for its whole life.

`DStackManager.run_instance` calls `subprocess.run(qemu, check=True)` — it does not return until
the VM is gone — and the host API server `start_server` opens has to stay up beside it for the
guest to fetch its sealing key. Neither fits inside an HTTP handler, so cvmd spawns this module
and the QEMU process lives under it.

The process **double-forks**: the one cvmd starts forks, exits, and the grandchild holds the VM
with init as its parent. A CVM must outlive a cvmd restart — a daemon upgrade that took the
node's CVM with it would be worse than the outage it was fixing, and `RECONCILING` exists
precisely so cvmd can rediscover a CVM it did not start. Orphaning to init also means the
supervisor is reaped the instant it exits rather than lingering as a zombie in its own process
group, which is what made every teardown on au11 burn its full signal ladder and then report a
failure that had not happened.

Run as: `python -u -m cvmd.dstack.child <scripts_dir> <vm_dir> <kp_port>`

stdout carries the guest's serial console — QEMU is started with `-nographic -serial chardev:ser0`
over stdio — plus the full QEMU command line that `run_instance` prints before exec. The parent
points both streams at `<vm_dir>/console.log`, which makes that file the primary evidence for
what was launched and how it booted. `-u` is not optional: block-buffered, that command line
stays in this process's buffer for the VM's whole life.
"""

import os
import sys
from pathlib import Path

from cvmd.dstack.loader import DStackUnavailable, load_dstack

USAGE = "usage: python -u -m cvmd.dstack.child <scripts_dir> <vm_dir> <kp_port>"

PID_FILE = "supervisor.pid"

# Distinct from any exit code QEMU itself can return, so the parent can tell "the launcher never
# got started" apart from "the VM ran and exited".
EXIT_BAD_USAGE = 64
EXIT_NO_DSTACK = 65
EXIT_LAUNCH_FAILED = 66
EXIT_NO_DETACH = 67


def _detach(vm_dir: Path) -> None:
    """Fork, orphan this process to init, and record the pid that holds the VM.

    The fork parent leaves through `os._exit`, which skips atexit handlers and — the part that
    matters — skips flushing the stdio buffers it inherited. A normal exit would replay whatever
    this process had already buffered into the console log a second time.

    `setsid` after the fork makes the surviving process its own session and process-group
    leader, so QEMU lands in a group whose id is this pid. That is what lets cvmd stop the VM by
    signalling the group: dstack's own `--force` kills the pid in `runtime.json`, which is this
    process, leaving QEMU orphaned and still holding the guest's RAM.
    """
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    (vm_dir / PID_FILE).write_text(f"{os.getpid()}\n")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print(USAGE, file=sys.stderr)
        return EXIT_BAD_USAGE

    scripts_dir, vm_dir, raw_port = argv
    try:
        kp_port = int(raw_port)
    except ValueError:
        print(f"{USAGE}\nkp_port must be an integer, got {raw_port!r}", file=sys.stderr)
        return EXIT_BAD_USAGE

    # Before the launcher import, so a host that cannot import dstack.py fails as cvmd's
    # immediate child with a legible exit code rather than as a detached process cvmd has
    # already recorded and has to discover the death of.
    try:
        dstack = load_dstack(Path(scripts_dir))
    except DStackUnavailable as exc:
        print(f"cvmd: cannot import the dstack launcher: {exc}", file=sys.stderr)
        return EXIT_NO_DSTACK

    try:
        _detach(Path(vm_dir))
    except OSError as exc:
        print(f"cvmd: the supervisor could not detach: {exc}", file=sys.stderr)
        return EXIT_NO_DETACH

    # The same two calls dstack.py's own `run` subcommand makes, in the same order. The host
    # API server has to be listening before the guest boots, and `run_instance` needs the port
    # it landed on to write into the guest config.
    try:
        server = dstack.start_server(vm_dir, kp_port)
        dstack.DStackManager().run_instance(vm_dir, server.host_port)
    except Exception as exc:  # noqa: BLE001 - the child's job is to report, not to interpret
        print(f"cvmd: the CVM exited abnormally: {exc}", file=sys.stderr)
        return EXIT_LAUNCH_FAILED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
