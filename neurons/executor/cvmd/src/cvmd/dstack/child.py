"""The supervisor process: one CVM, held for its whole life.

`DStackManager.run_instance` calls `subprocess.run(qemu, check=True)` — it does not return until
the VM is gone — and the host API server `start_server` opens has to stay up beside it for the
guest to fetch its sealing key. Neither fits inside an HTTP handler, so cvmd spawns this module
and the QEMU process lives under it.

The process is deliberately in its own session (the parent passes `start_new_session=True`).
A CVM must outlive a cvmd restart: a daemon upgrade that took the node's CVM with it would be
worse than the outage it was fixing, and `RECONCILING` exists precisely so cvmd can rediscover
a CVM it did not start.

Run as: `python -m cvmd.dstack.child <scripts_dir> <vm_dir> <kp_port>`

stdout carries the guest's serial console — QEMU is started with `-nographic -serial chardev:ser0`
over stdio — plus the full QEMU command line that `run_instance` prints before exec. The parent
points both streams at `<vm_dir>/console.log`, which makes that file the primary evidence for
what was launched and how it booted.
"""

import sys
from pathlib import Path

from cvmd.dstack.loader import DStackUnavailable, load_dstack

USAGE = "usage: python -m cvmd.dstack.child <scripts_dir> <vm_dir> <kp_port>"

# Distinct from any exit code QEMU itself can return, so the parent can tell "the launcher never
# got started" apart from "the VM ran and exited".
EXIT_BAD_USAGE = 64
EXIT_NO_DSTACK = 65
EXIT_LAUNCH_FAILED = 66


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

    try:
        dstack = load_dstack(Path(scripts_dir))
    except DStackUnavailable as exc:
        print(f"cvmd: cannot import the dstack launcher: {exc}", file=sys.stderr)
        return EXIT_NO_DSTACK

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
