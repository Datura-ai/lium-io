"""Enter a workspace path inside a trusted root without following any symlink.

The storage helpers reach workspace paths from outside the container (the rental's root seen from a
host-PID helper, or a mounted /workspace). restic, tar or sh given such a path would follow a link
the workspace holds, possibly out of the workspace. The script walks the path one component at a time
with O_NOFOLLOW from the root, refuses any symlink, fchdirs into the result and execs the command
there, so the command only ever sees "."."""

from __future__ import annotations

from pathlib import PurePosixPath

PATH_MISSING_EXIT = 66
NOT_A_DIRECTORY_EXIT = 20
NOT_EMPTY_EXIT = 21
SYMLINK_EXIT = 67

# backup: absent or empty -> 66 (EMPTY no-op, DAH-2002). restore: missing parts are created.
# restore-empty: as restore, and the target must be empty. check-*: same shape checks, an absent
# path passes and nothing is created or run.
MODES = (
    "backup",
    "restore",
    "restore-empty",
    "check-backup",
    "check-restore",
    "check-restore-empty",
)

SAFE_ENTER_SCRIPT = r"""
import os, stat, sys
root, rel, mode = sys.argv[1:4]
command = sys.argv[4:]
fd = os.open(root, os.O_PATH | os.O_DIRECTORY)
# "." is the root itself: a whole-volume backup or restore maps the volume path to the root
for part in [p for p in rel.split("/") if p not in ("", ".")]:
    if part == "..":
        sys.stderr.write("path component not allowed: %s\n" % part)
        sys.exit(67)
    try:
        nxt = os.open(part, os.O_PATH | os.O_NOFOLLOW, dir_fd=fd)
    except FileNotFoundError:
        if mode == "backup":
            sys.exit(66)
        if mode.startswith("check-"):
            sys.exit(0)
        os.mkdir(part, 0o755, dir_fd=fd)
        nxt = os.open(part, os.O_PATH | os.O_NOFOLLOW, dir_fd=fd)
    kind = os.fstat(nxt).st_mode
    if stat.S_ISLNK(kind):
        sys.stderr.write("symlink in path: %s\n" % part)
        sys.exit(67)
    if not stat.S_ISDIR(kind):
        sys.stderr.write("not a directory: %s\n" % part)
        sys.exit(20)
    os.close(fd)
    fd = nxt
os.fchdir(fd)
entries = os.listdir(".")
if mode == "backup" and not entries:
    sys.exit(66)
if mode.endswith("restore-empty") and entries:
    sys.exit(21)
if command:
    os.execvp(command[0], command)
"""


def safe_enter_command(
    root: PurePosixPath, path: PurePosixPath, mode: str, command: list[str]
) -> list[str]:
    """argv that runs `command` with its cwd at `path`, reached from `root` without symlinks."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    relative = path.relative_to(root)
    return [
        "python3",
        "-I",
        "-S",
        "-c",
        SAFE_ENTER_SCRIPT,
        str(root),
        str(relative),
        mode,
        *command,
    ]
