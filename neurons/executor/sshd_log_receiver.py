"""The container's /dev/log for sshd: it runs no syslog daemon, so sshd's lines would be dropped.

Keeps the `Accepted ...` login lines with sshd's own envelope (time and `sshd[pid]`), prefixed by the
time received in UTC, in SSHD_AUTH_LOG for the validator to read (it keeps its own cursor there)."""

import os
import re
import socket
from datetime import UTC, datetime

DEV_LOG = "/dev/log"
SSHD_AUTH_LOG = "/var/log/sshd-auth.log"
# past this the file is renamed to .1, but only while no .1 exists: the validator deletes .1 once its cursor has
# read all of it, so an unread .1 is never overwritten (the live file keeps growing until then, up to MAX_SPOOL_BYTES)
ROTATE_BYTES = 50 * 1024 * 1024
# a session that authenticated, matched right after libc's syslog envelope ("<pri>Mmm dd hh:mm:ss sshd[pid]: ")
# so a client-chosen user name inside "Invalid user ..." cannot pose as one; sshd-session is OpenSSH 9.8+.
# VERBOSE also logs "Accepted key ... found at" for a key merely offered, which "Accepted <method> for" skips.
LOGIN = re.compile(
    r"^<\d+>[A-Z][a-z]{2} \d{1,2} \d\d:\d\d:\d\d sshd(?:-session)?\[\d+\]: Accepted \S+ for "
)
# live file plus an unread .1: past this, logins are dropped (with one marker line, then a count once there is room)
# so session churn faster than the validator drains cannot fill the executor's disk
MAX_SPOOL_BYTES = 2 * ROTATE_BYTES
_dropped = 0
# records kept while the disk refuses writes; sshd must never wait on this socket
MAX_PENDING = 1000


def is_login(message: str) -> bool:
    return bool(LOGIN.match(" ".join(message.split())))


def _size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def record(message: str, log: str = SSHD_AUTH_LOG) -> None:
    global _dropped
    line = " ".join(message.split())
    if not LOGIN.match(line):
        return
    now = datetime.now(UTC).isoformat(timespec="seconds")
    if _size(log) + _size(f"{log}.1") >= MAX_SPOOL_BYTES:
        if not _dropped:
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"{now} sshd-log-receiver: spool full, dropping logins\n")
        _dropped += 1
        return
    if (
        os.path.exists(log)
        and os.path.getsize(log) > ROTATE_BYTES
        and not os.path.exists(f"{log}.1")
    ):
        os.replace(log, f"{log}.1")
    with open(log, "a", encoding="utf-8") as f:
        if _dropped:
            f.write(
                f"{now} sshd-log-receiver: {_dropped} logins dropped while the spool was full\n"
            )
        f.write(f"{now} {line}\n")
    _dropped = 0


def main() -> None:
    if os.path.exists(DEV_LOG):
        os.unlink(DEV_LOG)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(DEV_LOG)
    os.chmod(DEV_LOG, 0o666)
    serve(sock)


def serve(sock: socket.socket) -> None:
    """A failed write (a full disk) keeps the records and retries with backoff instead of ending the receiver."""
    pending: list[str] = []
    delay = None
    while True:
        sock.settimeout(delay)
        try:
            message = sock.recv(16384).decode("utf-8", errors="replace")
            # filtered before it is kept: scanner noise during a full disk must not push a login out
            if is_login(message):
                pending.append(message)
        except TimeoutError:
            pass
        try:
            while pending:
                record(pending[0])
                pending.pop(0)
            delay = None
        except OSError:
            del pending[:-MAX_PENDING]
            delay = min((delay or 0.5) * 2, 60)


if __name__ == "__main__":
    main()
