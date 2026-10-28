from __future__ import annotations

import asyncio
import logging

from core.utils import _m, get_extra_info

from ..messages import SshdAuthLogMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# Written by the executor's sshd_log_receiver.py: one `Accepted ...` line per login, sshd's envelope kept.
SSHD_AUTH_LOG = "/var/log/sshd-auth.log"
# "<inode> <byte offset>" of the first line not logged yet; moved only after the lines are logged
SSHD_AUTH_CURSOR = "/var/log/sshd-auth.cursor"
# A cycle is ~15 min; the rest waits for the next cycle.
MAX_LOGINS_PER_CYCLE = 200
SSH_TIMEOUT_SECONDS = 10


def read_command(log: str = SSHD_AUTH_LOG, cursor: str = SSHD_AUTH_CURSOR) -> str:
    """Prints `<inode> <offset> <1 if reading .1>`, then up to MAX_LOGINS_PER_CYCLE lines from the offset.

    The live file is opened first and its inode, size and lines are all read through that descriptor, so a
    rotation during the command cannot pair the new file's lines with the old inode. A `.1` (the file rotated
    away) is read before the live file: from the cursor's offset when the cursor names it, from 0 when there is
    no usable cursor. Once it has been read to its end it is deleted, which lets the receiver rotate again. A
    cursor on the live file means any `.1` left was read already. Only `.1` may exist (the receiver rotated but
    could not create the live file, e.g. a full disk): it is still read, or deleted once read, to free the space."""
    return (
        f'f={log}; ino=; off=0; rotated=0; cleaned=; [ -f "$f" ] && exec 3<"$f" && ino=$(stat -L -c %i /dev/fd/3); '
        f"set -- $(cat {cursor} 2>/dev/null); c_ino=${{1:-}}; c_off=${{2:-0}}; "
        f"case $c_off in ''|*[!0-9]*) c_ino=; c_off=0;; esac; "
        f'if [ -n "$ino" ] && [ "$c_ino" = "$ino" ]; then off=$c_off; '
        f'elif [ -f "$f.1" ]; then o1=0; [ "$(stat -c %i "$f.1")" = "$c_ino" ] && o1=$c_off; '
        f'if [ "$o1" -lt "$(stat -c %s "$f.1")" ]; then '
        f'exec 3<"$f.1"; ino=$(stat -L -c %i /dev/fd/3); off=$o1; rotated=1; else rm -f "$f.1"; cleaned=1; fi; fi; '
        f'[ -n "$ino" ] || {{ [ -n "$cleaned" ] || exit 3; echo "0 0 0"; exit 0; }}; [ "$off" -le "$(stat -L -c %s /dev/fd/3)" ] || off=0; '
        f'echo "$ino $off $rotated"; tail -c +$((off + 1)) <&3 | head -n {MAX_LOGINS_PER_CYCLE}'
    )


def advance_command(inode: int, offset: int, cursor: str = SSHD_AUTH_CURSOR) -> str:
    return f'printf "%s %s\\n" {inode} {offset} > {cursor}.tmp && mv {cursor}.tmp {cursor}'


class SshdAuthLogCheck:
    """Ship the executor container's sshd logins to the validator log (DAH-4629, I-78).

    An unregistered key is appended to renters' pods from root sessions into executor containers,
    and nothing recorded which key opened those sessions. Each `Accepted publickey ... SHA256:<fp>`
    line not logged yet is logged here, so Loki names the key of every session, ours included
    (the validator makes a fresh key per job). Logged directly, not through the event: the sink
    drops an info event that repeats the last one to debug. Observation only: never fails or halts,
    runs first, so a node a fatal check stops and a rented node report too.
    """

    check_id = "executor.observe.sshd_auth_log"
    fatal = False

    def __init__(self, log: str = SSHD_AUTH_LOG, cursor: str = SSHD_AUTH_CURSOR):
        self.log, self.cursor = log, cursor

    async def run(self, ctx: Context) -> CheckResult:
        try:
            result = await self._ssh(ctx, read_command(self.log, self.cursor))
        except Exception as exc:  # noqa: BLE001 - observation only, never fatal; a timeout lands here too
            return self._unread(ctx, reason=f"ssh error: {exc!r}")
        if result.exit_status == 3:
            # an executor image from before DAH-4629
            return self._unread(ctx, reason="no sshd auth log on this executor")
        header, _, body = (result.stdout or "").partition("\n")
        try:
            inode, offset, rotated = (int(part) for part in header.split())
        except ValueError:
            return self._unread(ctx, reason=f"unreadable cursor line {header[:80]!r}")
        *complete, tail = body.split("\n")
        # only whole lines: a line still being written to the live file is read again next cycle
        lines = [f"{line}\n" for line in complete]
        logins = [line.strip() for line in lines if line.strip()]
        consumed = sum(len(line.encode("utf-8")) for line in lines)
        if rotated and tail:
            # .1 is never written again: a cut-off last record (a write the disk cut short) is reported and
            # passed, or the cursor would wait on it forever and never reach the live file
            fragment = tail
            logger.warning(
                _m(
                    "Executor sshd log: incomplete record skipped in the rotated file",
                    extra=get_extra_info(
                        {
                            **ctx.default_extra,
                            "executor_uuid": ctx.executor.uuid,
                            "fragment": fragment[:300],
                        }
                    ),
                )
            )
            consumed += len(fragment.encode("utf-8"))
        if logins:
            logger.info(
                _m(
                    "Executor sshd logins",
                    extra=get_extra_info(
                        {**ctx.default_extra, "executor_uuid": ctx.executor.uuid, "logins": logins}
                    ),
                )
            )
        if consumed:
            try:
                await self._ssh(ctx, advance_command(inode, offset + consumed, self.cursor))
            except Exception as exc:  # noqa: BLE001 - the same lines are logged again next cycle
                return self._unread(ctx, reason=f"cursor not moved: {exc!r}")
        event = render_message(
            Msg.LOGINS,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid, "login_count": len(logins)},
        )
        return CheckResult(passed=True, event=event)

    @staticmethod
    async def _ssh(ctx: Context, command: str):
        # a stalled executor must not hold up the checks behind this one
        return await asyncio.wait_for(
            ctx.ssh.run(command, check=False), timeout=SSH_TIMEOUT_SECONDS
        )

    def _unread(self, ctx: Context, *, reason: str) -> CheckResult:
        event = render_message(
            Msg.UNREAD,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid, "reason": reason},
        )
        return CheckResult(passed=True, event=event)
