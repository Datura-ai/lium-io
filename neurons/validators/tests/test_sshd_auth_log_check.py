from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from services.task.checks import sshd_auth_log
from services.task.checks.sshd_auth_log import SshdAuthLogCheck
from services.task.messages import SshdAuthLogMessages as Msg

# what the executor's sshd_log_receiver.py writes: its UTC time, then sshd's syslog envelope
LOGIN_A = "2026-10-18T06:27:30+00:00 <38>Oct 18 06:27:30 sshd[812]: Accepted publickey for root from 203.0.113.9 port 51022 ssh2: ED25519 SHA256:aaa"
LOGIN_B = "2026-10-18T06:27:31+00:00 <38>Oct 18 06:27:31 sshd[813]: Accepted publickey for root from 203.0.113.9 port 51023 ssh2: ED25519 SHA256:bbb"


class LocalShell:
    """ctx.ssh over a local `sh`, so the check's commands run for real against tmp files."""

    def __init__(
        self,
        fail_on: str | None = None,
        after_open: str = "",
        at: str = 'exec 3<"$f" && ino=$(stat -L -c %i /dev/fd/3); ',
    ):
        self.fail_on = fail_on
        self.after_open = after_open  # shell run right after `at` in the read command (default: opening the live file)
        self.at = at
        self.commands: list[str] = []

    async def run(self, command: str, check: bool = False):
        self.commands.append(command)
        if self.fail_on and self.fail_on in command:
            raise OSError("connection lost")
        if self.after_open:
            command = command.replace(self.at, f"{self.at}{self.after_open}; ", 1)
        done = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return SimpleNamespace(stdout=done.stdout, exit_status=done.returncode)


def _logged(caplog) -> list[list[str]]:
    return [
        r.msg.extra["logins"]
        for r in caplog.records
        if getattr(r.msg, "message", None) == "Executor sshd logins"
    ]


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "sshd-auth.log", tmp_path / "sshd-auth.cursor"


async def _cycle(context_factory, paths, ssh):
    log, cursor = paths
    return await SshdAuthLogCheck(str(log), str(cursor)).run(context_factory(ssh=ssh))


@pytest.mark.asyncio
async def test_each_login_is_logged_once_with_sshds_envelope(context_factory, paths, caplog):
    paths[0].write_text(f"{LOGIN_A}\n")
    with caplog.at_level("INFO"):
        first = await _cycle(context_factory, paths, LocalShell())
        second = await _cycle(context_factory, paths, LocalShell())

    assert first.event.what_we_saw["login_count"] == 1
    assert second.event.what_we_saw["login_count"] == 0
    assert _logged(caplog) == [[LOGIN_A]]
    assert paths[0].read_text() == f"{LOGIN_A}\n"  # nothing on the executor is erased


@pytest.mark.asyncio
async def test_a_login_written_after_the_read_is_logged_next_cycle(context_factory, paths, caplog):
    paths[0].write_text(f"{LOGIN_A}\n")
    ssh = LocalShell()
    original = ssh.run

    async def append_after_read(command, check=False):
        result = await original(command, check=check)
        if "tail -c" in command:
            with paths[0].open("a") as f:
                f.write(f"{LOGIN_B}\n")
        return result

    ssh.run = append_after_read
    with caplog.at_level("INFO"):
        await _cycle(context_factory, paths, ssh)
        await _cycle(context_factory, paths, LocalShell())

    assert _logged(caplog) == [[LOGIN_A], [LOGIN_B]]


@pytest.mark.asyncio
async def test_a_lost_connection_before_the_cursor_moves_logs_the_batch_again(
    context_factory, paths, caplog
):
    paths[0].write_text(f"{LOGIN_A}\n")
    with caplog.at_level("INFO"):
        lost = await _cycle(context_factory, paths, LocalShell(fail_on="mv "))
        await _cycle(context_factory, paths, LocalShell())

    assert lost.passed is True and lost.event.reason_code == Msg.UNREAD.reason
    assert _logged(caplog) == [[LOGIN_A], [LOGIN_A]]


@pytest.mark.asyncio
async def test_more_logins_than_a_cycle_takes_wait_for_the_next(
    context_factory, paths, caplog, monkeypatch
):
    monkeypatch.setattr(sshd_auth_log, "MAX_LOGINS_PER_CYCLE", 1)
    paths[0].write_text(f"{LOGIN_A}\n{LOGIN_B}\n")
    with caplog.at_level("INFO"):
        await _cycle(context_factory, paths, LocalShell())
        await _cycle(context_factory, paths, LocalShell())

    assert _logged(caplog) == [[LOGIN_A], [LOGIN_B]]


@pytest.mark.asyncio
async def test_a_rotated_log_is_read_from_its_start(context_factory, paths, caplog):
    paths[0].write_text(f"{LOGIN_A}\n")
    await _cycle(context_factory, paths, LocalShell())
    paths[0].rename(f"{paths[0]}.1")
    paths[0].write_text(f"{LOGIN_B}\n")

    with caplog.at_level("INFO"):
        await _cycle(context_factory, paths, LocalShell())

    assert _logged(caplog) == [[LOGIN_A], [LOGIN_B]]  # B from the new file's start, A not again


LOGIN_C = LOGIN_B.replace("51023", "51024").replace("SHA256:bbb", "SHA256:ccc")


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [f"{LOGIN_B}\n", LOGIN_B[:40]])
async def test_logins_left_unread_in_a_rotated_log_are_read_before_the_new_file(
    context_factory, paths, caplog, tail
):
    paths[0].write_text(f"{LOGIN_A}\n")
    await _cycle(context_factory, paths, LocalShell())
    with paths[0].open("a") as f:
        f.write(tail)  # complete, or cut short by a full disk
    paths[0].rename(f"{paths[0]}.1")
    paths[0].write_text(f"{LOGIN_C}\n")

    with caplog.at_level("INFO"):
        for _ in range(3):
            await _cycle(context_factory, paths, LocalShell())

    expected = [[LOGIN_A], [LOGIN_B], [LOGIN_C]] if tail.endswith("\n") else [[LOGIN_A], [LOGIN_C]]
    assert _logged(caplog) == expected


@pytest.mark.asyncio
async def test_an_executor_without_the_log_passes_unread(context_factory, paths):
    result = await _cycle(context_factory, paths, LocalShell())

    assert result.passed is True
    assert result.event.reason_code == Msg.UNREAD.reason


@pytest.mark.asyncio
async def test_a_stalled_executor_times_out_unread(context_factory, paths, monkeypatch):
    monkeypatch.setattr(sshd_auth_log, "SSH_TIMEOUT_SECONDS", 0.01)
    ssh = AsyncMock()

    async def stall(*_args, **_kwargs):
        await asyncio.sleep(5)

    ssh.run.side_effect = stall

    result = await _cycle(context_factory, paths, ssh)

    assert result.passed is True
    assert result.event.reason_code == Msg.UNREAD.reason


@pytest.mark.asyncio
async def test_a_rotation_during_the_read_keeps_the_old_files_lines_with_its_inode(
    context_factory, paths, caplog
):
    log, _ = paths
    log.write_text(f"{LOGIN_A}\n")
    await _cycle(context_factory, paths, LocalShell())
    with log.open("a") as f:
        f.write(f"{LOGIN_B}\n")
    rotate = f'mv "{log}" "{log}.1" && printf "%s\\n" "{LOGIN_C}" > "{log}"'

    with caplog.at_level("INFO"):
        await _cycle(context_factory, paths, LocalShell(after_open=rotate))
        for _ in range(2):
            await _cycle(context_factory, paths, LocalShell())

    assert _logged(caplog) == [[LOGIN_A], [LOGIN_B], [LOGIN_C]]
    assert not (
        log.parent / f"{log.name}.1"
    ).exists()  # read to its end, so deleted for the next rotation


@pytest.mark.parametrize(
    "at", ['exec 3<"$f" && ino=$(stat -L -c %i /dev/fd/3); ', "then off=$c_off; "]
)
@pytest.mark.asyncio
async def test_a_rotation_after_open_keeps_lines_past_the_cap(
    context_factory, paths, caplog, monkeypatch, at
):
    monkeypatch.setattr(sshd_auth_log, "MAX_LOGINS_PER_CYCLE", 1)
    log, _ = paths
    log.write_text(f"{LOGIN_A}\n")
    await _cycle(context_factory, paths, LocalShell())
    with log.open("a") as f:
        f.write(f"{LOGIN_B}\n{LOGIN_C}\n")
    rotate = f'mv "{log}" "{log}.1" && : > "{log}"'

    with caplog.at_level("INFO"):
        await _cycle(context_factory, paths, LocalShell(after_open=rotate, at=at))
        for _ in range(2):
            await _cycle(context_factory, paths, LocalShell())

    assert _logged(caplog) == [[LOGIN_A], [LOGIN_B], [LOGIN_C]]


@pytest.mark.asyncio
async def test_an_unread_rotated_log_is_kept(context_factory, paths, monkeypatch):
    monkeypatch.setattr(sshd_auth_log, "MAX_LOGINS_PER_CYCLE", 1)
    log, _ = paths
    log.write_text(f"{LOGIN_A}\n")
    await _cycle(context_factory, paths, LocalShell())
    with log.open("a") as f:
        f.write(f"{LOGIN_B}\n{LOGIN_C}\n")
    log.rename(f"{log}.1")
    log.write_text("")

    await _cycle(context_factory, paths, LocalShell())

    assert (log.parent / f"{log.name}.1").read_text().endswith(f"{LOGIN_C}\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor_text", [None, "garbage x\n", "999999999 12\n"])
async def test_a_rotated_log_without_a_usable_cursor_is_read_then_removed(
    context_factory, paths, caplog, cursor_text
):
    log, cursor = paths
    (log.parent / f"{log.name}.1").write_text(f"{LOGIN_A}\n")
    log.write_text(f"{LOGIN_B}\n")
    if cursor_text is not None:
        cursor.write_text(cursor_text)

    with caplog.at_level("INFO"):
        for _ in range(3):
            await _cycle(context_factory, paths, LocalShell())

    assert _logged(caplog) == [[LOGIN_A], [LOGIN_B]]
    assert not (log.parent / f"{log.name}.1").exists()


@pytest.mark.parametrize("acknowledged", [True, False])
@pytest.mark.asyncio
async def test_a_rotation_without_a_new_live_file_is_still_collected(
    context_factory, paths, caplog, acknowledged
):
    log, _ = paths
    log.write_text(f"{LOGIN_A}\n")
    await _cycle(context_factory, paths, LocalShell())
    if not acknowledged:
        with log.open("a") as f:
            f.write(f"{LOGIN_B}\n")
    log.rename(f"{log}.1")  # the receiver rotated, then creating the live file failed (ENOSPC)

    with caplog.at_level("INFO"):
        result = await _cycle(context_factory, paths, LocalShell())

    assert result.event.reason_code != Msg.UNREAD.reason  # not "no sshd auth log on this executor"
    assert _logged(caplog) == [[LOGIN_A]] + ([] if acknowledged else [[LOGIN_B]])
    assert (log.parent / f"{log.name}.1").exists() is (not acknowledged)
