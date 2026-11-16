from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest
from services.task.checks.executor_key_cleanup import (
    EXECUTOR_FOREIGN_KEY_BODIES,
    ExecutorKeyCleanupCheck,
    cleanup_command,
    fingerprint,
)
from services.task.messages import ExecutorKeyCleanupMessages as Msg

FOREIGN = f"ssh-ed25519 {EXECUTOR_FOREIGN_KEY_BODIES[1]} x@y"
MARKED = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFUX52GPGfsqHZvpAG23FhPPJ5xFbCShipPgmWgxCUnl lium-uploaded-at=1700000000"
OWN_BODY = "AAAAC3NzaC1lZDI1NTE5AAAAIGxyN7ezXSfGyqM4/VUSwY1I/eur+1sFbwj8nwrfbfJl"
OWN = f"ssh-ed25519 {OWN_BODY} provider@laptop"


class LocalShell:
    """ctx.ssh over a local `sh`, so the command runs for real against tmp files."""

    def __init__(self, fail: bool = False, before: str = ""):
        self.fail, self.before = fail, before

    async def run(self, command: str, check: bool = False):
        if self.fail:
            raise OSError("connection lost")
        done = subprocess.run(["sh", "-c", self.before + command], capture_output=True, text=True)
        return SimpleNamespace(stdout=done.stdout, exit_status=done.returncode)


def _events(caplog) -> list[dict]:
    return [
        r.msg.extra
        for r in caplog.records
        if getattr(r.msg, "message", None) == "Executor authorized_keys: foreign or unmarked keys"
    ]


async def _run(context_factory, tmp_path, ssh):
    return await ExecutorKeyCleanupCheck(str(tmp_path / "authorized_keys"), str(tmp_path)).run(
        context_factory(ssh=ssh)
    )


def test_the_command_holds_only_constants():
    command = cleanup_command()

    assert all(body in command for body in EXECUTOR_FOREIGN_KEY_BODIES)
    assert "/root/.ssh/authorized_keys" in command and "/root/.vh_pwned" in command


@pytest.mark.asyncio
async def test_only_the_foreign_lines_are_removed_and_one_event_names_the_rest_by_fingerprint(
    context_factory, tmp_path, caplog
):
    keys = tmp_path / "authorized_keys"
    keys.write_text(
        f"{MARKED}\n{FOREIGN}\n{OWN}\n\n# note\nssh-ed25519 {EXECUTOR_FOREIGN_KEY_BODIES[0]}\n"
    )
    keys.chmod(0o600)
    (tmp_path / ".vh_pwned").touch()

    with caplog.at_level("INFO"):
        result = await _run(context_factory, tmp_path, LocalShell())

    assert keys.read_text() == f"{MARKED}\n{OWN}\n\n# note\n"
    assert keys.stat().st_mode & 0o777 == 0o600
    assert result.passed is True and result.event.reason_code == Msg.CHECKED.reason
    [event] = _events(caplog)
    assert event["foreign_keys_removed"] == 2 and event["unmarked_keys"] == 1
    assert event["unmarked_fingerprints"] == [fingerprint(OWN)]
    assert event["vh_pwned_marker"] is True and event["xstar8_pwned_marker"] is False
    assert event["removal_failed"] is False
    assert not any(
        OWN_BODY in r.getMessage() or OWN_BODY in str(r.msg.extra) for r in caplog.records
    )


@pytest.mark.asyncio
async def test_a_line_appended_while_the_file_is_rewritten_is_kept(context_factory, tmp_path):
    keys = tmp_path / "authorized_keys"
    keys.write_text(f"{FOREIGN}\n{MARKED}\n")
    # an upload lands on the old file right after its size was taken
    append = f"echo {MARKED.replace('1700000000', '1700000001')!r} >> {keys}; "
    ssh = LocalShell()
    original = cleanup_command(str(keys), str(tmp_path))
    patched = original.replace("|| exit 1; ", f"|| exit 1; {append}", 1)
    assert patched != original

    async def run(command, check=False):
        return await LocalShell().run(patched, check)

    ssh.run = run
    await _run(context_factory, tmp_path, ssh)

    assert keys.read_text() == f"{MARKED}\n{MARKED.replace('1700000000', '1700000001')}\n"


@pytest.mark.asyncio
async def test_a_clean_file_is_left_alone_and_logs_nothing(context_factory, tmp_path, caplog):
    keys = tmp_path / "authorized_keys"
    keys.write_text(f"{MARKED}\n")
    inode = keys.stat().st_ino

    with caplog.at_level("INFO"):
        result = await _run(context_factory, tmp_path, LocalShell())

    assert keys.stat().st_ino == inode and keys.read_text() == f"{MARKED}\n"
    assert result.passed is True and _events(caplog) == []


@pytest.mark.asyncio
async def test_a_lost_connection_never_fails_the_node(context_factory, tmp_path):
    result = await _run(context_factory, tmp_path, LocalShell(fail=True))

    assert result.passed is True and result.event.reason_code == Msg.UNREAD.reason


def test_fingerprint_matches_ssh_keygen(tmp_path):
    pub = tmp_path / "k.pub"
    pub.write_text(f"{OWN}\n")
    expected = subprocess.run(
        ["ssh-keygen", "-lf", str(pub)], capture_output=True, text=True
    ).stdout.split()[1]

    assert fingerprint(OWN) == expected
