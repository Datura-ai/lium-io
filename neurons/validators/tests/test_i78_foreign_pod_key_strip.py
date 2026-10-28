import os
import subprocess
import tempfile
from types import SimpleNamespace

import pytest

from services.task.checks.rented_machine import (
    FOREIGN_POD_KEY_BODIES,
    _check_pod_running_and_read_authorized_keys,
)

FOREIGN = f"ssh-ed25519 {FOREIGN_POD_KEY_BODIES[0]}"
RENTER = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIrenterkey renter@laptop"


class FakeSSH:
    def __init__(self, keys, strip_exit=0):
        self.keys, self.strip_exit, self.commands = keys, strip_exit, []

    async def run(self, command):
        self.commands.append(command)
        if "docker ps" in command:
            return SimpleNamespace(stdout="cid", stderr="", exit_status=0)
        if "grep -vF" in command:
            return SimpleNamespace(stdout="", stderr="", exit_status=self.strip_exit)
        return SimpleNamespace(stdout="\n".join(self.keys), stderr="", exit_status=0)


@pytest.mark.asyncio
async def test_foreign_key_is_stripped_and_not_reported():
    ssh = FakeSSH([RENTER, FOREIGN, FOREIGN])
    result = await _check_pod_running_and_read_authorized_keys(ssh, "pod_x")
    assert result.authorized_keys == [RENTER]
    assert any("grep -vF" in c and "pod_x" in c for c in ssh.commands)


@pytest.mark.asyncio
async def test_clean_pod_gets_no_strip_exec():
    ssh = FakeSSH([RENTER])
    result = await _check_pod_running_and_read_authorized_keys(ssh, "pod_x")
    assert result.authorized_keys == [RENTER]
    assert not any("grep -vF" in c for c in ssh.commands)


@pytest.mark.asyncio
async def test_failed_strip_still_reports_the_key():
    ssh = FakeSSH([RENTER, FOREIGN], strip_exit=2)
    result = await _check_pod_running_and_read_authorized_keys(ssh, "pod_x")
    assert result.authorized_keys == [RENTER, FOREIGN]


@pytest.mark.asyncio
@pytest.mark.parametrize("lines", [[RENTER, FOREIGN, f"no-pty {FOREIGN} x@y", RENTER], [FOREIGN]])
async def test_strip_script_keeps_only_other_lines(lines):
    # run the exact script the validator sends, against a real file, with sh
    ssh = FakeSSH(lines)
    await _check_pod_running_and_read_authorized_keys(ssh, "pod_x")
    sent = next(c for c in ssh.commands if "grep -vF" in c)
    script = sent.split(" sh -c ", 1)[1]
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "authorized_keys")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.chmod(path, 0o600)
        run = subprocess.run(
            ["sh", "-c", f"sh -c {script}".replace("/root/.ssh/authorized_keys", path)], capture_output=True, text=True
        )
        assert run.returncode == 0, run.stderr
        assert open(path).read().split() == " ".join(l for l in lines if FOREIGN_POD_KEY_BODIES[0] not in l).split()
        assert os.stat(path).st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_strip_script_leaves_the_file_alone_when_grep_errors():
    # grep exit 2 (here: unreadable file; in a pod: a full disk) must not copy the empty temp file over the keys
    ssh = FakeSSH([RENTER, FOREIGN])
    await _check_pod_running_and_read_authorized_keys(ssh, "pod_x")
    script = next(c for c in ssh.commands if "grep -vF" in c).split(" sh -c ", 1)[1]
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "authorized_keys")
        run = subprocess.run(["sh", "-c", f"sh -c {script}".replace("/root/.ssh/authorized_keys", path)], capture_output=True, text=True)
        assert run.returncode != 0 and not os.path.exists(path)
