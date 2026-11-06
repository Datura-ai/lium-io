"""Per-pod SSH host key: derivation, and pod_ssh_host_key.sh run as a real subprocess."""

from __future__ import annotations

import os
import shlex
import signal
import stat
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncssh
import pytest
from cryptography.hazmat.primitives import serialization

from services.pod_ssh_host_key import derive_pod_ssh_host_key

SCRIPT = Path(__file__).resolve().parent.parent / "src" / "services" / "assets" / "pod_ssh_host_key.sh"
SECRET = "test-master-secret-32-chars-long!!"
POD_ID = "00000000-0000-0000-0000-0000000000aa"


def _public_of(private_openssh: str) -> str:
    key = serialization.load_ssh_private_key(private_openssh.encode(), password=None)
    return key.public_key().public_bytes(
        serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
    ).decode()


def test_derived_host_key_is_the_same_for_the_same_pod():
    first = derive_pod_ssh_host_key(SECRET, POD_ID)

    second = derive_pod_ssh_host_key(SECRET, POD_ID)

    assert first.public_openssh == second.public_openssh
    assert first.public_openssh.startswith("ssh-ed25519 ")


def test_derived_host_key_differs_between_pods():
    first = derive_pod_ssh_host_key(SECRET, POD_ID)

    other = derive_pod_ssh_host_key(SECRET, "00000000-0000-0000-0000-0000000000bb")

    assert first.public_openssh != other.public_openssh


def test_derived_private_key_matches_its_public_key():
    host_key = derive_pod_ssh_host_key(SECRET, POD_ID)

    public = _public_of(host_key.private_openssh)

    assert public == host_key.public_openssh


def test_derivation_refuses_an_empty_pod_id():
    with pytest.raises(ValueError):
        derive_pod_ssh_host_key(SECRET, "")


def _run_script(tmp_path: Path, public: str, private: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "LIUM_RUN_DIR": str(tmp_path / "run"),
        "LIUM_SSH_HOST_KEY_DIR": str(tmp_path / "etc-ssh"),
    }
    return subprocess.run(
        ["sh", str(SCRIPT), public],
        input=private,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def _spawn_fake_sshd(tmp_path: Path) -> tuple[subprocess.Popen, Path]:
    hup_marker = tmp_path / "hup"
    proc = subprocess.Popen(
        ["sh", "-c", f"trap 'echo hup >> {hup_marker}' HUP; while :; do sleep 0.1; done"]
    )
    (tmp_path / "run").mkdir(parents=True, exist_ok=True)
    (tmp_path / "run" / "sshd.pid").write_text(f"{proc.pid}\n")
    time.sleep(0.3)  # let the shell install its trap before anything signals it
    return proc, hup_marker


def _wait_for(path: Path) -> bool:
    for _ in range(50):
        if path.exists():
            return True
        time.sleep(0.1)
    return False


def test_script_installs_the_host_key_with_private_permissions(tmp_path):
    host_key = derive_pod_ssh_host_key(SECRET, POD_ID)

    result = _run_script(tmp_path, host_key.public_openssh, host_key.private_openssh)

    key = tmp_path / "etc-ssh" / "ssh_host_ed25519_key"
    assert result.returncode == 0, result.stderr
    assert _public_of(key.read_text()) == host_key.public_openssh
    assert (tmp_path / "etc-ssh" / "ssh_host_ed25519_key.pub").read_text() == f"{host_key.public_openssh}\n"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert not (tmp_path / "run" / "lium-ssh-setup.lock").exists()


def test_script_replaces_an_image_generated_key_and_reloads_sshd(tmp_path):
    old = derive_pod_ssh_host_key(SECRET, "image-generated")
    _run_script(tmp_path, old.public_openssh, old.private_openssh)
    proc, hup_marker = _spawn_fake_sshd(tmp_path)
    host_key = derive_pod_ssh_host_key(SECRET, POD_ID)

    try:
        result = _run_script(tmp_path, host_key.public_openssh, host_key.private_openssh)
        reloaded = _wait_for(hup_marker)
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()

    assert result.returncode == 0, result.stderr
    assert _public_of((tmp_path / "etc-ssh" / "ssh_host_ed25519_key").read_text()) == host_key.public_openssh
    assert reloaded


def test_script_leaves_sshd_alone_when_the_key_is_already_installed(tmp_path):
    host_key = derive_pod_ssh_host_key(SECRET, POD_ID)
    _run_script(tmp_path, host_key.public_openssh, host_key.private_openssh)
    proc, hup_marker = _spawn_fake_sshd(tmp_path)

    try:
        result = _run_script(tmp_path, host_key.public_openssh, host_key.private_openssh)
        time.sleep(0.5)
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()

    assert result.returncode == 0, result.stderr
    assert "already installed" in result.stdout
    assert not hup_marker.exists()


def test_script_refuses_a_public_key_that_is_not_ed25519(tmp_path):
    host_key = derive_pod_ssh_host_key(SECRET, POD_ID)

    result = _run_script(tmp_path, "ssh-rsa AAAA", host_key.private_openssh)

    assert result.returncode == 2
    assert not (tmp_path / "etc-ssh" / "ssh_host_ed25519_key").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,refused,fallback_status", [(0, False, 0), (1, False, 0), (0, True, 0), (1, False, 1)])
async def test_install_uses_existing_ssh_and_keeps_private_key_out_of_command(monkeypatch, status, refused, fallback_status):
    from core.config import settings
    from services.docker_service import DockerService

    monkeypatch.setattr(settings, "VOLUME_MASTER_SECRET", SECRET)
    service = object.__new__(DockerService)
    ssh = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(exit_status=status)))
    ssh.run.side_effect = asyncssh.ChannelOpenError(1, "refused") if refused else None
    container = "pod name';$(touch /tmp/unwanted)"

    fallback = AsyncMock(return_value=SimpleNamespace(exit_status=fallback_status))
    await service.install_pod_ssh_host_key(
        ssh, docker_client=SimpleNamespace(exec_in_container=fallback), container_name=container, pod_id=POD_ID, log_extra={},
    )
    assert fallback.await_count == (status != 0 or refused)
    if fallback.await_count:
        assert fallback.await_args.args[0].stdin == ssh.run.call_args.kwargs["input"]

    args, kwargs = ssh.run.call_args
    argv = shlex.split(args[0])
    host_key = derive_pod_ssh_host_key(SECRET, POD_ID)
    assert argv[:6] == ["/usr/bin/docker", "exec", "-i", "-u", "0", container]
    assert argv[-1] == host_key.public_openssh
    assert host_key.private_openssh not in args[0]
    assert _public_of(kwargs.pop("input")) == host_key.public_openssh
    assert kwargs == {"check": False}
    ssh.run.assert_awaited_once()
