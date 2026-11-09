"""scripts/install_executor_on_ubuntu.sh run against stub commands (ticket 1124: Docker preinstalled, user not in
the docker group). The stubs keep group membership in a file and fail `docker info` like the real socket does."""

import os
import pty
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "install_executor_on_ubuntu.sh"

STUBS = {
    "apt-get": 'echo "apt-get $*" >> "$STATE/log"; [[ "$*" == *install*compose* && -n "$COMPOSE_INSTALLABLE" ]] && touch "$STATE/compose"; exit 0',
    # -v caches the password; with SUDO_NEEDS_PASSWORD, `sudo -n` fails until then
    "sudo": '[[ "$1" == -v ]] && { touch "$STATE/sudo_ok"; exit 0; }; [[ "$1" == -n ]] && shift; '
    '[[ -n "$SUDO_NEEDS_PASSWORD" && ! -f "$STATE/sudo_ok" ]] && { echo "sudo: a password is required" >&2; exit 1; }; '
    '[[ "$1" == true ]] && exit 0; exec "$@"',
    "systemctl": 'echo "systemctl $*" >> "$STATE/log"; exit 0',
    "groupadd": "exit 0",
    "usermod": 'echo "usermod $*" >> "$STATE/log"; touch "$STATE/in_group"',
    # `id -nG user` reads the group database; `id -nG` alone is this process's groups (only sg adds docker)
    "id": '[[ "$1" == -u ]] && { echo 1000; exit 0; }; '
    'if [[ $# -eq 2 ]]; then [[ -f "$STATE/in_group" ]] && echo "u docker" || echo u; '
    'else [[ -n "$IN_DOCKER_GROUP" ]] && echo "u docker" || echo u; fi',
    "sg": 'shift; shift; IN_DOCKER_GROUP=1 exec bash -c "$1"',
    "docker": 'case "$1" in '
    "--version) echo 'Docker version 29.1.3';; "
    'info) [[ -n "$DAEMON_DOWN" ]] && { echo "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?" >&2; exit 1; }; '
    '[[ -n "$IN_DOCKER_GROUP" ]] || { echo "permission denied while trying to connect to the docker API at unix:///var/run/docker.sock" >&2; exit 1; };; '
    'compose) [[ -f "$STATE/compose" ]] || { echo "unknown command: docker compose" >&2; exit 1; };; '
    "esac",
}


def _run(
    tmp_path: Path, compose: bool = True, tty: bool = False, **env: str
) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    (state / "log").touch()
    if compose:
        (state / "compose").touch()
    for name, body in STUBS.items():
        path = bin_dir / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    base = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "STATE": str(state),
        "USER": "u",
        "SHELL": "/bin/true",
        "HOME": str(tmp_path),
    }
    if tty:
        return _run_at_a_terminal({**base, **env}), (state / "log").read_text()
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**base, **env},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    return proc, (state / "log").read_text()


def _run_at_a_terminal(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """The script on a pseudo-terminal (stdin, stdout and stderr are a TTY); the output comes back as stdout."""
    master, slave = pty.openpty()
    proc = subprocess.Popen(["bash", str(SCRIPT)], env=env, stdin=slave, stdout=slave, stderr=slave)
    os.close(slave)
    out = b""
    while True:
        try:
            chunk = os.read(master, 4096)
        except OSError:  # EIO once the child closed the terminal
            break
        if not chunk:
            break
        out += chunk
    os.close(master)
    return subprocess.CompletedProcess(
        proc.args, proc.wait(timeout=60), out.decode(errors="replace"), ""
    )


pytestmark = pytest.mark.skipif(
    os.geteuid() == 0, reason="the script takes the root path; these cover a sudo user"
)


def test_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_preinstalled_docker_adds_the_user_to_the_docker_group_and_verifies_through_sg(
    tmp_path: Path,
) -> None:
    proc, log = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "Docker already installed" in proc.stdout
    assert "usermod -aG docker u" in log
    assert "Docker daemon reachable & compose available" in proc.stdout
    # no terminal (lium mine captures the output): no new shell, a re-login hint instead
    assert "newgrp docker" in proc.stdout and "Activating docker group" not in proc.stdout


def test_a_stopped_daemon_is_reported_as_stopped_not_as_a_permission_problem(
    tmp_path: Path,
) -> None:
    proc, _ = _run(tmp_path, DAEMON_DOWN="1")
    assert proc.returncode != 0
    assert "Docker daemon not running" in proc.stderr and "permission denied" not in proc.stderr


def test_missing_compose_plugin_is_installed(tmp_path: Path) -> None:
    proc, log = _run(tmp_path, compose=False, COMPOSE_INSTALLABLE="1")
    assert proc.returncode == 0, proc.stderr
    assert "install --no-install-recommends --no-remove docker-compose-plugin" in log


def test_missing_compose_plugin_that_cannot_be_installed_names_the_package(tmp_path: Path) -> None:
    proc, _ = _run(tmp_path, compose=False)
    assert proc.returncode != 0
    assert "docker-compose-v2" in proc.stderr


def test_an_expired_sudo_password_without_a_terminal_says_run_sudo_v(tmp_path: Path) -> None:
    proc, _ = _run(tmp_path, SUDO_NEEDS_PASSWORD="1")
    assert proc.returncode != 0
    assert "Run 'sudo -v', then rerun" in proc.stderr


def test_an_expired_sudo_password_at_a_terminal_is_asked_once_with_sudo_v(tmp_path: Path) -> None:
    proc, _ = _run(tmp_path, tty=True, SUDO_NEEDS_PASSWORD="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "sudo needs your password once" in proc.stdout
