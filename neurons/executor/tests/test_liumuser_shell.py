"""The `liumuser` login shell / forced command (DAH-3522, bounty ticket-0329).

Regression: the forced command in the Dockerfile ran `docker exec -it "$CONTAINER_NAME" bash`
for any non-empty CONTAINER_NAME the SSH client sent, so a `liumuser` credential opened a root
shell in the executor container (host Docker socket) or in another renter's pod, and the
account's login shell was bash. The script now accepts only `pod_<uuid>` and is the login
shell; the tests drive it with a stand-in `docker` on PATH that records its argv, so a refusal
is proven by docker never being called. The Match block is checked with the real `sshd -T`
where one is installed (Debian's openssh-server, the GitHub runner, macOS).
"""

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

EXECUTOR_DIR = Path(__file__).resolve().parents[1]
SCRIPT = EXECUTOR_DIR / "liumuser_shell.sh"
SSHD_CONF = EXECUTOR_DIR / "sshd_liumuser.conf"
DOCKERFILE = EXECUTOR_DIR / "Dockerfile"

POD = "pod_014f407c-1f3a-4bbb-b652-37d2fa800f7c"

FAKE_DOCKER = """#!/bin/sh
printf '%s\\n' "$@" > "$DOCKER_ARGV_FILE"
exit 0
"""


def run_shell(
    tmp_path: Path, container_name: str | None, *shell_args: str
) -> tuple[subprocess.CompletedProcess, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    argv_file = tmp_path / "docker_argv"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "DOCKER_ARGV_FILE": str(argv_file),
        "HOME": str(tmp_path),
    }
    if container_name is not None:
        env["CONTAINER_NAME"] = container_name
    result = subprocess.run(
        ["bash", str(SCRIPT), *shell_args], env=env, capture_output=True, text=True, timeout=30
    )
    return result, argv_file


@pytest.mark.parametrize("pod", [POD, f"pod_{uuid.uuid4()}"])
def test_pod_name_opens_a_shell_in_that_pod_only(tmp_path, pod):
    # the validator names a rental container POD_CONTAINER_PREFIX ("pod_") + the pod's uuid
    result, argv_file = run_shell(tmp_path, pod)

    assert result.returncode == 0, result.stderr
    assert argv_file.read_text().splitlines() == ["exec", "-it", pod, "bash"]


def test_missing_name_is_refused_before_docker(tmp_path):
    result, argv_file = run_shell(tmp_path, None)

    assert result.returncode == 1
    assert "CONTAINER_NAME not set" in result.stderr
    assert not argv_file.exists()


@pytest.mark.parametrize(
    "name",
    [
        "executor-executor-1",  # the privileged executor container (report test C)
        "e69fd24a900b",  # a short container id
        "filler_014f407c-1f3a-4bbb-b652-37d2fa800f7c",  # validator-owned filler, not a rental
        "pod_victim111",  # a pod_ prefix without a uuid
        "pod_014f407c-1f3a-4bbb-b652-37d2fa800f7c ",  # trailing space
        "pod_014f407c-1f3a-4bbb-b652-37d2fa800f7c\npod_x",  # a second line
        "POD_014F407C-1F3A-4BBB-B652-37D2FA800F7C",  # upper case is not how the validator names pods
        "pod_014f407c-1f3a-4bbb-b652-37d2fa800f7c; id",
        "--privileged",
    ],
)
def test_anything_but_a_pod_uuid_is_refused_before_docker(tmp_path, name):
    result, argv_file = run_shell(tmp_path, name)

    assert result.returncode == 1
    assert "not a rental pod name" in result.stderr
    assert not argv_file.exists()


def test_as_a_login_shell_it_ignores_the_command_it_is_given(tmp_path):
    # sshd runs the forced command as `<login shell> -c <command>`; with the script as the
    # login shell that argument, or any other, must not be run: the pod shell is all it does.
    marker = tmp_path / "ran_the_argument"
    result, argv_file = run_shell(tmp_path, POD, "-c", f"touch {marker}")

    assert result.returncode == 0, result.stderr
    assert argv_file.read_text().splitlines() == ["exec", "-it", POD, "bash"]
    assert not marker.exists()
    argv_file.unlink()

    result, argv_file = run_shell(tmp_path, "executor-executor-1", "-c", f"touch {marker}")

    assert result.returncode == 1
    assert not argv_file.exists()
    assert not marker.exists()


def test_sshd_match_block_closes_the_paths_around_the_forced_command():
    lines = [
        line.strip()
        for line in SSHD_CONF.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines[0] == "AcceptEnv CONTAINER_NAME"
    assert lines[1] == "Match User liumuser"
    inside = set(lines[2:])
    for directive in (
        "ForceCommand /usr/local/bin/lium-pod-shell",
        "PasswordAuthentication no",
        "DisableForwarding yes",
        "AllowTcpForwarding no",
        "AllowStreamLocalForwarding no",
        "AllowAgentForwarding no",
        "X11Forwarding no",
        "PermitTunnel no",
        "PermitUserRC no",
    ):
        assert directive in inside, directive
    # one Match block, nothing after it that would fall into it by accident
    assert lines.count("Match User liumuser") == 1


def test_real_sshd_applies_the_match_block_to_liumuser_only(tmp_path):
    # `sshd -T -C user=...` prints the effective configuration for a connection by that user.
    sshd = shutil.which("sshd") or "/usr/sbin/sshd"
    if not Path(sshd).exists():
        if os.environ.get("CI"):
            pytest.fail(
                "no sshd on the CI runner: install openssh-server in the workflow, this test must run there"
            )
        pytest.skip("needs sshd (openssh-server) on this machine")
    hostkey = tmp_path / "hostkey"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(hostkey)], check=True)
    config = tmp_path / "sshd_config"
    config.write_text("Port 22\n" + SSHD_CONF.read_text())

    def effective(user: str) -> dict[str, str]:
        spec = f"user={user},host=executor,addr=203.0.113.9,laddr=203.0.113.10,lport=22"
        out = subprocess.run(
            [sshd, "-T", "-f", str(config), "-h", str(hostkey), "-C", spec],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert out.returncode == 0, out.stderr
        return dict(line.split(" ", 1) for line in out.stdout.splitlines() if " " in line)

    liumuser = effective("liumuser")
    assert liumuser["forcecommand"] == "/usr/local/bin/lium-pod-shell"
    assert liumuser["acceptenv"] == "CONTAINER_NAME"
    for directive in (
        "passwordauthentication",
        "kbdinteractiveauthentication",
        "permitemptypasswords",
        "allowtcpforwarding",
        "allowstreamlocalforwarding",
        "allowagentforwarding",
        "x11forwarding",
        "permittunnel",
        "permituserrc",
    ):
        assert liumuser[directive] == "no", directive
    assert liumuser["disableforwarding"] == "yes"
    assert liumuser["permittty"] == "yes"

    root = effective("root")
    assert root["forcecommand"] == "none"
    assert root["allowtcpforwarding"] == "yes"


def test_dockerfile_installs_the_script_as_the_locked_accounts_login_shell():
    text = DOCKERFILE.read_text()
    assert "install -m 0755 liumuser_shell.sh /usr/local/bin/lium-pod-shell" in text
    assert "useradd -m -s /usr/local/bin/lium-pod-shell liumuser" in text
    assert "passwd -l liumuser" in text
    assert "cat sshd_liumuser.conf >>/etc/ssh/sshd_config" in text
    assert "useradd -m -s /bin/bash liumuser" not in text
    assert "ForceCommand bash -c" not in text
    # no sudo in the image at all: the account's only privilege is the Docker socket group run.sh adds
    instructions = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert not any("sudo" in line for line in instructions)
