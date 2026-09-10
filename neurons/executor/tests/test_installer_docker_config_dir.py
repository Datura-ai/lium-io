"""DAH-3376: scripts/install_executor_on_ubuntu.sh creates ``$HOME/.docker`` on every run.

The stack bind-mounts ``$HOME/.docker`` (docker-compose.yml); when the directory is missing the
daemon creates the bind source as root and the operator's later ``docker login`` cannot write
``config.json``. The installer therefore creates it as the user running the script — the user
``lium mine`` starts the stack as — including on a host where Docker is already installed and
``install_docker`` returns early.

Same harness as test_sysbox_setup_preflight.py: the script runs under bash with stub commands
(apt-get, sudo, docker, id) ahead of the real PATH and a fixture ``$HOME``.
"""

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "install_executor_on_ubuntu.sh"

STUBS = {
    "apt-get": "#!/bin/bash\nexit 0\n",
    # `sudo -n <cmd>`: run <cmd> as-is
    "sudo": '#!/bin/bash\n[ "$1" = "-n" ] && shift\nexec "$@"\n',
    "docker": textwrap.dedent(
        """\
        #!/bin/bash
        case "$1" in
            --version) echo "Docker version 28.5.2, build stub" ;;
            info|compose) exit 0 ;;
            *) echo "unexpected docker $*" >&2; exit 1 ;;
        esac
        """
    ),
    # the operator is already in the docker group: no usermod, no `sg` re-exec at the end
    "id": '#!/bin/bash\n[ "$1" = "-nG" ] && { echo "docker"; exit 0; }\nexec /usr/bin/id "$@"\n',
}


@pytest.fixture
def host(tmp_path):
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in STUBS.items():
        p = stubs / name
        p.write_text(body)
        p.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    return stubs, home


def _run_installer(stubs: Path, home: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "HOME": str(home),
        "USER": os.environ.get("USER", "op"),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60
    )


def test_installer_creates_docker_config_dir_when_docker_is_already_installed(host):
    stubs, home = host
    assert not (home / ".docker").exists()

    result = _run_installer(stubs, home)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Docker already installed" in result.stdout  # install_docker returned early
    assert (home / ".docker").is_dir()
    assert f"Docker config directory ready: {home}/.docker" in result.stdout


def test_installer_leaves_an_existing_docker_config_dir_alone(host):
    stubs, home = host
    (home / ".docker").mkdir()
    config = home / ".docker" / "config.json"
    config.write_text('{"auths": {}}\n')

    result = _run_installer(stubs, home)

    assert result.returncode == 0, result.stdout + result.stderr
    assert config.read_text() == '{"auths": {}}\n'
