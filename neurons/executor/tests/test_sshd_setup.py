"""The executor image's sshd drop-in (DAH-3236).

``sshd_setup.sh`` runs at image build. With an sshd that knows ``PerSourcePenalties``
(OpenSSH 9.8+) it renders ``sshd_config.d/lium.conf`` with the directive off and proves
through ``sshd -T`` that the drop-in is in effect; with an older sshd it writes nothing,
so that sshd still starts; either way it fails the build when ``sshd -T`` rejects the
rendered config, and when it rejects it for any other reason. The tests drive the script
with stand-in ``sshd`` binaries on PATH; the real OpenSSH 10.0 and 9.2 runs are in the
PR's Ran-it section.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

EXECUTOR_DIR = Path(__file__).resolve().parents[1]
SCRIPT = EXECUTOR_DIR / "sshd_setup.sh"
DOCKERFILE = EXECUTOR_DIR / "Dockerfile"

# sshd -T stand-ins. OpenSSH 10's dump line for the directive follows the drop-in when the
# main config includes sshd_config.d/*.conf; OpenSSH 9.2 rejects the keyword outright.
SSHD_MODERN = """#!/bin/sh
case " $* " in *" PerSourcePenalties="*) exit 0;; esac
echo "port 22"
if grep -qi '^PerSourcePenalties no' "$SSHD_CONFIG_DIR"/*.conf 2>/dev/null; then
    echo "persourcepenalties no"
else
    echo "persourcepenalties crash:90 authfail:5 noauth:1 grace-exceeded:20 max:600 min:15"
fi
"""
SSHD_MODERN_WITHOUT_INCLUDE = """#!/bin/sh
case " $* " in *" PerSourcePenalties="*) exit 0;; esac
echo "port 22"
echo "persourcepenalties crash:90 authfail:5 noauth:1 grace-exceeded:20 max:600 min:15"
"""
SSHD_OLD = """#!/bin/sh
case " $* " in *" PerSourcePenalties="*)
    echo "command-line line 0: Bad configuration option: PerSourcePenalties" >&2
    exit 255;;
esac
echo "port 22"
"""
SSHD_BROKEN_CONFIG = """#!/bin/sh
echo "/etc/ssh/sshd_config line 7: Bad configuration option: AcceptEnvv" >&2
exit 255
"""

pytestmark = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="sshd_setup.sh needs ssh-keygen")


def run_setup(tmp_path: Path, sshd_script: str) -> tuple[subprocess.CompletedProcess, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sshd = bin_dir / "sshd"
    sshd.write_text(sshd_script)
    sshd.chmod(0o755)
    config_dir = tmp_path / "sshd_config.d"
    config = tmp_path / "sshd_config"
    config.write_text(f"Include {config_dir}/*.conf\nPort 22\n")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SSHD_CONFIG": str(config),
        "SSHD_CONFIG_DIR": str(config_dir),
        "SSHD_PRIVSEP_DIR": str(tmp_path / "run" / "sshd"),
    }
    result = subprocess.run(["sh", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)
    return result, config_dir / "lium.conf"


def test_modern_sshd_gets_per_source_penalties_off(tmp_path):
    result, drop_in = run_setup(tmp_path, SSHD_MODERN)

    assert result.returncode == 0, result.stderr
    assert "PerSourcePenalties no" in drop_in.read_text().splitlines()
    assert "sshd_setup: PerSourcePenalties no" in result.stdout
    assert "sshd_setup: sshd -T accepts" in result.stdout


def test_old_sshd_gets_no_drop_in_and_still_passes(tmp_path):
    # a stale drop-in from an earlier layer must go too, or sshd would refuse to start on it
    (tmp_path / "sshd_config.d").mkdir()
    (tmp_path / "sshd_config.d" / "lium.conf").write_text("PerSourcePenalties no\n")

    result, drop_in = run_setup(tmp_path, SSHD_OLD)

    assert result.returncode == 0, result.stderr
    assert not drop_in.exists()
    assert "predates PerSourcePenalties" in result.stdout
    assert "sshd_setup: sshd -T accepts" in result.stdout


def test_drop_in_that_sshd_does_not_read_fails_the_build(tmp_path):
    result, drop_in = run_setup(tmp_path, SSHD_MODERN_WITHOUT_INCLUDE)

    assert result.returncode == 1
    assert drop_in.exists()
    assert "is not in effect" in result.stderr


def test_config_sshd_rejects_for_another_reason_fails_the_build(tmp_path):
    result, drop_in = run_setup(tmp_path, SSHD_BROKEN_CONFIG)

    assert result.returncode == 1
    assert not drop_in.exists()
    assert "sshd -T failed: /etc/ssh/sshd_config line 7: Bad configuration option: AcceptEnvv" in result.stderr


def test_dockerfile_pins_the_base_by_digest_and_runs_the_setup():
    lines = DOCKERFILE.read_text().splitlines()
    from_lines = [line for line in lines if line.startswith("FROM ")]

    assert len(from_lines) == 1
    assert re.fullmatch(r"FROM python:3\.11-slim@sha256:[0-9a-f]{64}", from_lines[0])
    # the setup checks the executor's own sshd_config lines too, so it runs after they are appended
    accept_env = next(i for i, line in enumerate(lines) if line.startswith("RUN echo 'AcceptEnv CONTAINER_NAME'"))
    assert lines.index("RUN sh sshd_setup.sh") > accept_env
