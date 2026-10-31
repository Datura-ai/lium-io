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
import socket as socketlib
import subprocess
from pathlib import Path


EXECUTOR_DIR = Path(__file__).resolve().parents[1]
SCRIPT = EXECUTOR_DIR / "sshd_setup.sh"

# sshd -T stand-ins. Like OpenSSH, they keep the first value a keyword gets across the drop-ins
# (read in name order); OpenSSH 9.2 rejects PerSourcePenalties outright.
FIRST = """first() { cat "$SSHD_CONFIG_DIR"/*.conf 2>/dev/null | grep -i -m1 "^$1 " | cut -d' ' -f2; }
echo "port 22"
echo "loglevel $(first LogLevel || true)"
echo "logingracetime $(first LoginGraceTime || true)"
echo "maxstartups $(first MaxStartups || true)"
"""
SSHD_MODERN = """#!/bin/sh
case " $* " in *" PerSourcePenalties="*) exit 0;; esac
""" + FIRST + """echo "persourcemaxstartups $(first PerSourceMaxStartups || true)"
if [ "$(first PerSourcePenalties)" = no ]; then
    echo "persourcepenalties no"
else
    echo "persourcepenalties crash:90 authfail:5 noauth:1 grace-exceeded:20 max:600 min:15"
fi
"""
SSHD_OLD = """#!/bin/sh
case " $* " in *" PerSourcePenalties="*)
    echo "command-line line 0: Bad configuration option: PerSourcePenalties" >&2
    exit 255;;
esac
""" + FIRST
SSHD_WITHOUT_INCLUDE_OLD = """#!/bin/sh
case " $* " in *" PerSourcePenalties="*)
    echo "command-line line 0: Bad configuration option: PerSourcePenalties" >&2
    exit 255;;
esac
echo "port 22"
echo "loglevel INFO"
"""


def run_setup(tmp_path: Path, sshd_script: str, *args: str, **extra_env: str) -> tuple[subprocess.CompletedProcess, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
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
        "SSHD_SOURCE_OFF": str(tmp_path / "lium-source-limits-off.conf"),
        "DSTACK_SOCKET_PATH": str(tmp_path / "no-dstack.sock"),
        **extra_env,
    }
    result = subprocess.run(["sh", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=30)
    return result, config_dir / "lium.conf"


def test_modern_sshd_gets_per_source_limits_and_short_grace(tmp_path):
    result, drop_in = run_setup(tmp_path, SSHD_MODERN)

    assert result.returncode == 0, result.stderr
    lines = drop_in.read_text().splitlines()
    for line in ["LogLevel VERBOSE", "LoginGraceTime 20", "MaxStartups 60:30:200",
                 "PerSourceMaxStartups 10", "PerSourcePenalties yes"]:
        assert line in lines
    # the build proved the off drop-in wins over lium.conf, then took it out again
    assert "PerSourcePenalties no" in (tmp_path / "lium-source-limits-off.conf").read_text()
    assert not (tmp_path / "sshd_config.d" / "00-lium-source-limits-off.conf").exists()


def test_old_sshd_gets_no_per_source_lines_and_still_passes(tmp_path):
    # a stale PerSourcePenalties line from an earlier layer must go, or sshd would refuse to start on it
    (tmp_path / "sshd_config.d").mkdir()
    (tmp_path / "sshd_config.d" / "lium.conf").write_text("PerSourcePenalties no\n")

    result, drop_in = run_setup(tmp_path, SSHD_OLD)

    assert result.returncode == 0, result.stderr
    assert drop_in.read_text().splitlines()[1:] == ["LogLevel VERBOSE", "LoginGraceTime 20", "MaxStartups 60:30:200"]
    assert not (tmp_path / "lium-source-limits-off.conf").exists()
    assert "predates PerSourcePenalties" in result.stdout
    assert "sshd_setup: sshd -T accepts" in result.stdout


def test_log_level_that_sshd_does_not_apply_fails_the_build(tmp_path):
    result, _ = run_setup(tmp_path, SSHD_WITHOUT_INCLUDE_OLD)

    assert result.returncode == 1
    assert "loglevel VERBOSE" in result.stderr


def _start(tmp_path, **env):
    result, _ = run_setup(tmp_path, SSHD_MODERN, "start", **env)
    assert result.returncode == 0, result.stderr
    return (tmp_path / "sshd_config.d" / "00-lium-source-limits-off.conf").exists()


def test_start_keeps_limits_on_by_default_and_drops_a_stale_off_file(tmp_path):
    assert run_setup(tmp_path, SSHD_MODERN)[0].returncode == 0
    (tmp_path / "sshd_config.d" / "00-lium-source-limits-off.conf").write_text("PerSourcePenalties no\n")

    assert not _start(tmp_path)


def test_start_turns_limits_off_on_request(tmp_path):
    assert run_setup(tmp_path, SSHD_MODERN)[0].returncode == 0

    assert _start(tmp_path, EXECUTOR_SSH_SOURCE_LIMITS="False")


def test_start_turns_limits_off_in_a_cvm(tmp_path, monkeypatch):
    assert run_setup(tmp_path, SSHD_MODERN)[0].returncode == 0
    monkeypatch.chdir(tmp_path)  # a unix socket path must stay short
    with socketlib.socket(socketlib.AF_UNIX) as dstack:
        dstack.bind("dstack.sock")
        assert _start(tmp_path, DSTACK_SOCKET_PATH="dstack.sock")


def test_start_on_an_old_sshd_writes_nothing(tmp_path):
    assert run_setup(tmp_path, SSHD_OLD)[0].returncode == 0

    assert not _start(tmp_path, EXECUTOR_SSH_SOURCE_LIMITS="false")
