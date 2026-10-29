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
import subprocess
from pathlib import Path


EXECUTOR_DIR = Path(__file__).resolve().parents[1]
SCRIPT = EXECUTOR_DIR / "sshd_setup.sh"

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
if grep -qi '^LogLevel VERBOSE' "$SSHD_CONFIG_DIR"/*.conf 2>/dev/null; then echo "loglevel VERBOSE"; else echo "loglevel INFO"; fi
"""
SSHD_OLD = """#!/bin/sh
case " $* " in *" PerSourcePenalties="*)
    echo "command-line line 0: Bad configuration option: PerSourcePenalties" >&2
    exit 255;;
esac
echo "port 22"
if grep -qi '^LogLevel VERBOSE' "$SSHD_CONFIG_DIR"/*.conf 2>/dev/null; then echo "loglevel VERBOSE"; else echo "loglevel INFO"; fi
"""
SSHD_WITHOUT_INCLUDE_OLD = """#!/bin/sh
case " $* " in *" PerSourcePenalties="*)
    echo "command-line line 0: Bad configuration option: PerSourcePenalties" >&2
    exit 255;;
esac
echo "port 22"
echo "loglevel INFO"
"""


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


def test_modern_sshd_logs_verbose(tmp_path):
    result, drop_in = run_setup(tmp_path, SSHD_MODERN)

    assert result.returncode == 0, result.stderr
    assert "LogLevel VERBOSE" in drop_in.read_text().splitlines()


def test_old_sshd_gets_only_verbose_logging_and_still_passes(tmp_path):
    # a stale PerSourcePenalties line from an earlier layer must go, or sshd would refuse to start on it
    (tmp_path / "sshd_config.d").mkdir()
    (tmp_path / "sshd_config.d" / "lium.conf").write_text("PerSourcePenalties no\n")

    result, drop_in = run_setup(tmp_path, SSHD_OLD)

    assert result.returncode == 0, result.stderr
    assert drop_in.read_text().splitlines()[1:] == ["LogLevel VERBOSE"]
    assert "predates PerSourcePenalties" in result.stdout
    assert "sshd_setup: sshd -T accepts" in result.stdout


def test_log_level_that_sshd_does_not_apply_fails_the_build(tmp_path):
    result, _ = run_setup(tmp_path, SSHD_WITHOUT_INCLUDE_OLD)

    assert result.returncode == 1
    assert "LogLevel VERBOSE" in result.stderr


