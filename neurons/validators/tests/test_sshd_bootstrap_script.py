"""DAH-2341 — behavioral tests for sshd_bootstrap.sh's decision flow.

The script is exercised as a real subprocess with a shim PATH (pgrep,
ssh-keygen, sleep, nohup, apt-get) and LIUM_* path overrides, so every branch
of the race-safe flow runs for real without root or a container:

- adopt an already-running sshd (no keygen),
- adopt an sshd that appears during the grace window,
- fall back to own bring-up when the image never starts sshd,
- skip the grace wait entirely when the image ships no sshd binary,
- tolerate losing the sshd start race (bind conflict),
- reload the hardened config when the fallback path adopts an sshd,
- fail (exit 1) only when sshd is genuinely not serving,
- require a :22 listener for success, not just a process,
- proceed after the shared setup-lock times out.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent / "src" / "services" / "assets" / "sshd_bootstrap.sh"
)

# One LISTEN row on 0.0.0.0:22 (0016 hex), plus noise that must not match:
# an ESTABLISHED (01) row on :22 and a LISTEN row on another port.
PROC_TCP_WITH_LISTENER = (
    "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid\n"
    "   0: 00000000:0016 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0\n"
    "   1: 0100007F:0016 0100007F:D2A0 01 00000000:00000000 00:00000000 00000000     0\n"
)
PROC_TCP_WITHOUT_LISTENER = (
    "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid\n"
    "   1: 0100007F:0016 0100007F:D2A0 01 00000000:00000000 00:00000000 00000000     0\n"
    "   2: 00000000:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0\n"
)

PGREP_SHIM = """#!/bin/sh
# sshd is "running" when the state marker exists. An optional countdown file
# makes it appear after N pgrep calls, simulating an image entrypoint that
# starts sshd mid-grace.
if [ -f "$SHIM_STATE/sshd_running" ]; then
    exit 0
fi
if [ -f "$SHIM_STATE/pgrep_countdown" ]; then
    n=$(cat "$SHIM_STATE/pgrep_countdown")
    n=$((n - 1))
    printf '%s\\n' "$n" > "$SHIM_STATE/pgrep_countdown"
    if [ "$n" -le 0 ]; then
        touch "$SHIM_STATE/sshd_running"
        exit 0
    fi
fi
exit 1
"""

SLEEP_SHIM = """#!/bin/sh
# Keep the script's second-granularity loops honest but fast.
exec /bin/sleep 0.01
"""

# is_sshd_running falls through to `ps` when pgrep reports nothing — without
# this shim the host machine's real sshd would leak into every scenario.
PS_SHIM = """#!/bin/sh
if [ -f "$SHIM_STATE/sshd_running" ]; then
    echo "root         1     0  0 00:00 ?        00:00:00 sshd"
fi
exit 0
"""

SSH_KEYGEN_SHIM = """#!/bin/sh
printf '%s\\n' "$*" >> "$SHIM_STATE/keygen_calls"
exit 0
"""

NOHUP_SHIM = """#!/bin/sh
# Record the watchdog spawn without actually starting the loop.
printf '%s\\n' "$*" >> "$SHIM_STATE/nohup_calls"
exit 0
"""

APT_GET_SHIM = """#!/bin/sh
printf '%s\\n' "$*" >> "$SHIM_STATE/apt_calls"
case "$*" in
    *install*)
        cp "$SHIM_STATE/sshd_payload" "$LIUM_SSHD_BIN"
        chmod +x "$LIUM_SSHD_BIN"
        ;;
esac
exit 0
"""

SSHD_STARTS = """#!/bin/sh
touch "$SHIM_STATE/sshd_running"
printf 'started\\n' >> "$SHIM_STATE/sshd_started"
exit 0
"""

# Exits non-zero as if the bind failed, while the concurrent (image) sshd is
# in fact up — the script must treat this as success after re-checking.
SSHD_LOSES_BIND_RACE = """#!/bin/sh
touch "$SHIM_STATE/sshd_running"
printf 'failed\\n' >> "$SHIM_STATE/sshd_started"
exit 1
"""

SSHD_NEVER_SERVES = """#!/bin/sh
printf 'failed\\n' >> "$SHIM_STATE/sshd_started"
exit 1
"""


class BootstrapHarness:
    def __init__(self, tmp_path: Path):
        self.root = tmp_path
        self.shims = tmp_path / "shims"
        self.state = tmp_path / "state"
        self.run_dir = tmp_path / "run"
        self.ssh_dir = tmp_path / "ssh"
        self.shims.mkdir()
        self.state.mkdir()
        self.run_dir.mkdir()

        self.sshd_config = tmp_path / "sshd_config"
        self.sshd_config.write_text("PasswordAuthentication yes\n")

        self.proc_tcp = tmp_path / "proc_tcp"
        self.proc_tcp.write_text(PROC_TCP_WITH_LISTENER)

        self.sshd_bin = self.state / "sshd"

        for name, body in (
            ("pgrep", PGREP_SHIM),
            ("ps", PS_SHIM),
            ("sleep", SLEEP_SHIM),
            ("ssh-keygen", SSH_KEYGEN_SHIM),
            ("nohup", NOHUP_SHIM),
            ("apt-get", APT_GET_SHIM),
        ):
            self._write_executable(self.shims / name, body)

    def _write_executable(self, path: Path, body: str) -> None:
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def install_sshd_bin(self, body: str) -> None:
        self._write_executable(self.sshd_bin, body)

    def stage_sshd_payload(self, body: str) -> None:
        """Payload `apt-get install` copies onto LIUM_SSHD_BIN."""
        self._write_executable(self.state / "sshd_payload", body)

    def mark_sshd_running(self) -> None:
        (self.state / "sshd_running").touch()

    def set_pgrep_countdown(self, calls: int) -> None:
        (self.state / "pgrep_countdown").write_text(str(calls))

    def keygen_calls(self) -> list[str]:
        path = self.state / "keygen_calls"
        return path.read_text().splitlines() if path.exists() else []

    def apt_calls(self) -> list[str]:
        path = self.state / "apt_calls"
        return path.read_text().splitlines() if path.exists() else []

    def sshd_start_attempts(self) -> list[str]:
        path = self.state / "sshd_started"
        return path.read_text().splitlines() if path.exists() else []

    def watchdog_spawned(self) -> bool:
        return (self.state / "nohup_calls").exists()

    def run(self, *args: str, verify_secs: int = 2) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "PATH": f"{self.shims}:{os.environ['PATH']}",
            "SHIM_STATE": str(self.state),
            "LIUM_RUN_DIR": str(self.run_dir),
            "LIUM_SSH_DIR": str(self.ssh_dir),
            "LIUM_SSHD_CONFIG": str(self.sshd_config),
            "LIUM_SSHD_BIN": str(self.sshd_bin),
            "LIUM_PROC_TCP_FILES": str(self.proc_tcp),
            "LIUM_SSHD_VERIFY_SECS": str(verify_secs),
            "LIUM_SSH_LOCK_TIMEOUT_SECS": "1",
        }
        return subprocess.run(
            ["sh", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )


@pytest.fixture
def harness(tmp_path):
    return BootstrapHarness(tmp_path)


def test_adopts_already_running_sshd_without_touching_host_keys(harness):
    harness.mark_sshd_running()
    harness.install_sshd_bin(SSHD_STARTS)

    result = harness.run("--grace", "5")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "adopting existing daemon" in result.stdout
    assert harness.keygen_calls() == []
    assert harness.sshd_start_attempts() == []
    assert harness.watchdog_spawned()


def test_adopt_path_hardens_config_for_key_only_auth(harness):
    harness.mark_sshd_running()

    result = harness.run()

    assert result.returncode == 0, result.stdout + result.stderr
    hardened = harness.sshd_config.read_text()
    assert "# lium-hardened" in hardened
    assert "# lium-disabled PasswordAuthentication yes" in hardened
    assert "\nPasswordAuthentication no\n" in hardened
    # No sshd pidfile exists in the harness, so the reload is skipped loudly.
    assert "no sshd pidfile to reload" in result.stdout


def test_adopts_sshd_that_appears_during_grace(harness):
    harness.set_pgrep_countdown(3)
    harness.install_sshd_bin(SSHD_STARTS)

    result = harness.run("--grace", "10")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Waiting up to 10s for image-provided sshd" in result.stdout
    assert "Image-provided sshd came up" in result.stdout
    assert harness.keygen_calls() == []
    assert harness.sshd_start_attempts() == []


def test_fallback_owns_bringup_when_image_never_starts_sshd(harness):
    harness.install_sshd_bin(SSHD_STARTS)

    result = harness.run("--grace", "1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "falling back to own bring-up" in result.stdout
    assert harness.keygen_calls() == ["-A"]
    assert harness.sshd_start_attempts() == ["started"]
    assert harness.watchdog_spawned()


def test_no_grace_wait_when_image_ships_no_sshd_binary(harness):
    # LIUM_SSHD_BIN does not exist yet; the apt-get shim "installs" it.
    harness.stage_sshd_payload(SSHD_STARTS)

    result = harness.run()

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Waiting up to" not in result.stdout
    assert any("install" in call for call in harness.apt_calls())
    assert harness.keygen_calls() == ["-A"]
    assert harness.sshd_start_attempts() == ["started"]


def test_grace_zero_skips_waiting_even_with_sshd_binary(harness):
    harness.install_sshd_bin(SSHD_STARTS)

    result = harness.run("--grace", "0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Waiting up to" not in result.stdout
    assert harness.keygen_calls() == ["-A"]


def test_tolerates_losing_the_sshd_start_race(harness):
    harness.install_sshd_bin(SSHD_LOSES_BIND_RACE)

    result = harness.run("--grace", "1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "sshd was started concurrently" in result.stdout


def test_fallback_adopt_after_bind_race_reloads_hardened_config(harness):
    """The adopted (image-started) sshd loaded the pre-hardening config, so the
    fallback path must SIGHUP the master after hardening — not just converge."""
    harness.install_sshd_bin(SSHD_LOSES_BIND_RACE)
    master = subprocess.Popen(["/bin/sleep", "60"])
    (harness.run_dir / "sshd.pid").write_text(f"{master.pid}\n")

    try:
        result = harness.run("--grace", "1")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "sshd was started concurrently" in result.stdout
        assert f"Sent SIGHUP to sshd (pid {master.pid})" in result.stdout
        # SIGHUP's default disposition terminates sleep — proof it was delivered.
        assert master.wait(timeout=5) == -signal.SIGHUP
    finally:
        if master.poll() is None:
            master.kill()
            master.wait()


def test_fails_when_sshd_never_serves(harness):
    harness.install_sshd_bin(SSHD_NEVER_SERVES)

    result = harness.run("--grace", "1")

    assert result.returncode != 0
    assert "sshd verification failed" in result.stdout
    assert "running=no" in result.stdout


def test_verify_requires_listener_on_port_22_not_just_a_process(harness):
    harness.mark_sshd_running()
    harness.proc_tcp.write_text(PROC_TCP_WITHOUT_LISTENER)

    result = harness.run(verify_secs=1)

    assert result.returncode != 0
    assert "sshd verification failed" in result.stdout
    assert "running=yes" in result.stdout
    assert "listening=no" in result.stdout


def test_lock_timeout_proceeds_instead_of_hanging(harness):
    harness.install_sshd_bin(SSHD_STARTS)
    # Simulate a crashed holder: the lock dir exists and nobody releases it.
    (harness.run_dir / "lium-ssh-setup.lock").mkdir()

    result = harness.run("--grace", "0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "proceeding without it" in result.stdout
    assert harness.sshd_start_attempts() == ["started"]
