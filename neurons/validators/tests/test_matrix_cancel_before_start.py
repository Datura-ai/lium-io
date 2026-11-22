"""Cancelling a matmul while its script is still being installed must stop it from ever starting, and only that tag."""

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from services import executor_payload


@pytest.fixture
def node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    payloads = tmp_path / "payloads"
    payloads.mkdir()
    # the stand-in matmul records that it started
    (payloads / "decrypt_challenge.py").write_text("import sys, pathlib\npathlib.Path(sys.argv[1]).write_text('started')\n")
    service = tmp_path / "service"
    service.mkdir()
    monkeypatch.setattr(executor_payload, "PAYLOAD_DIR", payloads)
    monkeypatch.setattr(executor_payload, "SERVICE_DIR_PARENT", str(service))
    monkeypatch.setattr(executor_payload, "EXECUTOR_PYTHON", sys.executable)
    return tmp_path


def _launch(node: Path, tag: str, paused: str = "before_start") -> subprocess.Popen:
    command = executor_payload.run_once("matrix", executor_payload.MATRIX, str(node / f"{tag}.started"), tag=tag)
    run = executor_payload.run(executor_payload.MATRIX)
    assert f"; {run} " in command and command.count("|| exit 1; { ") == 1
    if paused == "before_start":   # installed, interpreter not started yet
        slowed = command.replace(f"; {run} ", f"; sleep 1.5; {run} ", 1)
    else:   # past the cancel-marker check, its directory and script not written yet
        slowed = command.replace("|| exit 1; { ", "|| exit 1; sleep 1.5; { ", 1)
    # /bin/sh as an SSH exec runs it (dash on Debian/Ubuntu nodes), in its own session
    return subprocess.Popen(["/bin/sh", "-c", slowed], start_new_session=True)


def test_a_cancel_between_install_and_start_stops_that_run_and_no_other(node: Path) -> None:
    mine, other = executor_payload.new_run_tag(), executor_payload.new_run_tag()
    a, b = _launch(node, mine), _launch(node, other)
    try:
        time.sleep(0.5)   # both installed, neither interpreter started
        subprocess.run(["sh", "-c", executor_payload.kill_runs("matrix", executor_payload.MATRIX, mine)], check=True)
        a.wait(timeout=5)
        b.wait(timeout=10)
    finally:
        for p in (a, b):
            if p.poll() is None:
                p.kill()
    time.sleep(0.3)
    assert not (node / f"{mine}.started").exists()   # cancelled before it started: never started
    assert (node / f"{other}.started").read_text() == "started" and b.returncode == 0
    # the cancelled run's directory is gone, the other cleaned itself; only the cancel's marker stays
    assert [p.name for p in (node / "service").iterdir()] == [Path(executor_payload.cancel_marker("matrix", mine)).name]


def test_a_cancel_that_lands_before_the_launch_shell_exists_still_stops_that_run(node: Path) -> None:
    # the cancel's SSH command can finish before the launch's exec has started its shell: nothing to kill yet
    mine, other = executor_payload.new_run_tag(), executor_payload.new_run_tag()
    subprocess.run(["sh", "-c", executor_payload.kill_runs("matrix", executor_payload.MATRIX, mine)], check=True)
    a, b = _launch(node, mine), _launch(node, other)
    try:
        a.wait(timeout=5)
        b.wait(timeout=10)
    finally:
        for p in (a, b):
            if p.poll() is None:
                p.kill()
    assert not (node / f"{mine}.started").exists() and a.returncode == 1
    assert (node / f"{other}.started").read_text() == "started" and b.returncode == 0
    assert [p.name for p in (node / "service").iterdir()] == [Path(executor_payload.cancel_marker("matrix", mine)).name]


def test_a_cancel_of_a_launcher_paused_before_its_install_kills_it_and_no_other(node: Path) -> None:
    # its directory does not exist yet, so only killing the launcher's process group stops it (dash's kill included)
    mine, other = executor_payload.new_run_tag(), executor_payload.new_run_tag()
    a, b = _launch(node, mine, paused="before_install"), _launch(node, other, paused="before_install")
    try:
        time.sleep(0.5)   # both past the marker check, neither installed
        subprocess.run(["/bin/sh", "-c", executor_payload.kill_runs("matrix", executor_payload.MATRIX, mine)], check=True)
        a.wait(timeout=5)
        b.wait(timeout=10)
    finally:
        for p in (a, b):
            if p.poll() is None:
                p.kill()
    time.sleep(2)   # a surviving launcher would have installed and started by now
    assert a.returncode == -signal.SIGKILL
    assert not (node / f"{mine}.started").exists()
    assert (node / f"{other}.started").read_text() == "started" and b.returncode == 0


def test_a_cancel_that_cannot_leave_its_marker_still_kills_and_says_it_is_incomplete(node: Path) -> None:
    # without the marker a launcher not up yet may still start: the kills run, and the cancel exits 1 so no retry follows
    mine = executor_payload.new_run_tag()
    # the marker cannot be written (its target's directory is missing), yet the launcher's check finds no marker
    Path(executor_payload.cancel_marker("matrix", mine)).symlink_to(node / "missing" / "marker")
    a = _launch(node, mine)
    try:
        time.sleep(0.5)
        done = subprocess.run(["sh", "-c", executor_payload.kill_runs("matrix", executor_payload.MATRIX, mine)])
        a.wait(timeout=5)
    finally:
        if a.poll() is None:
            a.kill()
    assert done.returncode == 1
    assert a.returncode != 0 and not (node / f"{mine}.started").exists()
