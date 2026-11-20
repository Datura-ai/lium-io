"""A validator miner job runs from its own fresh private directory, never importing node-local code."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from services import executor_payload


@pytest.fixture
def jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    src = tmp_path / "jobs"
    src.mkdir()
    (src / "job.py").write_text(
        "import sys, helper\nprint('helper from', helper.__file__)\nprint('args', sys.argv[1:])\n"
        "print('safe', sys.flags.ignore_environment, sys.flags.no_user_site)\n"
    )
    (src / "helper.py").write_text("")
    service = tmp_path / "service"
    service.mkdir()
    # node-local code beside where scripts used to be written, and on PYTHONPATH: neither may be imported
    (service / "helper.py").write_text("raise SystemExit('node-local helper imported')\n")
    monkeypatch.setattr(executor_payload, "MINER_JOBS", src)
    monkeypatch.setattr(executor_payload, "SERVICE_DIR_PARENT", str(service))
    monkeypatch.setattr(executor_payload, "EXECUTOR_PYTHON", sys.executable)
    return service


def _sh(command: str, service: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(service)}
    return subprocess.run(["sh", "-c", command], cwd=service, env=env, capture_output=True, text=True)


def test_a_foreground_job_imports_only_its_staged_helper_and_leaves_nothing(jobs: Path) -> None:
    done = _sh(executor_payload.run_job("backup", ("job.py", "helper.py"), ["--x", "a b"]), jobs)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].startswith("helper from " + executor_payload.service_dir_glob("backup", str(jobs)))
    assert lines[1:] == ["args ['--x', 'a b']", "safe 1 1"]
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]   # the staged directory is gone


def test_a_detached_job_logs_and_its_worker_removes_the_directory(jobs: Path, tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    done = _sh(executor_payload.run_job("restore", ("job.py", "helper.py"), ["--y"], log=str(log)), jobs)
    assert done.returncode == 0, done.stderr
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (len(list(jobs.iterdir())) > 1 or not log.exists() or "safe" not in log.read_text()):
        time.sleep(0.1)
    text = log.read_text()
    assert "helper from " + executor_payload.service_dir_glob("restore", str(jobs)) in text and "safe 1 1" in text
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]


def test_a_failed_staging_leaves_no_partial_directory(jobs: Path) -> None:
    command = executor_payload.run_job("backup", ("job.py", "helper.py"), [])
    broken = command.replace("base64 -d >", "false >", 1)   # the first write fails after mktemp
    assert _sh(broken, jobs).returncode == 1
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]
