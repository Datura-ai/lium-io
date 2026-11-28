"""A validator miner job runs from its own fresh private directory, never importing node-local code."""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from services import executor_payload, storage_operations

SECRET = '{"auth_token":"tok"}'


@pytest.fixture
def jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    src = tmp_path / "jobs"
    src.mkdir()
    (src / "job.py").write_text(
        "import sys, helper\nprint('helper from', helper.__file__)\nprint('args', sys.argv[1:])\n"
        "print('safe', sys.flags.ignore_environment, sys.flags.no_user_site)\n"
        "print('stdin', sys.stdin.readline().strip())\n"
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


def _sh(command: str, service: Path, input: str | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(service)}
    return subprocess.run(["sh", "-c", command], cwd=service, env=env, capture_output=True, text=True, input=input)


def _stage_and_run(purpose: str, args: list[str], service: Path, log: str | None = None) -> subprocess.CompletedProcess:
    files = ("job.py", "helper.py")
    staged = _sh(executor_payload.stage_job(purpose, files), service)
    assert staged.returncode == 0, staged.stderr
    return _sh(executor_payload.run_job(executor_payload.staged_directory(purpose, staged.stdout), files, args, log=log), service, SECRET + "\n")


def test_a_foreground_job_imports_only_its_staged_helper_and_leaves_nothing(jobs: Path) -> None:
    done = _stage_and_run("backup", ["--x", "a b"], jobs)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].startswith("helper from " + executor_payload.service_dir_glob("backup", str(jobs)))
    assert lines[1:] == ["args ['--x', 'a b']", "safe 1 1", f"stdin {SECRET}"]
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]   # the staged directory is gone


def test_a_detached_job_logs_and_its_worker_removes_the_directory(jobs: Path, tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    done = _stage_and_run("restore", ["--y"], jobs, log=str(log))
    assert done.returncode == 0, done.stderr
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (len(list(jobs.iterdir())) > 1 or not log.exists() or "safe" not in log.read_text()):
        time.sleep(0.1)
    text = log.read_text()
    assert "helper from " + executor_payload.service_dir_glob("restore", str(jobs)) in text and "safe 1 1" in text
    assert f"stdin {SECRET}" in text   # the credentials reached the detached job through the pipe, not a file
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]


def test_a_failed_staging_leaves_no_partial_directory(jobs: Path) -> None:
    command = executor_payload.stage_job("backup", ("job.py", "helper.py"))
    broken = command.replace("base64 -d >", "false >", 1)   # the first write fails after mktemp
    assert _sh(broken, jobs).returncode == 1
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]


def _volume() -> SimpleNamespace:
    return SimpleNamespace(iam_user_access_key="ak-value", iam_user_secret_key="sk-value")


def _no_secret_on_disk(root: Path) -> bool:
    return not any(b"sk-value" in p.read_bytes() for p in root.rglob("*") if p.is_file())


def _runner(jobs: Path, starts: list[str], fails: str = ""):
    async def run(command: str, check: bool = False, input: str | None = None, **_) -> SimpleNamespace:
        if input is not None:
            starts.append(command)
            if fails == "cancel":
                raise asyncio.CancelledError
            if fails == "start":
                raise RuntimeError("the start failed")
        if fails == "staging" and "mktemp" in command:
            command = command.replace("base64 -d >", "false >", 1)   # the node fails a write after mktemp
        done = _sh(command, jobs, input)
        if check and done.returncode:
            raise RuntimeError(f"exit {done.returncode}")
        return SimpleNamespace(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)

    return run


def test_a_detached_job_gets_its_credentials_on_stdin_and_none_touch_the_disk(jobs: Path, tmp_path: Path) -> None:
    starts: list[str] = []
    log = tmp_path / "job.log"
    asyncio.run(storage_operations.run_job_with_secrets(
        SimpleNamespace(run=_runner(jobs, starts)), "backup", ("job.py", "helper.py"), "token", _volume(), ["--z"],
        log=str(log),
    ))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (
        len(list(jobs.iterdir())) > 1 or "stdin" not in (log.read_text() if log.exists() else "")
    ):
        time.sleep(0.1)
    assert '"aws_secret_access_key": "sk-value"' in log.read_text()   # the job printed what it read on stdin
    assert len(starts) == 1 and "sk-value" not in starts[0]
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]
    assert _no_secret_on_disk(jobs)


@pytest.mark.parametrize("fails", ["staging", "start", "log", "cancel"])
def test_a_job_that_did_not_start_leaves_no_directory_and_no_credentials(
    jobs: Path, tmp_path: Path, fails: str
) -> None:
    starts: list[str] = []
    log = tmp_path / "job.log"
    if fails == "log":
        log.mkdir()   # the log is opened before detaching; a job that cannot open it must not leave its directory
    with pytest.raises((RuntimeError, asyncio.CancelledError)):
        asyncio.run(storage_operations.run_job_with_secrets(
            SimpleNamespace(run=_runner(jobs, starts, fails)), "backup", ("job.py", "helper.py"), "token", _volume(),
            ["--z"], log=str(log),
        ))
    assert len(starts) == (0 if fails == "staging" else 1)
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]
    assert _no_secret_on_disk(jobs)
