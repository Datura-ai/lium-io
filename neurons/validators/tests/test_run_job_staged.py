"""A validator miner job runs from its own fresh private directory, never importing node-local code."""

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from services import executor_payload, storage_operations


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


def _stage_and_run(purpose: str, args: list[str], service: Path, log: str | None = None) -> subprocess.CompletedProcess:
    files = ("job.py", "helper.py")
    staged = _sh(executor_payload.stage_job(purpose, files), service)
    assert staged.returncode == 0, staged.stderr
    return _sh(executor_payload.run_job(executor_payload.staged_directory(purpose, staged.stdout), files, args, log=log), service)


def test_a_foreground_job_imports_only_its_staged_helper_and_leaves_nothing(jobs: Path) -> None:
    done = _stage_and_run("backup", ["--x", "a b"], jobs)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0].startswith("helper from " + executor_payload.service_dir_glob("backup", str(jobs)))
    assert lines[1:] == ["args ['--x', 'a b']", "safe 1 1"]
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
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]


def test_a_failed_staging_leaves_no_partial_directory(jobs: Path) -> None:
    command = executor_payload.stage_job("backup", ("job.py", "helper.py"))
    broken = command.replace("base64 -d >", "false >", 1)   # the first write fails after mktemp
    assert _sh(broken, jobs).returncode == 1
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]


@pytest.mark.parametrize("fails", ["staging", "start"])
def test_the_credentials_file_never_outlives_a_job_that_did_not_start(
    jobs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: str
) -> None:
    # the job reads and deletes its credentials file: the file is uploaded only once the job is staged, and a failed
    # start removes it with the staged directory
    import asyncio

    uploaded: list[str] = []

    async def upload(ssh_client, auth_token, volume_info, directory):
        # into the job's own staged directory (the real upload_secrets_file's place)
        path = Path(directory) / "secrets.json"
        path.write_text("{}")
        uploaded.append(str(path))
        return str(path)

    monkeypatch.setattr(storage_operations, "upload_secrets_file", upload)
    async def run(command: str, check: bool = False, **_) -> SimpleNamespace:
        if fails == "start" and "nohup" in command:
            raise RuntimeError("the start failed")
        if fails == "staging" and "mktemp" in command:
            command = command.replace("base64 -d >", "false >", 1)   # the node fails a write after mktemp
        done = _sh(command, jobs)
        if check and done.returncode:
            raise RuntimeError(f"exit {done.returncode}")
        return SimpleNamespace(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)

    with pytest.raises(RuntimeError):
        asyncio.run(storage_operations.run_job_with_secrets(
            SimpleNamespace(run=run), "backup", ("job.py", "helper.py"), "token", None,
            lambda secrets_file: ["--secrets-file", secrets_file], log=str(tmp_path / "job.log"),
        ))
    assert len(uploaded) == (0 if fails == "staging" else 1)
    assert not any(Path(path).exists() for path in uploaded)
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]


def test_the_credentials_file_never_outlives_a_detached_job_whose_log_cannot_be_opened(
    jobs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the log is opened before detaching: a job that could not open it would never start the shell removing the directory
    import asyncio

    uploaded: list[str] = []

    async def upload(ssh_client, auth_token, volume_info, directory):
        path = Path(directory) / "secrets.json"
        path.write_text("{}")
        uploaded.append(str(path))
        return str(path)

    monkeypatch.setattr(storage_operations, "upload_secrets_file", upload)

    async def run(command: str, check: bool = False, **_) -> SimpleNamespace:
        done = _sh(command, jobs)
        if check and done.returncode:
            raise RuntimeError(f"exit {done.returncode}")
        return SimpleNamespace(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)

    log = tmp_path / "job.log"
    log.mkdir()
    with pytest.raises(RuntimeError):
        asyncio.run(storage_operations.run_job_with_secrets(
            SimpleNamespace(run=run), "backup", ("job.py", "helper.py"), "token", None,
            lambda secrets_file: ["--secrets-file", secrets_file], log=str(log),
        ))
    assert len(uploaded) == 1 and not Path(uploaded[0]).exists()
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]


def test_a_job_cancelled_after_its_credentials_were_written_leaves_none_behind(
    jobs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    uploaded: list[str] = []
    started: list[str] = []

    async def upload(ssh_client, auth_token, volume_info, directory):
        # written whole, then cancelled while the upload closes, before the job starts
        path = Path(directory) / "secrets.json"
        path.write_text("{}")
        uploaded.append(str(path))
        raise asyncio.CancelledError

    monkeypatch.setattr(storage_operations, "upload_secrets_file", upload)

    async def run(command: str, check: bool = False, **_) -> SimpleNamespace:
        if "nohup" in command:
            started.append(command)
        done = _sh(command, jobs)
        return SimpleNamespace(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(storage_operations.run_job_with_secrets(
            SimpleNamespace(run=run), "backup", ("job.py", "helper.py"), "token", None,
            lambda secrets_file: ["--secrets-file", secrets_file], log=str(tmp_path / "job.log"),
        ))
    assert len(uploaded) == 1 and not Path(uploaded[0]).exists() and started == []
    assert sorted(p.name for p in jobs.iterdir()) == ["helper.py"]
