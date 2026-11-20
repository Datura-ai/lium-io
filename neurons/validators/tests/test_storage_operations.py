import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from services import executor_payload, storage_operations

OPERATION_ID = UUID("11111111-1111-4111-8111-111111111111")


@pytest.mark.asyncio
@pytest.mark.parametrize("runner_alive", [True, False])
async def test_cancel_fallback_removes_the_helper_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner_alive: bool
) -> None:
    # Runs the real cancel shell with a fake docker that records its calls.
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(f'#!/bin/sh\necho "$*" >> {docker_log}\n')
    fake_docker.chmod(0o755)
    runner_pid = None
    if runner_alive:
        # ignores the cancel file like an unresponsive runner; detached so no zombie keeps `kill -0` true
        spawned = subprocess.run(["sh", "-c", "sleep 60 >/dev/null 2>&1 & echo $!"], capture_output=True, text=True)
        runner_pid = int(spawned.stdout)
        (tmp_path / f"{OPERATION_ID}.pid").write_text(str(runner_pid))

    async def run_locally(command: str, check: bool = False) -> SimpleNamespace:
        # the remote path is fixed; point it at the fake and shorten the bounded wait
        command = command.replace("/usr/bin/docker", str(fake_docker)).replace("sleep 0.25", "sleep 0.01")
        result = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return SimpleNamespace(exit_status=result.returncode, stdout=result.stdout, stderr=result.stderr)

    ssh_client = SimpleNamespace(run=run_locally)
    try:
        await storage_operations.cancel_storage_operation(ssh_client, OPERATION_ID)
    finally:
        if runner_pid:
            subprocess.run(["kill", "-KILL", str(runner_pid)], capture_output=True)

    helper = f"lium-storage-{str(OPERATION_ID)[:12]}"
    assert docker_log.read_text().splitlines() == [f"stop --time 5 {helper}", f"rm -f -- {helper}"]
    if runner_pid:
        assert subprocess.run(["kill", "-0", str(runner_pid)], capture_output=True).returncode != 0


@pytest.mark.asyncio
@pytest.mark.parametrize("runner_alive", [True, False])
async def test_a_repeated_dispatch_never_starts_a_second_runner(
    monkeypatch: pytest.MonkeyPatch, runner_alive: bool
) -> None:
    # The first runner has read (and deleted) its spec but has no PID or helper yet: the state is STARTING.
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    monkeypatch.setattr(storage_operations.asyncio, "sleep", AsyncMock())
    commands: list[str] = []

    async def run(command: str, check: bool = False) -> SimpleNamespace:
        commands.append(command)
        stdout = ""
        status = 1 if command.startswith("test -e") else 0
        if "echo STARTING" in command:
            stdout = "STARTING\n"
        elif command.startswith("pgrep -f"):
            status = 0 if runner_alive else 1
        return SimpleNamespace(exit_status=status, stdout=stdout, stderr="")

    ssh_client = SimpleNamespace(run=run, start_sftp_client=None)
    if runner_alive:
        files = await storage_operations.start_storage_operation(
            ssh_client, "python3", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        )
        assert files == storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    else:
        monkeypatch.setattr(storage_operations, "report_launch_failure", AsyncMock())
        with pytest.raises(TypeError):  # stops at the sftp spec write, after the stale-helper removal
            await storage_operations.start_storage_operation(
                ssh_client, "python3", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
            )
    probe = next(c for c in commands if c.startswith("pgrep -f"))
    assert f"[-]-spec {storage_operations.StorageOperationFiles.for_operation(OPERATION_ID).spec}" in probe
    assert any("docker rm -f" in c for c in commands) is not runner_alive
    assert any("nohup" in c for c in commands) is False  # never launched a second wrapper


@pytest.mark.asyncio
async def test_a_cancel_reports_with_the_reporter_the_runner_left_in_its_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    report = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_cancelled", report)
    reporter = {"api_url": "https://api.example", "auth_token": "t", "resource": "restore"}
    # what the runner leaves once it has read the spec; no copy of the token is kept anywhere else
    spec = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID).spec
    spec.write_text(json.dumps({"reporter": reporter}))

    async def run_locally(command: str, check: bool = False) -> SimpleNamespace:
        result = subprocess.run(["sh", "-c", command.replace("/usr/bin/docker", "true")], capture_output=True, text=True)
        return SimpleNamespace(exit_status=result.returncode, stdout=result.stdout, stderr=result.stderr)

    await storage_operations.cancel_storage_operation(SimpleNamespace(run=run_locally), OPERATION_ID)

    report.assert_awaited_once_with(OPERATION_ID, {"reporter": reporter})
    assert "redis" not in inspect.signature(storage_operations.start_storage_operation).parameters


@pytest.mark.parametrize("exit_code", [0, 3])
def test_detached_wrapper_keeps_the_log_only_when_the_runner_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    # runs the real launch wrapper with a fake runner; a failed run must leave its log behind
    import asyncio as aio

    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path)))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    # the validator's runner, staged from its payloads: a stand-in that logs and exits
    payloads = tmp_path / "payloads"
    payloads.mkdir()
    (payloads / "storage_runner.py").write_text(f"print('working', flush=True)\nraise SystemExit({exit_code})\n")
    monkeypatch.setattr(executor_payload, "PAYLOAD_DIR", payloads)
    monkeypatch.setattr(executor_payload, "STORAGE", ("storage_runner.py",))
    monkeypatch.setattr(executor_payload, "SERVICE_DIR_PARENT", str(tmp_path / "service"))
    (tmp_path / "service").mkdir()
    launched: list[tuple[str, str]] = []

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "nohup" in command:
            launched.append((command, input))
            return SimpleNamespace(exit_status=0, stdout="", stderr="")
        done = subprocess.run(["sh", "-c", command], input=input, capture_output=True, text=True)
        if check and done.returncode:
            raise RuntimeError(f"exit {done.returncode}")
        return SimpleNamespace(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)

    class Sftp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def open(self, path, mode, attrs=None):
            class Handle:
                async def __aenter__(self):
                    if "x" in mode:
                        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
                    return self

                async def __aexit__(self, *exc):
                    return False

                async def write(self, data):
                    open(path, "w").write(data)

            return Handle()

        async def chmod(self, path, mode):
            return None

    ssh_client = SimpleNamespace(run=run, start_sftp_client=lambda: Sftp())
    files = aio.run(
        storage_operations.start_storage_operation(
            ssh_client, sys.executable, OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        )
    )
    command, _ = launched[0]
    assert "storage_runner.py" in command and storage_operations.REMOTE_RUNNER_PATH not in command
    # the launch in the foreground: the wrapper's own end (its cleanup) is then over when this returns
    subprocess.run(["sh", "-c", command.replace("nohup ", "").rsplit(" &", 1)[0]], text=True, check=True)
    assert not list((tmp_path / "service").iterdir())  # the staged runner goes when it has exited
    log = Path(str(files.log))
    assert not Path(str(files.status)).exists()
    if exit_code:
        assert log.read_text().splitlines() == ["working", "runner exited 3"]
    else:
        assert not log.exists()


@pytest.mark.parametrize("bad_archive", ["not a tar", "changed runner"])
def test_a_failed_unpack_leaves_no_spec_and_no_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_archive: str
) -> None:
    # the spec carries the repository credentials: it must not exist on the node unless a runner is there to take it
    import asyncio as aio
    import base64
    import io
    import tarfile

    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path)))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    reported = AsyncMock()
    monkeypatch.setattr(storage_operations, "report_launch_failure", reported)
    payloads = tmp_path / "payloads"
    payloads.mkdir()
    (payloads / "storage_runner.py").write_text("raise SystemExit(0)\n")
    monkeypatch.setattr(executor_payload, "PAYLOAD_DIR", payloads)
    monkeypatch.setattr(executor_payload, "STORAGE", ("storage_runner.py",))
    monkeypatch.setattr(executor_payload, "SERVICE_DIR_PARENT", str(tmp_path / "service"))
    (tmp_path / "service").mkdir()
    if bad_archive == "not a tar":
        monkeypatch.setattr(executor_payload, "archive", lambda files: base64.b64encode(b"not a tar"))
    else:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            data = b"print('not the validator runner')\n"
            entry = tarfile.TarInfo("storage_runner.py")
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
        monkeypatch.setattr(executor_payload, "archive", lambda files: base64.b64encode(buffer.getvalue()))
    sftp_used: list[str] = []

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        done = subprocess.run(["sh", "-c", command], input=input, capture_output=True, text=True)
        if check and done.returncode:
            raise RuntimeError(f"exit {done.returncode}")
        return SimpleNamespace(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)

    ssh_client = SimpleNamespace(run=run, start_sftp_client=lambda: sftp_used.append("opened"))
    spec = {"engine": "restic", "repository": {"s3_access_key_id": "AKIA-TEST", "password": "restic-test"}}
    with pytest.raises(RuntimeError):
        aio.run(
            storage_operations.start_storage_operation(
                ssh_client, sys.executable, OPERATION_ID, spec, retain_terminal_artifacts=False
            )
        )
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    assert not Path(str(files.spec)).exists() and sftp_used == []
    assert not list((tmp_path / "service").iterdir())  # the partial staging directory is gone
    reported.assert_awaited_once()


def test_a_repeated_dispatch_that_stages_alongside_leaves_the_operation_to_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # two dispatches of one operation both pass the checks while staging: the spec's exclusive create decides the
    # owner, and the other launches nothing, keeps the owner's spec and removes only what it staged
    import asyncio as aio

    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path)))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    monkeypatch.setattr(storage_operations, "report_launch_failure", AsyncMock())
    payloads = tmp_path / "payloads"
    payloads.mkdir()
    (payloads / "storage_runner.py").write_text("raise SystemExit(0)\n")
    monkeypatch.setattr(executor_payload, "PAYLOAD_DIR", payloads)
    monkeypatch.setattr(executor_payload, "STORAGE", ("storage_runner.py",))
    monkeypatch.setattr(executor_payload, "SERVICE_DIR_PARENT", str(tmp_path / "service"))
    (tmp_path / "service").mkdir()
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    launched: list[str] = []

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "nohup" in command:
            launched.append(command)
            return SimpleNamespace(exit_status=0, stdout="", stderr="")
        done = subprocess.run(["sh", "-c", command], input=input, capture_output=True, text=True)
        if "sha256sum" in command:
            Path(str(files.spec)).write_text("first dispatch")   # the other dispatch claims it while this one stages
        if check and done.returncode:
            raise RuntimeError(f"exit {done.returncode}")
        return SimpleNamespace(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)

    class Sftp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def open(self, path, mode, attrs=None):
            class Handle:
                async def __aenter__(self):
                    os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))   # raises: it exists
                    return self

                async def __aexit__(self, *exc):
                    return False

            return Handle()

    ssh_client = SimpleNamespace(run=run, start_sftp_client=lambda: Sftp())
    returned = aio.run(
        storage_operations.start_storage_operation(
            ssh_client, sys.executable, OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        )
    )
    assert returned == files and launched == []
    assert Path(str(files.spec)).read_text() == "first dispatch"
    assert not list((tmp_path / "service").iterdir())
