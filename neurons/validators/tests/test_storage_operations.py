import inspect
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from services import storage_operations

OPERATION_ID = UUID("11111111-1111-4111-8111-111111111111")


@pytest.fixture(autouse=True)
def _cancel_markers_in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        storage_operations, "REMOTE_CANCEL_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path / "cancels"))
    )


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
async def test_a_cancel_that_cannot_write_its_marker_still_stops_the_runner_then_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    reported = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_cancelled", reported)
    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(f'#!/bin/sh\necho "$*" >> {docker_log}\n')
    fake_docker.chmod(0o755)
    spawned = subprocess.run(["sh", "-c", "sleep 60 >/dev/null 2>&1 & echo $!"], capture_output=True, text=True)
    runner_pid = int(spawned.stdout)
    (tmp_path / f"{OPERATION_ID}.pid").write_text(str(runner_pid))

    async def run_locally(command: str, check: bool = False) -> SimpleNamespace:
        # a full or read-only filesystem: the marker write fails
        command = (
            command.replace("/usr/bin/docker", str(fake_docker))
            .replace("sleep 0.25", "sleep 0.01")
            .replace("touch ", "false ")
        )
        result = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return SimpleNamespace(exit_status=result.returncode, stdout=result.stdout, stderr=result.stderr)

    try:
        with pytest.raises(storage_operations.StorageOperationLaunchError):
            await storage_operations.cancel_storage_operation(SimpleNamespace(run=run_locally), OPERATION_ID)
    finally:
        subprocess.run(["kill", "-KILL", str(runner_pid)], capture_output=True)
    helper = f"lium-storage-{str(OPERATION_ID)[:12]}"
    assert docker_log.read_text().splitlines() == [f"stop --time 5 {helper}", f"rm -f -- {helper}"]
    reported.assert_not_awaited()


@pytest.mark.parametrize("field", ["bootstrap", "exclude_patterns"])
def test_workspace_field_probe_finds_the_field_in_this_repo_s_executor_models(field: str) -> None:
    # The probe imports the executor's own storage.models; run it against this checkout's copy
    # so a renamed field or a wrong sys.path root fails here, not on every encrypted create.
    executor_src = Path(__file__).resolve().parents[2] / "executor" / "src"
    probe = storage_operations._WORKSPACE_FIELD_PROBE.replace("/root/app/src", str(executor_src))

    assert subprocess.run([sys.executable, "-c", probe, field], check=False).returncode == 0
    assert subprocess.run([sys.executable, "-c", probe, "no_such_field"], check=False).returncode == 3


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
    runner = tmp_path / "runner"
    runner.write_text(f"#!/bin/sh\necho working\nexit {exit_code}\n")
    runner.chmod(0o755)
    monkeypatch.setattr(storage_operations, "REMOTE_RUNNER_PATH", str(runner))
    launched: list[str] = []

    async def run(command: str, check: bool = False) -> SimpleNamespace:
        if "nohup" in command:
            launched.append(command)
        done = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return SimpleNamespace(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)

    class Sftp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def open(self, path, mode):
            class Handle:
                async def __aenter__(self):
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
            ssh_client, "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        )
    )
    assert len(launched) == 1
    for _ in range(200):  # the detached wrapper removes its PID file last
        if not Path(str(files.pid)).exists() and not Path(str(files.spec)).exists():
            break
        time.sleep(0.05)
    log = Path(str(files.log))
    assert not Path(str(files.status)).exists()
    if exit_code:
        assert log.read_text().splitlines() == ["working", "runner exited 3"]
    else:
        assert not log.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("executor_restarted", [False, True])
async def test_a_launch_that_reaches_the_node_after_its_cancel_never_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, executor_restarted: bool
) -> None:
    # the cancel lands before the runner exists (no PID), on a node that never ran an operation
    # (no operations directory yet); the dispatch sent earlier arrives after it, possibly after an
    # executor restart that wiped /root/app's overlay (the operations directory)
    monkeypatch.setattr(
        storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path / "ops"))
    )
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    launch_failure = AsyncMock()
    monkeypatch.setattr(storage_operations, "report_launch_failure", launch_failure)
    launched: list[str] = []

    async def run(command: str, check: bool = False) -> SimpleNamespace:
        if command.startswith("nohup"):
            launched.append(command)
            return SimpleNamespace(exit_status=0, stdout="", stderr="")
        done = subprocess.run(["sh", "-c", command.replace("/usr/bin/docker", "true")], capture_output=True, text=True)
        return SimpleNamespace(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)

    ssh_client = SimpleNamespace(run=run, start_sftp_client=None)
    await storage_operations.cancel_storage_operation(ssh_client, OPERATION_ID)
    if executor_restarted:
        subprocess.run(["rm", "-rf", str(tmp_path / "ops")], check=True)

    with pytest.raises(storage_operations.StorageOperationCancelledBeforeLaunch):
        await storage_operations.start_storage_operation(
            ssh_client, "python3", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        )
    assert launched == []
    launch_failure.assert_not_awaited()


def _local_ssh(fake_docker: str = "true"):
    async def run(command: str, check: bool = False) -> SimpleNamespace:
        command = command.replace("/usr/bin/docker", fake_docker).replace("sleep 0.25", "sleep 0.01")
        done = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return SimpleNamespace(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)

    class Sftp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def open(self, path, mode):
            class Handle:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *exc):
                    return False

                async def write(self, data):
                    open(path, "w").write(data)

            return Handle()

        async def chmod(self, path, mode):
            return None

    return SimpleNamespace(run=run, start_sftp_client=lambda: Sftp())


@pytest.mark.asyncio
async def test_a_cancel_between_the_launch_check_and_its_start_stops_the_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the launch passes its early marker check, then the cancel marks the operation before the runner starts
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path / "ops")))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    monkeypatch.setattr(storage_operations, "report_launch_failure", AsyncMock())
    started = tmp_path / "runner-started"
    runner = tmp_path / "runner"
    runner.write_text(f"#!/bin/sh\ntouch {started}\n")
    runner.chmod(0o755)
    monkeypatch.setattr(storage_operations, "REMOTE_RUNNER_PATH", str(runner))
    local = _local_ssh()
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)

    async def run(command: str, check: bool = False) -> SimpleNamespace:
        if "nohup" in command:
            await storage_operations.cancel_storage_operation(local, OPERATION_ID)
        return await local.run(command, check)

    with pytest.raises(storage_operations.StorageOperationCancelledBeforeLaunch):
        await storage_operations.start_storage_operation(
            SimpleNamespace(run=run, start_sftp_client=local.start_sftp_client),
            "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False,
        )
    time.sleep(0.2)
    assert not started.exists()
    assert not Path(str(files.spec)).exists() and not Path(str(files.pid)).exists()


@pytest.mark.asyncio
async def test_a_cancel_right_after_a_launch_finds_and_stops_its_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the launch holds the lock until its PID is written, so the cancel that follows always sees the runner
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path / "ops")))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\nexec sleep 60\n")  # ignores the marker like an unresponsive runner
    runner.chmod(0o755)
    monkeypatch.setattr(storage_operations, "REMOTE_RUNNER_PATH", str(runner))
    local = _local_ssh()
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)

    await storage_operations.start_storage_operation(
        local, "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
    )
    runner_pid = int(Path(str(files.pid)).read_text())
    try:
        await storage_operations.cancel_storage_operation(local, OPERATION_ID)
    finally:
        subprocess.run(["kill", "-KILL", str(runner_pid)], capture_output=True)
    assert subprocess.run(["kill", "-0", str(runner_pid)], capture_output=True).returncode != 0
    assert Path(str(files.cancel)).exists()


@pytest.mark.asyncio
async def test_a_launch_whose_pid_is_not_written_in_time_stops_its_runner_and_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the wrapper starts the runner but is delayed before writing its PID past the launch's wait:
    # the launch must not release the lock with a runner a cancel cannot find
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path / "ops")))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    launch_failure = AsyncMock()
    monkeypatch.setattr(storage_operations, "report_launch_failure", launch_failure)
    worked = tmp_path / "worked"
    runner = tmp_path / "runner"
    runner.write_text(f"#!/bin/sh\nsleep 1\ntouch {worked}\n")
    runner.chmod(0o755)
    monkeypatch.setattr(storage_operations, "REMOTE_RUNNER_PATH", str(runner))
    local = _local_ssh()
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)

    async def run(command: str, check: bool = False) -> SimpleNamespace:
        if "nohup" in command:
            assert "child=$!; " in command and "-lt 100; do sleep 0.05" in command
            command = command.replace("child=$!; ", "child=$!; sleep 2; ").replace(
                "-lt 100; do sleep 0.05", "-lt 4; do sleep 0.05"
            )
        return await local.run(command, check)

    with pytest.raises(storage_operations.StorageOperationLaunchError):
        await storage_operations.start_storage_operation(
            SimpleNamespace(run=run, start_sftp_client=local.start_sftp_client),
            "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False,
        )
    launch_failure.assert_awaited_once()
    time.sleep(1.5)
    assert not worked.exists()
    assert not Path(str(files.spec)).exists() and not Path(str(files.pid)).exists()
