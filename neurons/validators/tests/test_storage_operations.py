import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import asyncssh
import pytest

from services import storage_operations

OPERATION_ID = UUID("11111111-1111-4111-8111-111111111111")
HELPER = f"lium-storage-{str(OPERATION_ID)[:12]}"


def _fake_docker(tmp_path: Path, inspect_output: str | None) -> tuple[Path, Path]:
    """A docker that logs its calls; `inspect` prints ``inspect_output`` (no such container when None)."""
    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "docker"
    inspect = f'echo "{inspect_output}"; exit 0' if inspect_output is not None else "exit 1"
    fake_docker.write_text(
        f'#!/bin/sh\necho "$*" >> {docker_log}\nif [ "$1" = inspect ]; then {inspect}; fi\n'
    )
    fake_docker.chmod(0o755)
    return fake_docker, docker_log


def _local_ssh(fake_docker: Path) -> SimpleNamespace:
    async def run_locally(command: str, check: bool = False) -> SimpleNamespace:
        # the remote path is fixed; point it at the fake and shorten the bounded waits
        command = command.replace("/usr/bin/docker", str(fake_docker)).replace(
            "sleep 0.25", "sleep 0.01"
        )
        result = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return SimpleNamespace(
            exit_status=result.returncode, stdout=result.stdout, stderr=result.stderr
        )

    return SimpleNamespace(run=run_locally, start_sftp_client=None)


def _legacy_runner(tmp_path: Path) -> subprocess.Popen:
    # an executor before the wrapper helper: its runner is a local process named by its spec path
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            "--spec",
            str(tmp_path / f"{OPERATION_ID}.json"),
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_cancel_signals_the_helper_by_name_never_a_stored_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, legacy: bool
) -> None:
    # Runs the real cancel shell with a fake docker. A stale PID file names a live unrelated process.
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    fake_docker, docker_log = _fake_docker(tmp_path, "false")  # stopped once signalled
    unrelated = subprocess.Popen(["sleep", "60"])
    (tmp_path / f"{OPERATION_ID}.pid").write_text(str(unrelated.pid))
    runner = _legacy_runner(tmp_path) if legacy else None
    try:
        await storage_operations.cancel_storage_operation(_local_ssh(fake_docker), OPERATION_ID)
        assert unrelated.poll() is None  # the stored PID is never signalled
        if runner:
            assert runner.wait(timeout=5) is not None
    finally:
        for process in (unrelated, runner):
            if process:
                process.kill()
                process.wait()

    calls = docker_log.read_text().splitlines()
    # the wrapper in the helper turns SIGTERM into a CANCELLED report; the helper is then removed
    assert calls[0] == f"kill --signal TERM {HELPER}"
    assert calls[-3:] == [f"stop --time 5 {HELPER}", f"rm -f -- {HELPER}", f"rm -f -- {HELPER}"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("helper_state", "status_file", "expected"),
    [
        ("1 running 0 false", None, "RUNNING"),
        ("1 exited 0 false", "0", "STATUS:0"),  # the launcher's 0 only says it started the helper
        ("1 exited 1 false", None, "STATUS:1"),
        ("1 exited 137 true", None, "STATUS:137"),
        ("1 exited 1 true", None, "STATUS:137"),  # docker recorded an OOM kill
        (
            "<no value> running 0 false",
            None,
            "STARTING",
        ),  # an older executor's helper: its files decide
        (None, "1", "STATUS:1"),  # the launcher failed before any helper existed
        (None, None, "STARTING"),
    ],
)
async def test_state_is_read_from_the_wrapper_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    helper_state: str | None,
    status_file: str | None,
    expected: str,
) -> None:
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    fake_docker, _ = _fake_docker(tmp_path, helper_state)
    if status_file is not None:
        (tmp_path / f"{OPERATION_ID}.status").write_text(status_file + "\n")
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    assert await storage_operations._operation_state(_local_ssh(fake_docker), files) == expected


@pytest.mark.asyncio
async def test_wait_reattaches_after_the_executor_restarts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    fake_docker, _ = _fake_docker(
        tmp_path, "1 exited 0 false"
    )  # the helper finished after the restart
    fresh = _local_ssh(fake_docker)

    async def dropped(command: str, check: bool = False) -> SimpleNamespace:
        raise asyncssh.ConnectionLost("executor restarted")

    live = await storage_operations.wait_for_storage_operation(
        SimpleNamespace(run=dropped),
        files,
        poll_interval_seconds=0,
        reconnect=AsyncMock(return_value=fresh),
    )
    assert live is fresh  # the caller continues on the reopened session


@pytest.mark.asyncio
async def test_wait_reports_a_killed_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the wrapper cannot report its own OOM kill or SIGKILL: the validator's poll does
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    report = AsyncMock()
    monkeypatch.setattr(storage_operations, "_report_operation_status", report)
    fake_docker, _ = _fake_docker(tmp_path, "1 exited 137 true")
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    spec = {"operation_id": str(OPERATION_ID), "action": "restore", "reporter": {}}

    with pytest.raises(storage_operations.StorageOperationLaunchError, match="exit 137"):
        await storage_operations.wait_for_storage_operation(
            _local_ssh(fake_docker), files, poll_interval_seconds=0, spec=spec
        )
    assert (
        report.await_args.kwargs["status"] == "FAILED"
        and report.await_args.kwargs["stage"] == "RESTORING"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "helper_state", ["1 running 0 false", "1 exited 137 true", "1 exited 1 false"]
)
async def test_redispatch_keeps_a_running_helper_and_relaunches_a_finished_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, helper_state: str
) -> None:
    # Real state shell with a fake docker. A running helper owns the run; a finished one is removed and run again,
    # and a killed one is first reported, since it could not report for itself.
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        storage_operations, "supports_storage_operation", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(storage_operations, "report_launch_failure", AsyncMock())
    killed = AsyncMock()
    monkeypatch.setattr(storage_operations, "report_killed_helper", killed)
    fake_docker, docker_log = _fake_docker(tmp_path, helper_state)
    ssh_client = _local_ssh(fake_docker)
    running = "running" in helper_state

    if running:
        files = await storage_operations.start_storage_operation(
            ssh_client,
            "python3",
            OPERATION_ID,
            {"engine": "restic"},
            retain_terminal_artifacts=False,
        )
        assert files.helper == HELPER
    else:
        with pytest.raises(TypeError):  # reaches the sftp spec write: launched again
            await storage_operations.start_storage_operation(
                ssh_client,
                "python3",
                OPERATION_ID,
                {"engine": "restic"},
                retain_terminal_artifacts=False,
            )
    assert (f"rm -f -- {HELPER}" in docker_log.read_text().splitlines()) is not running
    assert killed.await_count == ("137" in helper_state)


@pytest.mark.asyncio
@pytest.mark.parametrize("runner_alive", [True, False])
async def test_an_older_executor_s_running_helper_is_owned_only_by_a_live_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner_alive: bool
) -> None:
    # Runs the real ownership shell: docker lists the running helper; pgrep looks for the runner's command line.
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(f'#!/bin/sh\ncase "$*" in *"name=^{HELPER}$"*) echo abc123;; esac\n')
    fake_docker.chmod(0o755)
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    runner = _legacy_runner(tmp_path) if runner_alive else None
    try:
        files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
        owned = await storage_operations._helper_owned_by_live_runner(
            _local_ssh(fake_docker), OPERATION_ID, files
        )
    finally:
        if runner:
            runner.kill()
            runner.wait()
    assert owned is runner_alive


@pytest.mark.parametrize("field", ["bootstrap", "exclude_patterns"])
def test_workspace_field_probe_finds_the_field_in_this_repo_s_executor_models(field: str) -> None:
    # The probe imports the executor's own storage.models; run it against this checkout's copy
    # so a renamed field or a wrong sys.path root fails here, not on every encrypted create.
    executor_src = Path(__file__).resolve().parents[2] / "executor" / "src"
    probe = storage_operations._WORKSPACE_FIELD_PROBE.replace("/root/app/src", str(executor_src))

    assert subprocess.run([sys.executable, "-c", probe, field], check=False).returncode == 0
    assert (
        subprocess.run([sys.executable, "-c", probe, "no_such_field"], check=False).returncode == 3
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("helper_running", "runner_alive"), [(True, True), (False, True), (True, False), (False, False)]
)
async def test_start_never_removes_a_running_helper(
    monkeypatch: pytest.MonkeyPatch, helper_running: bool, runner_alive: bool
) -> None:
    # A repeated dispatch sees no runner PID yet (STARTING) while the first run's helper works.
    monkeypatch.setattr(
        storage_operations, "supports_storage_operation", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(storage_operations.asyncio, "sleep", AsyncMock())
    commands: list[str] = []

    async def run(command: str, check: bool = False) -> SimpleNamespace:
        commands.append(command)
        stdout = ""
        status = 1 if command.startswith("test -e") else 0
        if "echo STARTING" in command:
            stdout = "STARTING\n"
        elif "docker ps -q" in command:
            # a dead runner's helper is an orphan: only a live runner's helper is kept
            status = 0 if helper_running and runner_alive else 1
        return SimpleNamespace(exit_status=status, stdout=stdout, stderr="")

    ssh_client = SimpleNamespace(run=run, start_sftp_client=None)
    kept = helper_running and runner_alive
    if kept:
        files = await storage_operations.start_storage_operation(
            ssh_client,
            "python3",
            OPERATION_ID,
            {"engine": "restic"},
            retain_terminal_artifacts=False,
        )
        assert files == storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    else:
        monkeypatch.setattr(storage_operations, "report_launch_failure", AsyncMock())
        with pytest.raises(
            TypeError
        ):  # stops at the sftp spec write, after the stale-helper removal
            await storage_operations.start_storage_operation(
                ssh_client,
                "python3",
                OPERATION_ID,
                {"engine": "restic"},
                retain_terminal_artifacts=False,
            )
    assert any("docker rm -f" in c for c in commands) is not kept


@pytest.mark.asyncio
async def test_pod_delete_removes_exited_helpers_holding_its_volume() -> None:
    ssh = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(stdout="", exit_status=0)))

    await storage_operations.remove_finished_helpers_mounting(ssh, "volume_abc")

    command = ssh.run.await_args.args[0]
    assert "label=lium.storage-wrapper" in command and "status=exited" in command
    assert "volume=volume_abc" in command and "docker rm -f" in command
