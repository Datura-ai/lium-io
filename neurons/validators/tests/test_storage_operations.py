import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

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
        owned = await storage_operations._runner_alive(_local_ssh(fake_docker), files)
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
        if command.startswith("nohup"):
            launched.append(command)
            return SimpleNamespace(exit_status=0, stdout="", stderr="")
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
    subprocess.run(["sh", "-c", launched[0].removeprefix("nohup ").rsplit(" &", 1)[0]], check=True)
    log = Path(str(files.log))
    assert not Path(str(files.status)).exists()
    if exit_code:
        assert log.read_text().splitlines() == ["working", "runner exited 3"]
    else:
        assert not log.exists()



def _node(state: str, commands: list[str]):
    async def run(command: str) -> str:
        commands.append(command)
        if command.startswith("cat "):
            return '{"reporter": {"api_url": "https://x", "auth_token": "t", "resource": "backup"}}'
        return state if "docker inspect" in command else ""

    return run


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "message"),
    [("STATUS:137\n", "killed"), ("STATUS:1\n", "exited with code 1"), ("STATUS:139\n", "code 139")],
)
async def test_an_exited_helper_with_its_job_open_is_reported_failed_and_removed(monkeypatch, state, message):
    report = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_operation_status", report)
    commands: list[str] = []

    assert await storage_operations.report_exited_detached_helper(_node(state, commands), str(uuid4()))
    assert report.await_args.kwargs["status"] == "FAILED"
    assert report.await_args.kwargs["stage"] == "BACKUP"
    assert message in report.await_args.kwargs["message"]
    assert "docker rm -f" in commands[-1]


@pytest.mark.asyncio
async def test_a_running_helper_is_left_alone(monkeypatch):
    report = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_operation_status", report)
    commands: list[str] = []

    assert not await storage_operations.report_exited_detached_helper(_node("RUNNING\n", commands), str(uuid4()))
    report.assert_not_awaited()
    assert len(commands) == 1
