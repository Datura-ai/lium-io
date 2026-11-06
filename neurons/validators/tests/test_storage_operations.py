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


def _fake_docker(tmp_path: Path, inspect_output: str | None) -> tuple[Path, Path]:
    """A docker that logs its calls; `inspect` answers like a runner container when given an output."""
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

    return SimpleNamespace(run=run_locally)


@pytest.mark.asyncio
@pytest.mark.parametrize("runner_container", [True, False])
async def test_cancel_targets_the_runner_by_name_never_a_stored_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner_container: bool
) -> None:
    # Runs the real cancel shell with a fake docker. A stale PID file names a live unrelated process.
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    fake_docker, docker_log = _fake_docker(tmp_path, "running 0" if runner_container else None)
    unrelated = subprocess.Popen(["sleep", "60"])
    (tmp_path / f"{OPERATION_ID}.pid").write_text(str(unrelated.pid))
    legacy = None
    if not runner_container:
        # an executor before the runner container: its runner is a local process named by its spec path
        legacy = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(60)",
                "--spec",
                str(tmp_path / f"{OPERATION_ID}.json"),
            ]
        )
    try:
        await storage_operations.cancel_storage_operation(_local_ssh(fake_docker), OPERATION_ID)
        assert unrelated.poll() is None  # the stored PID is never signalled
        if legacy:
            assert legacy.wait(timeout=5) is not None
    finally:
        for process in (unrelated, legacy):
            if process:
                process.kill()
                process.wait()

    helper = f"lium-storage-{str(OPERATION_ID)[:12]}"
    runner = f"lium-storage-runner-{str(OPERATION_ID)[:12]}"
    stopped_runner = [f"stop --time 30 {runner}", f"rm -f -- {runner}"] if runner_container else []
    assert docker_log.read_text().splitlines() == [
        f"inspect {runner}",
        *stopped_runner,
        f"stop --time 5 {helper}",
        f"rm -f -- {helper}",
        f"rm -f -- {runner}",  # the artifacts cleanup
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("container_state", "expected"),
    [
        ("running 0", "RUNNING"),
        ("exited 0", "STATUS:0"),
        ("exited 137", "STATUS:137"),
        (None, "STARTING"),
    ],
)
async def test_state_is_read_from_the_runner_container_when_the_files_are_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, container_state: str | None, expected: str
) -> None:
    # after an executor restart its status and PID files are gone; the runner container is not
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    fake_docker, _ = _fake_docker(tmp_path, container_state)
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    assert await storage_operations._operation_state(_local_ssh(fake_docker), files) == expected


@pytest.mark.asyncio
async def test_wait_reattaches_after_the_executor_restarts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    fake_docker, _ = _fake_docker(
        tmp_path, "exited 0"
    )  # the runner container finished after the restart
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
async def test_start_relaunches_a_runner_that_never_received_its_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # an executor restart mid-handoff: the runner container exits with the spec-timeout code; launch afresh
    executor_runner = Path(__file__).resolve().parents[2] / "executor" / "src" / "storage_runner.py"
    assert (
        f"SPEC_TIMEOUT_EXIT = {storage_operations.RUNNER_SPEC_TIMEOUT_EXIT}"
        in executor_runner.read_text()
    )
    monkeypatch.setattr(
        storage_operations, "supports_storage_operation", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(storage_operations, "report_launch_failure", AsyncMock())
    commands: list[str] = []

    async def run(command: str, check: bool = False) -> SimpleNamespace:
        commands.append(command)
        stdout = (
            f"STATUS:{storage_operations.RUNNER_SPEC_TIMEOUT_EXIT}\n"
            if "echo STARTING" in command
            else ""
        )
        return SimpleNamespace(
            exit_status=1 if command.startswith(("test -e", "/usr/bin/docker ps")) else 0,
            stdout=stdout,
            stderr="",
        )

    with pytest.raises(TypeError):  # reaches the sftp spec write: a new launch, not an early return
        await storage_operations.start_storage_operation(
            SimpleNamespace(run=run, start_sftp_client=None),
            "python3",
            OPERATION_ID,
            {"engine": "restic"},
            retain_terminal_artifacts=False,
        )
    runner = f"lium-storage-runner-{str(OPERATION_ID)[:12]}"
    assert any(f"docker rm -f -- {runner}" in command for command in commands)


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
@pytest.mark.parametrize(
    ("running", "owned"),
    [("lium-storage-runner-", True), ("lium-storage-", False), ("", False)],
)
async def test_a_running_runner_container_owns_the_operation(
    tmp_path: Path, running: str, owned: bool
) -> None:
    # Runs the real ownership probe against a docker that lists only the named running container. A running
    # runner container (it survived an executor restart) owns the run; a helper alone, with no local runner, not.
    running_name = f"{running}{str(OPERATION_ID)[:12]}" if running else "none"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        f'#!/bin/sh\ncase "$*" in *"name=^{running_name}$"*) echo abc123;; esac\n'
    )
    fake_docker.chmod(0o755)
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    assert (
        await storage_operations._helper_owned_by_live_runner(
            _local_ssh(fake_docker), OPERATION_ID, files
        )
        is owned
    )
