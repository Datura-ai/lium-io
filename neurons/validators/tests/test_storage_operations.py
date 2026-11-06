import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from services import storage_operations

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
    runner = f"lium-storage-runner-{str(OPERATION_ID)[:12]}"
    # with no local runner the executor restarted: the surviving runner container is stopped too
    surviving_runner = [] if runner_pid else [f"stop --time 30 {runner}", f"rm -f -- {runner}"]
    assert docker_log.read_text().splitlines() == [
        *surviving_runner, f"stop --time 5 {helper}", f"rm -f -- {helper}"
    ]
    if runner_pid:
        assert subprocess.run(["kill", "-0", str(runner_pid)], capture_output=True).returncode != 0


@pytest.mark.parametrize("field", ["bootstrap", "exclude_patterns"])
def test_workspace_field_probe_finds_the_field_in_this_repo_s_executor_models(field: str) -> None:
    # The probe imports the executor's own storage.models; run it against this checkout's copy
    # so a renamed field or a wrong sys.path root fails here, not on every encrypted create.
    executor_src = Path(__file__).resolve().parents[2] / "executor" / "src"
    probe = storage_operations._WORKSPACE_FIELD_PROBE.replace("/root/app/src", str(executor_src))

    assert subprocess.run([sys.executable, "-c", probe, field], check=False).returncode == 0
    assert subprocess.run([sys.executable, "-c", probe, "no_such_field"], check=False).returncode == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("helper_running", "runner_alive"), [(True, True), (False, True), (True, False), (False, False)]
)
async def test_start_never_removes_a_running_helper(
    monkeypatch: pytest.MonkeyPatch, helper_running: bool, runner_alive: bool
) -> None:
    # A repeated dispatch sees no runner PID yet (STARTING) while the first run's helper works.
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
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
            ssh_client, "python3", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        )
        assert files == storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    else:
        monkeypatch.setattr(storage_operations, "report_launch_failure", AsyncMock())
        with pytest.raises(TypeError):  # stops at the sftp spec write, after the stale-helper removal
            await storage_operations.start_storage_operation(
                ssh_client, "python3", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
            )
    helper = f"lium-storage-{str(OPERATION_ID)[:12]}"
    probe = next(c for c in commands if "docker ps -q" in c)
    assert f"name=^{helper}$" in probe and "pgrep -f" in probe
    # a runner container that outlived an executor restart also owns the operation
    assert f"name=^lium-storage-runner-{str(OPERATION_ID)[:12]}$" in probe
    assert any("docker rm -f" in c for c in commands) is not kept
