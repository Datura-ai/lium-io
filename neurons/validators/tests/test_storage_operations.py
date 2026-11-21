import inspect
import json
import os
import subprocess
import sys
import asyncio
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


def _spawn_runner(spec_path: object) -> int:
    """An unresponsive runner (ignores the cancel file) that carries its spec path like the real one; detached so no
    zombie keeps `kill -0` true."""
    spawned = subprocess.run(
        ["sh", "-c", f"sh -c 'sleep 60; : --spec {spec_path}' >/dev/null 2>&1 & echo $!"], capture_output=True, text=True
    )
    return int(spawned.stdout)


def _record_runner(directory: Path, pid: int) -> None:
    """The PID and start time as a launch records them."""
    (directory / f"{OPERATION_ID}.start").write_text(_start_time(pid) + "\n")
    (directory / f"{OPERATION_ID}.pid").write_text(f"{pid}\n")


def _start_time(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]


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
        runner_pid = _spawn_runner(storage_operations.StorageOperationFiles.for_operation(OPERATION_ID).spec)
        _record_runner(tmp_path, runner_pid)

    async def run_locally(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        # the remote path is fixed; point it at the fake and shorten the bounded wait
        command = command.replace("/usr/bin/docker", str(fake_docker)).replace("sleep 0.25", "sleep 0.01")
        result = subprocess.run(["sh", "-c", command], capture_output=True, text=True, input=input)
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
@pytest.mark.parametrize("failing_step", ["marker", "lock_open"])
async def test_a_cancel_that_cannot_write_its_marker_still_stops_the_runner_then_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failing_step: str
) -> None:
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    reported = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_cancelled", reported)
    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(f'#!/bin/sh\necho "$*" >> {docker_log}\n')
    fake_docker.chmod(0o755)
    runner_pid = _spawn_runner(storage_operations.StorageOperationFiles.for_operation(OPERATION_ID).spec)
    _record_runner(tmp_path, runner_pid)
    if failing_step == "lock_open":
        # the lock file cannot be opened at all (a directory in its place): the open itself fails in the shell
        Path(storage_operations.StorageOperationFiles.for_operation(OPERATION_ID).lock).mkdir(parents=True)

    async def run_locally(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        # a full or read-only filesystem: the marker write fails
        command = command.replace("/usr/bin/docker", str(fake_docker)).replace("sleep 0.25", "sleep 0.01")
        if failing_step == "marker":
            command = command.replace("touch ", "false ")
        result = subprocess.run(["sh", "-c", command], capture_output=True, text=True, input=input)
        return SimpleNamespace(exit_status=result.returncode, stdout=result.stdout, stderr=result.stderr)

    try:
        with pytest.raises(storage_operations.StorageOperationLaunchError):
            await storage_operations.cancel_storage_operation(SimpleNamespace(run=run_locally), OPERATION_ID)
        assert subprocess.run(["kill", "-0", str(runner_pid)], capture_output=True).returncode != 0
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

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        commands.append(command)
        stdout = ""
        status = 1 if command.startswith("test -e") else 0
        if "echo STARTING" in command:
            stdout = "STARTING\n"
        elif command.startswith("pgrep -f"):
            status = 0 if runner_alive else 1
        return SimpleNamespace(exit_status=status, stdout=stdout, stderr="")

    ssh_client = SimpleNamespace(run=run)
    files = await storage_operations.start_storage_operation(
        ssh_client, "python3", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
    )
    assert files == storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    probe = next(c for c in commands if c.startswith("pgrep -f"))
    assert f"[-]-spec {storage_operations.StorageOperationFiles.for_operation(OPERATION_ID).spec}" in probe
    launches = [c for c in commands if "nohup" in c]
    # a live runner gets no second wrapper; a dead run is relaunched once, its helper removed only under the
    # launch lock, in the launch command itself
    assert len(launches) == (0 if runner_alive else 1)
    assert not any("docker rm -f" in c for c in commands if "nohup" not in c)


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

    async def run_locally(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        result = subprocess.run(["sh", "-c", command.replace("/usr/bin/docker", "true")], capture_output=True, text=True, input=input)
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

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "nohup" in command:
            launched.append(command)
        done = subprocess.run(["sh", "-c", command], capture_output=True, text=True, input=input)
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
    reported = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_cancelled", reported)
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    launch_failure = AsyncMock()
    monkeypatch.setattr(storage_operations, "report_launch_failure", launch_failure)
    launched: list[str] = []

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if command.startswith("nohup"):
            launched.append(command)
            return SimpleNamespace(exit_status=0, stdout="", stderr="")
        done = subprocess.run(["sh", "-c", command.replace("/usr/bin/docker", "true")], capture_output=True, text=True, input=input)
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
    # the cancel had no spec, so no reporter: the launch reports CANCELLED with its own spec
    reported.assert_awaited_once_with(OPERATION_ID, {"engine": "restic"})


def _local_ssh(fake_docker: str = "true"):
    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        command = command.replace("/usr/bin/docker", fake_docker).replace("sleep 0.25", "sleep 0.01")
        done = subprocess.run(["sh", "-c", command], capture_output=True, text=True, input=input)
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

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "nohup" in command:
            await storage_operations.cancel_storage_operation(local, OPERATION_ID)
        return await local.run(command, check, input)

    with pytest.raises(storage_operations.StorageOperationCancelledBeforeLaunch):
        await storage_operations.start_storage_operation(
            SimpleNamespace(run=run, start_sftp_client=local.start_sftp_client),
            "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False,
        )
    await asyncio.sleep(0.2)
    assert not started.exists()
    assert not Path(str(files.spec)).exists() and not Path(str(files.pid)).exists()


@pytest.mark.asyncio
async def test_a_cancel_right_after_a_launch_finds_and_stops_its_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the launch records the operation's process group before releasing the lock, so the cancel that
    # follows always stops the runner, even one that ignores the marker
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path / "ops")))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    runner_pid_file = tmp_path / "runner-pid"
    runner = tmp_path / "runner"
    runner.write_text(f"#!/bin/sh\necho $$ > {runner_pid_file}\nexec sleep 60\n")
    runner.chmod(0o755)
    monkeypatch.setattr(storage_operations, "REMOTE_RUNNER_PATH", str(runner))
    local = _local_ssh()
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)

    await storage_operations.start_storage_operation(
        local, "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
    )
    group = int(Path(str(files.pid)).read_text())
    for _ in range(100):
        if runner_pid_file.exists() and runner_pid_file.read_text().strip():
            break
        await asyncio.sleep(0.05)
    runner_pid = int(runner_pid_file.read_text())
    try:
        await storage_operations.cancel_storage_operation(local, OPERATION_ID)
    finally:
        subprocess.run(["sh", "-c", f"kill -KILL -{group}"], capture_output=True)
    assert subprocess.run(["kill", "-0", str(runner_pid)], capture_output=True).returncode != 0
    assert Path(str(files.cancel)).exists()


@pytest.mark.asyncio
async def test_a_cancel_before_a_slow_wrapper_starts_its_runner_leaves_nothing_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the wrapper is delayed before it starts the runner; the cancel that comes right after the launch
    # finds the recorded process group, and no runner ever does any work
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path / "ops")))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    worked = tmp_path / "worked"
    runner = tmp_path / "runner"
    runner.write_text(f"#!/bin/sh\ntouch {worked}\nexec sleep 60\n")
    runner.chmod(0o755)
    monkeypatch.setattr(storage_operations, "REMOTE_RUNNER_PATH", str(runner))
    local = _local_ssh()
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "nohup" in command:
            assert "exec 8>>" in command
            command = command.replace("exec 8>>", "sleep 1; exec 8>>", 1)
        return await local.run(command, check, input)

    await storage_operations.start_storage_operation(
        SimpleNamespace(run=run, start_sftp_client=local.start_sftp_client),
        "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False,
    )
    group = int(Path(str(files.pid)).read_text())
    try:
        await storage_operations.cancel_storage_operation(local, OPERATION_ID)
    finally:
        subprocess.run(["sh", "-c", f"kill -KILL -{group}"], capture_output=True)
    await asyncio.sleep(0.2)
    assert not worked.exists()
    with pytest.raises(ProcessLookupError):
        os.killpg(group, 0)
    assert not Path(str(files.spec)).exists() and not Path(str(files.pid)).exists()


@pytest.mark.asyncio
async def test_a_cancel_cannot_mark_the_operation_between_the_wrapper_check_and_the_runner_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the wrapper is delayed right before it starts the runner; it holds the lock until the runner has started,
    # so a cancel that comes in that delay marks the operation only after the runner exists
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path / "ops")))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    local = _local_ssh()
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    started = tmp_path / "runner-started"
    runner = tmp_path / "runner"
    # this runner stops on the marker; it records whether the marker was already there when it started
    runner.write_text(
        f"#!/bin/sh\ntest -e {files.cancel} && touch {tmp_path / 'marked-before-start'}\ntouch {started}\n"
        f"while ! test -e {files.cancel}; do sleep 0.05; done\n"
    )
    runner.chmod(0o755)
    monkeypatch.setattr(storage_operations, "REMOTE_RUNNER_PATH", str(runner))

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "nohup" in command:
            runner_start = f"sh {runner} --spec"
            assert command.count(runner_start) == 1
            command = command.replace(runner_start, "sleep 1; " + runner_start)
        return await local.run(command, check, input)

    await storage_operations.start_storage_operation(
        SimpleNamespace(run=run, start_sftp_client=local.start_sftp_client),
        "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False,
    )
    group = int(Path(str(files.pid)).read_text())
    await asyncio.sleep(0.3)
    try:
        await storage_operations.cancel_storage_operation(local, OPERATION_ID)
    finally:
        subprocess.run(["sh", "-c", f"kill -KILL -{group}"], capture_output=True)
    for _ in range(40):
        if started.exists():
            break
        await asyncio.sleep(0.05)
    assert started.exists()
    assert not (tmp_path / "marked-before-start").exists()


@pytest.mark.asyncio
async def test_a_wrapper_whose_pid_was_not_recorded_never_starts_the_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # the background shell is delayed before setsid, so the launcher's group kill after a failed PID write finds
    # no group; the surviving wrapper must still never start a runner no cancel could find
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

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "nohup" in command:
            # the PID write fails (its path is a directory) and setsid runs only after the launcher has given up
            Path(str(files.pid)).mkdir(parents=True, exist_ok=True)
            assert command.count("setsid nohup ") == 1
            command = command.replace("setsid nohup ", "sh -c 'sleep 0.5; exec \"$0\" \"$@\"' setsid nohup ")
        return await local.run(command, check, input)

    with pytest.raises(storage_operations.StorageOperationLaunchError):
        await storage_operations.start_storage_operation(
            SimpleNamespace(run=run, start_sftp_client=local.start_sftp_client),
            "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False,
        )
    await asyncio.sleep(1.5)
    assert not started.exists()


def _counting_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path / "ops")))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    monkeypatch.setattr(storage_operations, "report_launch_failure", AsyncMock())
    starts = tmp_path / "runner-starts"
    runner = tmp_path / "runner"
    runner.write_text(f"#!/bin/sh\necho x >> {starts}\nsleep 0.5\n")
    runner.chmod(0o755)
    monkeypatch.setattr(storage_operations, "REMOTE_RUNNER_PATH", str(runner))
    return starts


@pytest.mark.asyncio
async def test_a_repeated_dispatch_racing_the_first_launch_leaves_it_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # both dispatches would pass the state checks before either launched; the repeat must not rewrite the first's
    # PID and spec, which made both wrappers refuse and neither runner start
    starts = _counting_runner(monkeypatch, tmp_path)
    local = _local_ssh()

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        await asyncio.sleep(0)  # interleave the two dispatches at every host command
        # the wrapper starts late, as over a slow connection: the repeat's launch lands before it takes the lock
        return await local.run(command.replace("exec 8>>", "sleep 0.5; exec 8>>"), check, input)

    ssh_client = SimpleNamespace(run=run, start_sftp_client=local.start_sftp_client)
    launches = [
        storage_operations.start_storage_operation(
            ssh_client, "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=True
        )
        for _ in range(2)
    ]
    await asyncio.gather(*launches)
    await asyncio.sleep(1.6)

    assert starts.read_text() == "x\n"
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    assert Path(str(files.status)).read_text().strip() == "0"


def test_a_launch_finding_a_live_recorded_wrapper_rewrites_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # another validator process already launched the operation: its PID and spec stay
    _counting_runner(monkeypatch, tmp_path)
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    Path(str(files.pid)).parent.mkdir(parents=True, exist_ok=True)
    live = subprocess.Popen(["setsid", "sh", "-c", f"sleep 5; : {files.spec}"])
    try:
        time.sleep(0.2)
        Path(str(files.pid)).write_text(f"{live.pid}\n")
        Path(str(files.spec)).write_text("{}")
        launched = asyncio.run(
            storage_operations._start_storage_operation(
                _local_ssh(), "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=True
            )
        )
        assert launched == files
        assert Path(str(files.pid)).read_text().strip() == str(live.pid)
    finally:
        live.kill()


@pytest.mark.asyncio
async def test_a_wrapper_whose_record_another_launch_replaced_touches_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # another launch recorded its own wrapper after this one was started: the replaced wrapper must leave the
    # operation's PID, spec and status to the recorded one, and start no runner
    starts = _counting_runner(monkeypatch, tmp_path)
    local = _local_ssh()
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    other = "999999"

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "nohup" not in command:
            return await local.run(command, check, input)
        # the wrapper starts late; the other launch's record lands first
        done = await local.run(command.replace("exec 8>>", "sleep 0.5; exec 8>>"), check, input)
        Path(str(files.pid)).write_text(f"{other}\n")
        return done

    await storage_operations._start_storage_operation(
        SimpleNamespace(run=run, start_sftp_client=local.start_sftp_client),
        "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=True,
    )
    await asyncio.sleep(1.0)

    assert not starts.exists()
    assert Path(str(files.pid)).read_text().strip() == other
    assert Path(str(files.spec)).exists()
    assert not Path(str(files.status)).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("carries_spec_path", [False, True])
async def test_a_cancel_leaves_a_process_that_reused_the_recorded_pid_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, carries_spec_path: bool
) -> None:
    # the wrapper was killed before removing its PID file and the host gave that PID to an unrelated group leader,
    # also one whose command line happens to carry the spec path: its start time is not the recorded one
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(f'#!/bin/sh\necho "$*" >> {docker_log}\n')
    fake_docker.chmod(0o755)
    spec = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID).spec
    unrelated = subprocess.Popen(["setsid", "sh", "-c", f"sleep 60; : {spec}"] if carries_spec_path else ["setsid", "sleep", "60"])
    (tmp_path / f"{OPERATION_ID}.pid").write_text(str(unrelated.pid))
    if carries_spec_path:
        (tmp_path / f"{OPERATION_ID}.start").write_text(str(int(_start_time(unrelated.pid)) + 1) + "\n")

    async def run_locally(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        command = command.replace("/usr/bin/docker", str(fake_docker)).replace("sleep 0.25", "sleep 0.01")
        result = subprocess.run(["sh", "-c", command], capture_output=True, text=True, input=input)
        return SimpleNamespace(exit_status=result.returncode, stdout=result.stdout, stderr=result.stderr)

    try:
        await asyncio.sleep(0.2)
        await storage_operations.cancel_storage_operation(SimpleNamespace(run=run_locally), OPERATION_ID)
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
    helper = f"lium-storage-{str(OPERATION_ID)[:12]}"
    assert docker_log.read_text().splitlines() == [f"stop --time 5 {helper}", f"rm -f -- {helper}"]


@pytest.mark.asyncio
async def test_two_validator_processes_launching_one_operation_start_one_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # two processes, which the in-process launch lock does not cover: both find a dead run and pass the checks
    # before either launches, then the second pauses until the first's runner is up, before any step that changes
    # the node
    starts = _counting_runner(monkeypatch, tmp_path)
    local = _local_ssh()
    first_launched, second_checked = asyncio.Event(), asyncio.Event()
    read_only = ("test -e", "install -d", "pgrep -f")

    def process(first: bool) -> SimpleNamespace:
        # the first changes nothing until the second has passed its checks; the second then changes nothing (on the
        # node, beyond its own spec file) until the first's runner is up
        async def gate() -> None:
            if first:
                await second_checked.wait()
            else:
                second_checked.set()

        async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
            if not (command.startswith(read_only) or "echo STARTING" in command):
                await gate()
                if not first:
                    await first_launched.wait()
            result = await local.run(command, check, input)
            if first and "setsid nohup" in command:
                await asyncio.sleep(0.2)  # the first wrapper has taken the lock and started its runner
                first_launched.set()
            return result

        class Sftp:
            async def __aenter__(self):
                await gate()
                self.inner = local.start_sftp_client()
                return await self.inner.__aenter__()

            async def __aexit__(self, *exc):
                return await self.inner.__aexit__(*exc)

        return SimpleNamespace(run=run, start_sftp_client=Sftp)

    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    # an earlier run died and left its files: both processes find it dead and would clear it
    Path(str(files.pid)).parent.mkdir(parents=True, exist_ok=True)
    dead = subprocess.Popen(["true"])
    dead.wait()
    Path(str(files.pid)).write_text(f"{dead.pid}\n")
    Path(str(files.spec)).write_text("{}")
    await asyncio.gather(
        *(
            storage_operations._start_storage_operation(
                process(first), "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=True
            )
            for first in (True, False)
        )
    )
    await asyncio.sleep(1.2)

    assert starts.read_text() == "x\n"
    assert Path(str(files.status)).read_text().strip() == "0"
    assert not list(Path(str(files.spec)).parent.glob(f"{Path(str(files.spec)).name}.*"))


@pytest.mark.asyncio
async def test_a_cancel_never_signals_a_pid_recorded_without_its_start_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # a PID file from before start times were recorded: only waited for (a real runner stops on the marker),
    # the helper removed, and the cancel fails while that process still runs
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(f'#!/bin/sh\necho "$*" >> {docker_log}\n')
    fake_docker.chmod(0o755)
    runner_pid = _spawn_runner(storage_operations.StorageOperationFiles.for_operation(OPERATION_ID).spec)
    (tmp_path / f"{OPERATION_ID}.pid").write_text(str(runner_pid))
    local = _local_ssh(str(fake_docker))
    try:
        with pytest.raises(storage_operations.StorageOperationLaunchError):
            await storage_operations.cancel_storage_operation(local, OPERATION_ID)
        assert subprocess.run(["kill", "-0", str(runner_pid)], capture_output=True).returncode == 0
    finally:
        subprocess.run(["kill", "-KILL", str(runner_pid)], capture_output=True)
    helper = f"lium-storage-{str(OPERATION_ID)[:12]}"
    assert docker_log.read_text().splitlines() == [f"stop --time 5 {helper}", f"rm -f -- {helper}"]


def _patch_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path / "ops")))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    monkeypatch.setattr(storage_operations, "report_launch_failure", AsyncMock())
    started = tmp_path / "runner-started"
    runner = tmp_path / "runner"
    runner.write_text(f"#!/bin/sh\ntouch {started}\n")
    runner.chmod(0o755)
    monkeypatch.setattr(storage_operations, "REMOTE_RUNNER_PATH", str(runner))
    return started


@pytest.mark.asyncio
async def test_a_launch_whose_spec_arrives_cut_short_runs_nothing_and_leaves_no_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started = _patch_launch(monkeypatch, tmp_path)
    local = _local_ssh()
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        # the session drops mid-transfer: stdin ends early
        return await local.run(command, check, input[:-3] if input else input)

    with pytest.raises(storage_operations.StorageOperationLaunchError, match="exit 9"):
        await storage_operations.start_storage_operation(
            SimpleNamespace(run=run), "sh", OPERATION_ID, {"engine": "restic", "password": "secret"},
            retain_terminal_artifacts=False,
        )
    await asyncio.sleep(0.2)
    assert not started.exists()
    assert sorted(p.name for p in Path(str(files.spec)).parent.iterdir()) == []


def test_a_launch_cut_by_a_hangup_while_receiving_its_spec_removes_the_partial_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started = _patch_launch(monkeypatch, tmp_path)
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    Path(str(files.spec)).parent.mkdir(parents=True)
    Path(str(files.lock)).parent.mkdir(parents=True)
    commands: list[str] = []

    async def capture(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        commands.append(command)
        # nothing there yet (no marker, no runner); the launch command itself is only captured
        failed = "nohup" in command or command.startswith(("test -e", "pgrep"))
        return SimpleNamespace(exit_status=1 if failed else 0, stdout="", stderr="")

    with pytest.raises(storage_operations.StorageOperationLaunchError):
        asyncio.run(storage_operations.start_storage_operation(
            SimpleNamespace(run=capture), "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        ))
    launch = next(c for c in commands if "nohup" in c)
    shell = subprocess.Popen(["sh", "-c", launch], stdin=subprocess.PIPE, start_new_session=True)
    shell.stdin.write(b'{"engine":')
    shell.stdin.flush()
    part = Path(str(files.part))
    for _ in range(100):
        if part.exists():
            break
        time.sleep(0.02)
    assert part.exists()
    os.killpg(shell.pid, 1)  # the session's hangup reaches the whole command
    assert shell.wait(timeout=10) != 0
    assert not part.exists() and not Path(str(files.spec)).exists()
    assert not started.exists()


def test_a_launch_waiting_for_the_lock_leaves_the_holders_partial_spec_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # review at 42e8268: a second launch that times out on the lock must not remove the spec the holder is receiving
    import fcntl

    _patch_launch(monkeypatch, tmp_path)
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    Path(str(files.spec)).parent.mkdir(parents=True)
    Path(str(files.lock)).parent.mkdir(parents=True)
    commands: list[str] = []

    async def capture(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        commands.append(command)
        failed = "nohup" in command or command.startswith(("test -e", "pgrep"))
        return SimpleNamespace(exit_status=1 if failed else 0, stdout="", stderr="")

    with pytest.raises(storage_operations.StorageOperationLaunchError):
        asyncio.run(storage_operations.start_storage_operation(
            SimpleNamespace(run=capture), "sh", OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        ))
    launch = next(c for c in commands if "nohup" in c).replace("flock -w 30 9", "flock -w 0.2 9")
    part = Path(str(files.part))
    part.write_text('{"engine":')  # the lock holder's transfer in progress
    with open(str(files.lock), "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        done = subprocess.run(["sh", "-c", launch], input=b'{"engine":"restic"}', capture_output=True, timeout=10)
    assert done.returncode == 6
    assert part.exists()


@pytest.mark.asyncio
async def test_a_cancel_stops_a_runner_of_the_earlier_release_through_its_own_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # review at 42e8268: a runner started before start times were recorded watches the old marker; it gets it and
    # stops, while its PID is still never signalled
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    legacy = tmp_path / f"{OPERATION_ID}.cancel"
    spawned = subprocess.run(
        ["sh", "-c", f"sh -c 'while ! test -e {legacy}; do sleep 0.05; done; : --spec {files.spec}' "
         ">/dev/null 2>&1 & echo $!"],
        capture_output=True, text=True,
    )
    runner_pid = int(spawned.stdout)
    (tmp_path / f"{OPERATION_ID}.pid").write_text(str(runner_pid))
    try:
        await storage_operations.cancel_storage_operation(_local_ssh(), OPERATION_ID)
        assert subprocess.run(["kill", "-0", str(runner_pid)], capture_output=True).returncode != 0
    finally:
        subprocess.run(["kill", "-KILL", str(runner_pid)], capture_output=True)


@pytest.mark.asyncio
async def test_a_cancel_before_an_earlier_release_wrapper_records_its_pid_still_stops_its_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # review at ecaaa94: that wrapper writes the spec, removes any old marker, starts the runner and only then
    # writes its PID; a cancel in between must still leave the old marker for that runner
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    legacy = tmp_path / f"{OPERATION_ID}.cancel"
    seen = tmp_path / "runner.pid"
    Path(str(files.spec)).write_text('{"engine":"restic"}')
    wrapper = subprocess.Popen(
        ["sh", "-c", f"sleep 0.5; rm -f {legacy}; "
         f"sh -c 'while ! test -e {legacy}; do sleep 0.05; done; : --spec {files.spec}' >/dev/null 2>&1 & "
         f"echo $! > {seen}; echo $! > {files.pid}"],
    )
    try:
        await storage_operations.cancel_storage_operation(_local_ssh(), OPERATION_ID)
        wrapper.wait(timeout=10)
        runner_pid = int(seen.read_text())
        for _ in range(40):
            if subprocess.run(["kill", "-0", str(runner_pid)], capture_output=True).returncode != 0:
                break
            await asyncio.sleep(0.05)
        assert subprocess.run(["kill", "-0", str(runner_pid)], capture_output=True).returncode != 0
    finally:
        if seen.exists():
            subprocess.run(["kill", "-KILL", seen.read_text().strip()], capture_output=True)


@pytest.mark.asyncio
async def test_a_cancel_after_the_earlier_release_runner_read_its_spec_still_stops_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # review at 175c9ca: that runner unlinks the spec on read, which can be before its wrapper writes the PID;
    # the spec's absence is no sign the run is gone
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    legacy = tmp_path / f"{OPERATION_ID}.cancel"
    seen = tmp_path / "runner.pid"
    Path(str(files.spec)).write_text('{"engine":"restic"}')
    wrapper = subprocess.Popen(
        ["sh", "-c", f"sleep 0.2; rm -f {legacy}; "
         f"sh -c 'rm -f {files.spec}; while ! test -e {legacy}; do sleep 0.05; done; : --spec {files.spec}' "
         f">/dev/null 2>&1 & echo $! > {seen}; sleep 0.5; echo $! > {files.pid}"],
    )
    try:
        await storage_operations.cancel_storage_operation(_local_ssh(), OPERATION_ID)
        wrapper.wait(timeout=10)
        runner_pid = int(seen.read_text())
        for _ in range(40):
            if subprocess.run(["kill", "-0", str(runner_pid)], capture_output=True).returncode != 0:
                break
            await asyncio.sleep(0.05)
        assert subprocess.run(["kill", "-0", str(runner_pid)], capture_output=True).returncode != 0
    finally:
        wrapper.kill()
        if seen.exists():
            subprocess.run(["kill", "-KILL", seen.read_text().strip()], capture_output=True)


@pytest.mark.asyncio
async def test_a_cancel_fails_while_an_earlier_release_wrapper_runs_without_a_pid_past_its_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # review at 175c9ca: a wrapper delayed past the bounded wait would start a runner no marker reaches
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    report = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_cancelled", report)
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    # review at 64c448a: the failed cancel still stops and removes the helper, which holds the credentials
    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(f'#!/bin/sh\necho "$*" >> {docker_log}\n')
    fake_docker.chmod(0o755)
    helper = storage_operations.storage_helper_name(OPERATION_ID)
    wrapper = subprocess.Popen(["sh", "-c", f"sleep 30; : --spec {files.spec}"])
    try:
        with pytest.raises(storage_operations.StorageOperationLaunchError):
            await storage_operations.cancel_storage_operation(_local_ssh(str(fake_docker)), OPERATION_ID)
        report.assert_not_awaited()
        assert docker_log.read_text().splitlines() == [f"stop --time 5 {helper}", f"rm -f -- {helper}"]
    finally:
        wrapper.kill()
