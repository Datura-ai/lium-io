import inspect
import json
import os
import subprocess
import sys
import textwrap
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


@pytest.fixture
def node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in executor: the operation directory, the staging parent and the validator's runner payload (a script
    the test writes per case), with every remote command run by a local shell."""
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path)))
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    monkeypatch.setattr(storage_operations, "report_launch_failure", AsyncMock())
    payloads = tmp_path / "payloads"
    payloads.mkdir()
    monkeypatch.setattr(executor_payload, "PAYLOAD_DIR", payloads)
    monkeypatch.setattr(executor_payload, "STORAGE", ("storage_runner.py",))
    monkeypatch.setattr(executor_payload, "SERVICE_DIR_PARENT", str(tmp_path / "service"))
    (tmp_path / "service").mkdir()
    return tmp_path


def _runner(node: Path, body: str, at_import: str = "") -> None:
    # like the real runner, the body runs only as the script: the dispatch's start check imports the module
    (node / "payloads" / "storage_runner.py").write_text(
        at_import + "if __name__ == '__main__':\n" + textwrap.indent(body or "pass\n", "    ")
    )


async def _run_locally(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
    done = subprocess.run(["sh", "-c", command.replace("/usr/bin/docker", "true")], input=input, capture_output=True, text=True)
    if check and done.returncode:
        raise RuntimeError(f"exit {done.returncode}")
    return SimpleNamespace(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)


class _LocalSFTP:
    """The spec upload's SFTP, on the local disk."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def open(self, path: str, mode: str):
        class _File:
            async def __aenter__(self):
                self.f = open(path, mode)
                return self

            async def __aexit__(self, *exc):
                self.f.close()

            async def write(self, data):
                self.f.write(data)

        return _File()

    async def chmod(self, path: str, mode: int):
        os.chmod(path, mode)


def _node_client(run=None) -> SimpleNamespace:
    return SimpleNamespace(run=run or _run_locally, start_sftp_client=_LocalSFTP)


def _dispatch(spec: dict | None = None, retain: bool = False) -> storage_operations.StorageOperationFiles:
    import asyncio as aio

    return aio.run(
        storage_operations.start_storage_operation(
            _node_client(), sys.executable, OPERATION_ID, spec or {"engine": "restic"},
            retain_terminal_artifacts=retain,
        )
    )


def _wait_for(path: Path, gone: bool = False, seconds: float = 10.0) -> None:
    import time

    deadline = time.time() + seconds
    while path.exists() is gone:
        assert time.time() < deadline, f"{path} {'still there' if gone else 'never appeared'}"
        time.sleep(0.05)


def _staged(node: Path) -> list[str]:
    return [p.name for p in (node / "service").iterdir()]


@pytest.mark.parametrize("exit_code", [0, 3])
def test_an_online_run_runs_the_staged_runner_and_keeps_its_log_only_when_it_fails(node: Path, exit_code: int) -> None:
    _runner(node, f"print('working', flush=True)\nraise SystemExit({exit_code})\n")
    files = _dispatch()
    log = Path(str(files.log))
    import time

    deadline = time.time() + 10
    # the log is the wrapper's last step
    while (exit_code and "runner exited" not in log.read_text()) or (not exit_code and log.exists()):
        assert time.time() < deadline
        time.sleep(0.05)
    if exit_code:
        assert log.read_text().splitlines() == ["working", "runner exited 3"]
    assert _staged(node) == []   # the staged runner goes when it has exited
    assert not Path(str(files.status)).exists() and not Path(str(files.spec)).exists()
    assert not Path(str(files.pid)).exists()



@pytest.mark.parametrize("bad_archive", ["not a tar", "changed runner"])
def test_a_failed_unpack_leaves_no_spec_and_no_staging_directory(
    node: Path, monkeypatch: pytest.MonkeyPatch, bad_archive: str
) -> None:
    # the spec carries the repository credentials: it must not exist on the node unless a runner is there to take it
    import base64
    import io
    import tarfile

    _runner(node, "raise SystemExit(0)\n")
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
    spec = {"engine": "restic", "repository": {"s3_access_key_id": "AKIA-TEST", "password": "restic-test"}}
    with pytest.raises(RuntimeError):
        _dispatch(spec)
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    assert not Path(str(files.spec)).exists()
    assert _staged(node) == []   # the partial staging directory is gone
    storage_operations.report_launch_failure.assert_awaited_once()


def _legacy_runner(files: storage_operations.StorageOperationFiles) -> subprocess.Popen:
    """A runner an older validator launched: its PID file only (no launcher record), running this operation's spec."""
    runner = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "--spec", str(files.spec), "--cancel-file", str(files.cancel)]
    )
    Path(str(files.pid)).write_text(f"{runner.pid}\n")
    return runner


def test_a_re_dispatch_after_the_upgrade_leaves_the_older_validators_live_runner_alone(node: Path) -> None:
    runs = node / "runs"
    _runner(node, f"open({str(runs)!r}, 'a').write('run\\n')\n")
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    legacy_runner = _legacy_runner(files)
    try:
        assert _dispatch() == files
        import time

        time.sleep(0.5)
        assert legacy_runner.poll() is None
        assert not runs.exists() and _staged(node) == []   # no second wrapper, nothing staged
        assert Path(str(files.pid)).read_text() == f"{legacy_runner.pid}\n"
    finally:
        legacy_runner.kill()
        legacy_runner.wait()



def test_a_dispatch_whose_staging_fails_with_no_run_reports_the_failure(node: Path) -> None:
    import asyncio as aio

    _runner(node, "raise SystemExit(0)\n")
    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "printf \"%s\" \"$d\"" in command:
            raise OSError(28, "No space left on device")
        return await _run_locally(command, check, input)

    with pytest.raises(OSError):
        aio.run(storage_operations.start_storage_operation(
            _node_client(run), sys.executable, OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        ))
    storage_operations.report_launch_failure.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("starts", [True, False])
async def test_an_executor_whose_interpreter_cannot_start_does_not_support_storage_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, starts: bool
) -> None:
    runner = tmp_path / "storage_runner.py"
    runner.touch()
    monkeypatch.setattr(storage_operations, "REMOTE_RUNNER_PATH", str(runner))
    monkeypatch.setattr(executor_payload, "EXECUTOR_PYTHON", sys.executable if starts else str(tmp_path / "python"))

    async def run(command: str, check: bool = False) -> SimpleNamespace:
        return await _run_locally(command.replace("/usr/local/bin/restic", "/bin/sh"), check)

    assert await storage_operations.supports_storage_operation(SimpleNamespace(run=run), "restic") is starts


@pytest.mark.parametrize("python", ["missing", "import-error"])
def test_a_runner_that_cannot_start_fails_the_dispatch_while_it_can_report(node: Path, python: str) -> None:
    """F-98: a missing interpreter or a runner import that fails is the dispatch's failure, reported now; nothing is
    launched, no spec (credentials) is written and the staged copy goes"""
    import asyncio as aio

    _runner(node, "pass\n", at_import="import storage_module_that_is_not_there\n" if python == "import-error" else "")
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    spec = {"engine": "restic", "reporter": {"api_url": "https://api.example", "auth_token": "t0k", "resource": "backup"}}
    with pytest.raises(storage_operations.StorageOperationLaunchError, match="runner cannot start"):
        aio.run(storage_operations.start_storage_operation(
            _node_client(), sys.executable if python == "import-error" else str(node / "no-python"),
            OPERATION_ID, spec, retain_terminal_artifacts=False,
        ))
    storage_operations.report_launch_failure.assert_awaited_once()
    assert not Path(str(files.spec)).exists() and _staged(node) == []
