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
    monkeypatch.setattr(storage_operations, "CANCEL_MARKER_DIRECTORY", tmp_path / "reserve")
    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    fake_docker, docker_log = _fake_docker(tmp_path, "false")  # stopped once signalled
    unrelated = subprocess.Popen(["sleep", "60"])
    (tmp_path / f"{OPERATION_ID}.pid").write_text(str(unrelated.pid))
    runner = _legacy_runner(tmp_path) if legacy else None
    try:
        await storage_operations.cancel_storage_operation(_local_ssh(fake_docker), _HeldRedis(), OPERATION_ID)
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
    monkeypatch.setattr(storage_operations, "CANCEL_MARKER_DIRECTORY", tmp_path / "reserve")
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
    monkeypatch.setattr(storage_operations, "CANCEL_MARKER_DIRECTORY", tmp_path / "reserve")
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
    monkeypatch.setattr(storage_operations, "CANCEL_MARKER_DIRECTORY", tmp_path / "reserve")
    runner = _legacy_runner(tmp_path) if runner_alive else None
    try:
        files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
        owned = await storage_operations._runner_alive(_local_ssh(fake_docker), files)
    finally:
        if runner:
            runner.kill()
            runner.wait()
    assert owned is runner_alive


class _HeldRedis:
    """Just the validator-side Redis calls the reporter record uses."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def sadd(self, *_: object) -> None:
        pass

    async def expire(self, *_: object) -> None:
        pass


REPORTER = {"api_url": "https://api.example", "auth_token": "t", "resource": "restore"}


async def _held(action: str = "restore") -> _HeldRedis:
    redis = _HeldRedis()
    await storage_operations.note_started_storage_helper(
        redis, "executor", OPERATION_ID, {"action": action, "reporter": REPORTER}
    )
    return redis


def _planted_node_spec() -> None:
    # the provider writes what it likes on its node: a spec aimed at its own URL must never be followed
    spec = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID).spec
    spec.write_text(json.dumps({"reporter": {**REPORTER, "api_url": "http://169.254.169.254/"}}))


@pytest.mark.asyncio
async def test_a_cancel_reports_through_the_reporter_this_validator_launched_with(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    monkeypatch.setattr(storage_operations, "CANCEL_MARKER_DIRECTORY", tmp_path / "reserve")
    report = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_cancelled", report)
    _planted_node_spec()

    async def run_locally(command: str, check: bool = False) -> SimpleNamespace:
        result = subprocess.run(["sh", "-c", command.replace("/usr/bin/docker", "true")], capture_output=True, text=True)
        return SimpleNamespace(exit_status=result.returncode, stdout=result.stdout, stderr=result.stderr)

    await storage_operations.cancel_storage_operation(SimpleNamespace(run=run_locally), await _held(), OPERATION_ID)

    report.assert_awaited_once_with(OPERATION_ID, {"action": "restore", "reporter": REPORTER})


@pytest.mark.asyncio
async def test_a_cancel_with_no_held_reporter_sends_nothing_even_with_a_node_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", tmp_path)
    monkeypatch.setattr(storage_operations, "CANCEL_MARKER_DIRECTORY", tmp_path / "reserve")
    report = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_operation_status", report)
    _planted_node_spec()

    async def run_locally(command: str, check: bool = False) -> SimpleNamespace:
        result = subprocess.run(["sh", "-c", command.replace("/usr/bin/docker", "true")], capture_output=True, text=True)
        return SimpleNamespace(exit_status=result.returncode, stdout=result.stdout, stderr=result.stderr)

    await storage_operations.cancel_storage_operation(SimpleNamespace(run=run_locally), _HeldRedis(), OPERATION_ID)

    report.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_repeated_dispatch_never_starts_a_second_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    # The first runner has read (and deleted) its spec but has no PID or helper yet: the state is STARTING.
    monkeypatch.setattr(storage_operations, "supports_storage_operation", AsyncMock(return_value=True))
    monkeypatch.setattr(storage_operations.asyncio, "sleep", AsyncMock())
    commands: list[str] = []

    async def run(command: str, check: bool = False, **_) -> SimpleNamespace:
        commands.append(command)
        stdout = ""
        status = 1 if command.startswith("test -e") else 0
        if "echo STARTING" in command:
            stdout = "STARTING\n"
        return SimpleNamespace(exit_status=status, stdout=stdout, stderr="")

    files = await storage_operations.start_storage_operation(
        SimpleNamespace(run=run, start_sftp_client=None), "python3", OPERATION_ID, {"engine": "restic"},
        retain_terminal_artifacts=False,
    )
    assert files == storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    probe = next(c for c in commands if c.startswith("pgrep -f"))
    assert f"[-]-spec {files.spec}" in probe
    assert not any("docker rm -f" in c or "nohup" in c for c in commands)  # never a second wrapper, the helper stays


@pytest.fixture
def node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in executor: the operation directory, the staging parent and the validator's runner payload (a script
    the test writes per case), with every remote command run by a local shell."""
    monkeypatch.setattr(storage_operations, "REMOTE_OPERATION_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path)))
    # the executor's reserve_data volume, outside the operation directory an executor restart wipes
    monkeypatch.setattr(storage_operations, "CANCEL_MARKER_DIRECTORY", storage_operations.PurePosixPath(str(tmp_path / "reserve")))
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


def _legacy_runner_for(files: storage_operations.StorageOperationFiles) -> subprocess.Popen:
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
    legacy_runner = _legacy_runner_for(files)
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


def test_a_dispatch_with_no_live_run_removes_the_stale_helper_then_writes_the_spec_then_starts_the_runner(
    node: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio as aio

    runs = node / "runs"
    _runner(node, f"open({str(runs)!r}, 'a').write('run\\n')\n")
    steps: list[str] = []

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        steps.extend(s for s, hit in (("helper", "docker rm -f" in command), ("launch", "nohup" in command)) if hit)
        return await _run_locally(command, check, input)

    class _Recording(_LocalSFTP):
        def open(self, path: str, mode: str):
            steps.append("spec")
            return super().open(path, mode)

    aio.run(storage_operations.start_storage_operation(
        SimpleNamespace(run=run, start_sftp_client=_Recording), sys.executable, OPERATION_ID, {"engine": "restic"},
        retain_terminal_artifacts=False,
    ))
    _wait_for(runs)
    assert steps == ["helper", "spec", "launch"]


def test_the_runner_start_check_never_imports_from_the_working_directory(node: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """a module beside the SSH session's working directory must not stand in for one the staged runner lacks"""
    cwd = node / "cwd"
    cwd.mkdir()
    (cwd / "planted_dependency.py").write_text("")
    monkeypatch.chdir(cwd)
    _runner(node, "pass\n", at_import="import planted_dependency\n")
    with pytest.raises(storage_operations.StorageOperationLaunchError, match="runner cannot start"):
        _dispatch()
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    assert not Path(str(files.spec)).exists() and _staged(node) == []




def _node(state: str, commands: list[str]):
    async def run(command: str) -> str:
        commands.append(command)
        if command.startswith("cat "):
            return '{"reporter": {"api_url": "http://169.254.169.254/", "auth_token": "x", "resource": "backup"}}'
        return state if "docker inspect" in command else ""

    return run


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "message"),
    [("STATUS:137\n", "killed"), ("STATUS:1\n", "exited with code 1"), ("STATUS:139\n", "code 139")],
)
async def test_an_exited_helper_with_its_job_open_is_reported_through_the_held_reporter(monkeypatch, state, message):
    report = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_operation_status", report)
    commands: list[str] = []

    assert await storage_operations.report_exited_detached_helper(
        _node(state, commands), await _held("backup"), str(OPERATION_ID)
    )
    assert report.await_args.args[1] == {"action": "backup", "reporter": REPORTER}
    assert report.await_args.kwargs["status"] == "FAILED"
    assert report.await_args.kwargs["stage"] == "BACKUP"
    assert message in report.await_args.kwargs["message"]
    assert not any(c.startswith("cat ") for c in commands)  # the node's spec is never read
    assert "docker rm -f" in commands[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["RUNNING\n", "STATUS:0\n", "STATUS:130\n"])
async def test_a_running_or_succeeded_helper_is_not_reported(monkeypatch, state):
    report = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_operation_status", report)

    assert not await storage_operations.report_exited_detached_helper(
        _node(state, []), await _held(), str(OPERATION_ID)
    )
    report.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_planted_node_spec_with_no_held_reporter_sends_nothing(monkeypatch):
    report = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_operation_status", report)

    assert not await storage_operations.report_exited_detached_helper(
        _node("STATUS:137\n", []), _HeldRedis(), str(OPERATION_ID)
    )
    report.assert_not_awaited()


@pytest.mark.asyncio
async def test_forget_matches_the_record_and_reporter_whatever_the_id_case():
    removed: list[str] = []
    redis = await _held()
    redis.smembers = AsyncMock(return_value={f"{OPERATION_ID}|1".encode()})
    redis.srem = AsyncMock(side_effect=lambda key, raw: removed.append(raw))
    redis.delete = AsyncMock(side_effect=lambda key: redis.values.pop(key, None))

    await storage_operations.forget_storage_helper(redis, "executor", str(OPERATION_ID).upper())

    assert removed == [f"{OPERATION_ID}|1"]
    assert redis.values == {}


@pytest.mark.parametrize("cancel_lands", ["before_launch", "before_the_wrapper_starts"])
def test_a_cancelled_operation_never_runs(node: Path, monkeypatch: pytest.MonkeyPatch, cancel_lands: str) -> None:
    """The backend's cancel and its dispatch are separate validator tasks: the cancel can finish on the node before
    the launch reaches it, or between the launch's checks and its wrapper starting. Either way the runner never runs,
    and neither the spec (the repository credentials) nor the staged runner stays."""
    import asyncio as aio

    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    ran = node / "ran"
    _runner(node, f"open({str(ran)!r}, 'w').close()\n")
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)

    async def cancel() -> None:
        await storage_operations.cancel_storage_operation(_node_client(), await _held(), OPERATION_ID)

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "nohup" in command and cancel_lands == "before_the_wrapper_starts":
            await cancel()
        return await _run_locally(command, check, input)

    async def dispatch() -> None:
        if cancel_lands == "before_launch":
            await cancel()
        await storage_operations.start_storage_operation(
            _node_client(run), sys.executable, OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        )

    aio.run(dispatch())
    for staged in (node / "service").iterdir():
        _wait_for(staged, gone=True)  # a launched wrapper removes its staged runner as it ends

    assert not ran.exists()
    assert not Path(str(files.spec)).exists()
    assert Path(str(files.cancel)).exists()  # the marker stays for any later dispatch of the operation


def test_a_cancel_survives_an_executor_restart_before_a_late_dispatch(node: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An executor restart wipes the container's own files, the operation directory with them. A cancel that
    finished before it still keeps a dispatch reaching the node afterwards from running the operation."""
    import asyncio as aio

    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    ran = node / "ran"
    _runner(node, f"open({str(ran)!r}, 'w').close()\n")
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)

    async def cancel_restart_then_dispatch() -> None:
        await storage_operations.cancel_storage_operation(_node_client(), await _held(), OPERATION_ID)
        for path in node.glob(f"{OPERATION_ID}.*"):  # the restart wipes the operation directory
            path.unlink()
        await storage_operations.start_storage_operation(
            _node_client(), sys.executable, OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        )

    aio.run(cancel_restart_then_dispatch())
    for staged in (node / "service").iterdir():
        _wait_for(staged, gone=True)  # a launched wrapper removes its staged runner as it ends

    assert not ran.exists()
    assert not Path(str(files.spec)).exists()
    assert Path(str(files.cancel)).exists()


@pytest.mark.parametrize("cancel_lands", ["before_launch", "before_the_wrapper_starts", "while_it_runs"])
def test_a_bootstrap_wait_on_a_cancelled_operation_ends_cancelled_not_failed(
    node: Path, monkeypatch: pytest.MonkeyPatch, cancel_lands: str
) -> None:
    """The bootstrap restore waits for its operation and reports FAILED on a launch error. A cancelled operation
    (never started, or stopped by its runner with exit 130) must end the wait as cancelled, so that report never
    overwrites the CANCELLED the cancel already sent."""
    import asyncio as aio

    monkeypatch.setattr(storage_operations, "_report_cancelled", AsyncMock(return_value=True))
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    # the real runner sees the marker through its cancellation probe and exits 130
    _runner(node, f"import os, sys, time\nwhile not os.path.exists({str(files.cancel)!r}): time.sleep(0.05)\nsys.exit(130)\n")

    async def cancel() -> None:
        await storage_operations.cancel_storage_operation(_node_client(), await _held(), OPERATION_ID)

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "nohup" in command and cancel_lands == "before_the_wrapper_starts":
            await cancel()
        return await _run_locally(command, check, input)

    async def bootstrap() -> None:
        if cancel_lands == "before_launch":
            await cancel()
        launched = await storage_operations.start_storage_operation(
            _node_client(run), sys.executable, OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=True
        )
        if cancel_lands == "while_it_runs":
            Path(str(files.cancel)).parent.mkdir(exist_ok=True)  # the cancel creates the marker directory
            Path(str(files.cancel)).touch()  # the marker a cancel leaves while the runner works
        await storage_operations.wait_for_storage_operation(
            _node_client(), launched, poll_interval_seconds=0.05, launch_grace_polls=4
        )

    with pytest.raises(storage_operations.StorageOperationCancelled):
        aio.run(bootstrap())
