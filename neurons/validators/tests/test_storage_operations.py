import inspect
import json
import os
import shlex
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


def _runner(node: Path, body: str) -> None:
    (node / "payloads" / "storage_runner.py").write_text(body)


async def _run_locally(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
    done = subprocess.run(["sh", "-c", command.replace("/usr/bin/docker", "true")], input=input, capture_output=True, text=True)
    if check and done.returncode:
        raise RuntimeError(f"exit {done.returncode}")
    return SimpleNamespace(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)


def _dispatch(spec: dict | None = None, retain: bool = False) -> storage_operations.StorageOperationFiles:
    import asyncio as aio

    return aio.run(
        storage_operations.start_storage_operation(
            SimpleNamespace(run=_run_locally), sys.executable, OPERATION_ID, spec or {"engine": "restic"},
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
def test_an_online_run_leaves_its_done_marker_and_its_log_only_when_it_fails(node: Path, exit_code: int) -> None:
    _runner(node, f"print('working', flush=True)\nraise SystemExit({exit_code})\n")
    files = _dispatch()
    _wait_for(Path(str(files.done)))
    log = Path(str(files.log))
    # the log is the wrapper's last step, after the done marker
    if exit_code:
        import time

        deadline = time.time() + 10
        while "runner exited" not in log.read_text():
            assert time.time() < deadline
            time.sleep(0.05)
        assert log.read_text().splitlines() == ["working", "runner exited 3"]
    else:
        _wait_for(log, gone=True)
    assert _staged(node) == []   # the staged runner goes when it has exited
    assert not Path(str(files.status)).exists() and not Path(str(files.spec)).exists()
    assert not Path(str(files.pid)).exists()


def test_a_re_dispatch_after_an_online_run_finished_never_runs_it_again(node: Path) -> None:
    # the online wrapper removes its spec and status when the run ends; a re-dispatch (its terminal report not landed
    # yet) must still find the operation finished
    runs = node / "runs"
    _runner(node, f"open({str(runs)!r}, 'a').write('run\\n')\n")
    files = _dispatch()
    _wait_for(Path(str(files.done)))
    _wait_for(Path(str(files.pid)), gone=True)
    assert _dispatch() == files
    import time

    time.sleep(0.5)
    assert runs.read_text() == "run\n" and _staged(node) == []
    assert not Path(str(files.spec)).exists()   # the re-dispatch's credentials were never written


def test_a_re_dispatch_while_the_runner_works_launches_nothing_once_the_runner_removed_its_spec(node: Path) -> None:
    # the runner reads and removes its spec at once, before any PID of its own: the launcher recorded under the lock
    # is what a second dispatch finds, so it neither overwrites the spec nor starts a second wrapper
    runs, release = node / "runs", node / "release"
    _runner(node, (
        "import pathlib, sys, time\n"
        "spec = pathlib.Path(sys.argv[sys.argv.index('--spec') + 1]); spec.unlink()\n"
        f"open({str(runs)!r}, 'a').write('run\\n')\n"
        f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(0.05)\n"
    ))
    files = _dispatch()
    _wait_for(runs)
    _wait_for(Path(str(files.spec)), gone=True)
    try:
        assert _dispatch({"engine": "restic", "second": True}) == files
        assert not Path(str(files.spec)).exists() and len(_staged(node)) == 1   # only the first run's runner
    finally:
        release.write_text("")
    _wait_for(Path(str(files.done)))
    assert runs.read_text() == "run\n"


def test_two_dispatches_at_once_launch_one_wrapper(node: Path) -> None:
    # both pass staging together; the operation's lock lets one launch and the other find its launcher
    from concurrent.futures import ThreadPoolExecutor

    runs, release = node / "runs", node / "release"
    _runner(node, (
        "import pathlib, time\n"
        f"open({str(runs)!r}, 'a').write('run\\n')\n"
        f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(0.05)\n"
    ))
    try:
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda _: _dispatch(), range(4)))
        _wait_for(runs)
        import time

        time.sleep(0.5)
        assert runs.read_text() == "run\n" and len(_staged(node)) == 1
    finally:
        release.write_text("")
    assert len(set(results)) == 1
    _wait_for(Path(str(results[0].done)))


def test_a_dead_runs_leftovers_are_removed_and_the_operation_launched_again(node: Path) -> None:
    # a launcher that died without a result (the node rebooted): its PID no longer runs this operation's wrapper
    runs = node / "runs"
    _runner(node, f"open({str(runs)!r}, 'a').write('run\\n')\n")
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    Path(str(files.launcher)).write_text(f"{os.getpid()}\n")   # a live PID, but not this operation's wrapper
    Path(str(files.spec)).write_text("stale")
    _dispatch()
    _wait_for(Path(str(files.done)))
    assert runs.read_text() == "run\n"


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
    assert not Path(str(files.spec)).exists() and not Path(str(files.launcher)).exists()
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


# legacy_lock_uncreatable: an older validator's run (no lock file) on a node whose disk cannot create one
@pytest.mark.parametrize(
    "incumbent", ["launcher_before", "legacy_before", "launched_during_staging", "legacy_lock_uncreatable"]
)
def test_a_repeat_dispatch_whose_staging_fails_never_fails_the_running_operation(node: Path, incumbent: str) -> None:
    import asyncio as aio

    _runner(node, "raise SystemExit(0)\n")
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    runners: list[subprocess.Popen] = []

    def start_incumbent() -> None:
        runners.append(_legacy_runner(files))
        if incumbent == "legacy_lock_uncreatable":
            Path(str(files.lock)).mkdir()   # opening it for writing fails, as on a full disk
        elif incumbent != "legacy_before":   # this validator's launch: a launcher record, the PID as the runner
            Path(str(files.launcher)).write_text(Path(str(files.pid)).read_text())

    if incumbent != "launched_during_staging":
        start_incumbent()

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "printf \"%s\" \"$d\"" in command:   # the staging step: the disk is full
            if incumbent == "launched_during_staging":
                start_incumbent()
            raise OSError(28, "No space left on device")
        return await _run_locally(command, check, input)

    try:
        returned_files = aio.run(storage_operations.start_storage_operation(
            SimpleNamespace(run=run), sys.executable, OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        ))
        assert returned_files == files
        storage_operations.report_launch_failure.assert_not_awaited()
        assert all(r.poll() is None for r in runners)
    finally:
        for r in runners:
            r.kill()
            r.wait()


def test_a_dispatch_whose_staging_fails_with_no_run_reports_the_failure(node: Path) -> None:
    import asyncio as aio

    _runner(node, "raise SystemExit(0)\n")
    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "printf \"%s\" \"$d\"" in command:
            raise OSError(28, "No space left on device")
        return await _run_locally(command, check, input)

    with pytest.raises(OSError):
        aio.run(storage_operations.start_storage_operation(
            SimpleNamespace(run=run), sys.executable, OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        ))
    storage_operations.report_launch_failure.assert_awaited_once()


@pytest.mark.parametrize("retain_terminal_artifacts", [False, True])
def test_a_failed_dispatch_records_its_failure_before_reporting_so_no_later_dispatch_launches(
    node: Path, retain_terminal_artifacts: bool
) -> None:
    # the "no run" read and the failure's record are one step under the operation's lock: a dispatch that takes the lock
    # once the failed one released it finds that result and launches nothing, so the FAILED report is the operation's
    import asyncio as aio

    _runner(node, "raise SystemExit(0)\n")
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    launches: list[str] = []

    async def failing(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "printf \"%s\" \"$d\"" in command:
            raise OSError(28, "No space left on device")
        return await _run_locally(command, check, input)

    async def recording(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        result = await _run_locally(command, check, input)
        if "nohup" in command:
            launches.append((result.stdout or "").strip())
        return result

    with pytest.raises(OSError):
        aio.run(storage_operations.start_storage_operation(
            SimpleNamespace(run=failing), sys.executable, OPERATION_ID, {"engine": "restic"},
            retain_terminal_artifacts=retain_terminal_artifacts,
        ))
    storage_operations.report_launch_failure.assert_awaited_once()
    recorded = files.status if retain_terminal_artifacts else files.done
    assert Path(str(recorded)).exists() and not Path(str(files.launcher)).exists()

    returned_files = aio.run(storage_operations.start_storage_operation(
        SimpleNamespace(run=recording), sys.executable, OPERATION_ID, {"engine": "restic"},
        retain_terminal_artifacts=retain_terminal_artifacts,
    ))
    assert returned_files == files and launches == []
    assert not Path(str(files.launcher)).exists() and not Path(str(files.spec)).exists()
    storage_operations.report_launch_failure.assert_awaited_once()


def test_a_staging_failure_during_another_dispatchs_launch_handoff_waits_for_its_launcher(node: Path) -> None:
    # F-59: the other dispatch holds the launch lock and has started its wrapper, but neither its launcher nor the
    # runner's PID is recorded yet when this dispatch's staging fails: the recheck waits for the lock and finds the run
    import asyncio as aio
    import fcntl
    import threading

    _runner(node, "raise SystemExit(0)\n")
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    runners: list[subprocess.Popen] = []

    def finish_handoff(lock) -> None:
        import time

        time.sleep(0.5)
        runners.append(_legacy_runner(files))
        Path(str(files.launcher)).write_text(Path(str(files.pid)).read_text())
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

    async def run(command: str, check: bool = False, input: str | None = None) -> SimpleNamespace:
        if "printf \"%s\" \"$d\"" in command:
            lock = open(str(files.lock), "a")
            fcntl.flock(lock, fcntl.LOCK_EX)   # the other dispatch's launch, mid-handoff
            threading.Thread(target=finish_handoff, args=(lock,)).start()
            raise OSError(28, "No space left on device")
        return await _run_locally(command, check, input)

    try:
        returned_files = aio.run(storage_operations.start_storage_operation(
            SimpleNamespace(run=run), sys.executable, OPERATION_ID, {"engine": "restic"}, retain_terminal_artifacts=False
        ))
        assert returned_files == files
        storage_operations.report_launch_failure.assert_not_awaited()
    finally:
        for r in runners:
            r.kill()
            r.wait()


def test_a_run_that_ends_between_the_checks_is_still_found(node: Path) -> None:
    # the wrapper publishes its end as: status, PID record removed, done marker, status removed, exit. A stand-in grep
    # ends the run at the first liveness check, as if it finished between two reads: the later checks find it
    files = storage_operations.StorageOperationFiles.for_operation(OPERATION_ID)
    for path in (files.launcher, files.pid):
        Path(str(path)).write_text(f"{os.getpid()}\n")
    fake_bin, ended = node / "bin", node / "ended"
    fake_bin.mkdir()
    q = lambda path: shlex.quote(str(path))  # noqa: E731
    (fake_bin / "grep").write_text(
        "#!/bin/sh\n"
        f"if [ ! -e {q(ended)} ]; then : > {q(ended)}; printf '0\\n' > {q(files.status)}; rm -f {q(files.pid)}; "
        f": > {q(files.done)}; rm -f {q(files.status)}; cat > /dev/null; exit 1; fi\n"
        'exec /usr/bin/grep "$@"\n'
    )
    (fake_bin / "grep").chmod(0o755)
    check = f"if {storage_operations._incumbent_condition(files)}; then echo YES; fi"
    done = subprocess.run(["sh", "-c", check], capture_output=True, text=True, env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"})
    assert ended.exists() and done.stdout.strip() == "YES"
