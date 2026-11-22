"""The validator runs only its own scripts on an executor, with the interpreter it names.

`root_dir` and `python_path` come from the executor's answer. Here they point at a directory holding scripts and an
interpreter wrapper that write marker files; the validator's own payloads write a marker of their own. The commands
run in a local shell, with the service directory and the trusted interpreter moved into the test's temp dir.
"""

import ast
import asyncio
import os
import re
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from services import executor_payload
from services.inspector_validation_service import InspectorValidationService
from services.task.checks.start_gpu_monitor import StartGPUMonitorCheck
from services.task.runner import SSHCommandResult
from services.verifyx_validation_service import VerifyXValidationService
from tests.helpers import build_context_config, build_state, make_context

pytestmark = pytest.mark.asyncio

EXECUTOR_SOURCES = Path(__file__).resolve().parents[2] / "executor" / "src"
EMPTY_ACCOUNT_FILES = "hotkey=\nclaim=\n"


def _marker_script(marker: Path, *, sleep: bool = False) -> str:
    # records its arguments, then (a monitor) stays up like the real one
    tail = "import time; time.sleep(60)\n" if sleep else ""
    return f"import sys; open({str(marker)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n{tail}"


@pytest.fixture
def node(tmp_path, monkeypatch):
    """An executor whose reported root_dir and python_path lead to marker-writing code, and a validator whose payloads
    write a marker of their own."""
    attacker = tmp_path / "volume"
    for name in ("gpus_utility.py", "inspector_executor.py", "verifyx_executor.py"):
        (attacker / "src").mkdir(parents=True, exist_ok=True)
        (attacker / "src" / name).write_text(_marker_script(tmp_path / "ATTACKER_EXECUTED"))
    wrapper = attacker / "pywrap"
    wrapper.write_text(f"#!/bin/sh\ntouch {tmp_path / 'INTERPRETER_EXECUTED'}\nexec {sys.executable} \"$@\"\n")
    wrapper.chmod(0o755)

    payloads = tmp_path / "validator-payloads"
    (payloads / "services").mkdir(parents=True)
    (payloads / "gpus_utility.py").write_text(_marker_script(tmp_path / "VALIDATOR_MONITOR", sleep=True))
    (payloads / "services" / "pull_lock.py").write_text("")
    (payloads / "inspector_executor.py").write_text(_marker_script(tmp_path / "VALIDATOR_INSPECTOR"))
    (payloads / "verifyx_executor.py").write_text(_marker_script(tmp_path / "VALIDATOR_VERIFYX"))
    (payloads / "decrypt_challenge.py").write_text(_marker_script(tmp_path / "VALIDATOR_MATRIX"))
    (tmp_path / "service").mkdir()
    monkeypatch.setattr(executor_payload, "PAYLOAD_DIR", payloads)
    monkeypatch.setattr(executor_payload, "SERVICE_DIR_PARENT", str(tmp_path / "service"))
    monkeypatch.setattr(executor_payload, "EXECUTOR_PYTHON", sys.executable)

    executor = ExecutorSSHInfo(
        uuid="node-1", address="203.0.113.7", port=8001, ssh_username="root", ssh_port=22,
        python_path=str(wrapper), root_dir=str(attacker), expected_node_hotkeys=["5Account"], expected_account_claim="c",
    )
    yield SimpleNamespace(tmp=tmp_path, executor=executor)
    # the monitor payload stays up like the real one: stop what the tests started
    out = subprocess.run(["pgrep", "-f", str(tmp_path / "service")], capture_output=True, text=True).stdout
    for pid in out.split():
        os.kill(int(pid), signal.SIGKILL)


class LocalRunner:
    """The validator's SSH runner, on a local shell; the /etc/liumd read answers with no identity files."""

    def __init__(self, cwd: Path):
        self.cwd, self.commands = cwd, []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):  # noqa: ASYNC109
        from datetime import UTC, datetime

        self.commands.append(cmd)
        if "/etc/liumd/" in cmd:
            code, out, err = 0, EMPTY_ACCOUNT_FILES, ""
        else:
            done = subprocess.run(["sh", "-c", cmd], cwd=self.cwd, capture_output=True, text=True, timeout=20)
            code, out, err = done.returncode, done.stdout, done.stderr
        now = datetime.now(UTC)
        return SSHCommandResult(
            command=cmd, command_id="c", exit_code=code, stdout=out, stderr=err,
            duration_ms=1, started_at=now, finished_at=now, success=code == 0,
        )


def _ctx(node, runner, **config):
    keypair = SimpleNamespace(sign=lambda data: b"\x01", ss58_address="5Validator")
    return make_context(
        executor=node.executor, runner=runner, miner_hotkey="5Account",
        config=build_context_config(validator_keypair=keypair, **config),
        state=build_state(),
    )


async def _until(path: Path, seconds: float = 10.0) -> bool:
    for _ in range(int(seconds * 20)):
        if path.exists() and path.read_text():
            return True
        await asyncio.sleep(0.05)
    return False


def _attacker_ran(node) -> bool:
    return (node.tmp / "ATTACKER_EXECUTED").exists() or (node.tmp / "INTERPRETER_EXECUTED").exists()


async def test_gpu_monitor_runs_the_validators_script_and_interpreter(node):
    runner = LocalRunner(node.tmp)
    assert (await StartGPUMonitorCheck().run(_ctx(node, runner))).passed
    assert await _until(node.tmp / "VALIDATOR_MONITOR")
    assert not _attacker_ran(node)
    assert not any(node.executor.python_path in cmd or node.executor.root_dir in cmd for cmd in runner.commands)


@pytest.mark.parametrize("run_dir", [None, ".verifyx-retry-0123456789abcdef"])
async def test_verifyx_does_not_execute_script_from_executor_root_dir(node, run_dir):
    command = VerifyXValidationService()._verifyx_command(
        node.executor, SimpleNamespace(seed=7, cipher_text="ab12"), "lib", run_dir=run_dir
    )
    done = subprocess.run(["sh", "-c", command], cwd=node.tmp, capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr
    assert (node.tmp / "VALIDATOR_VERIFYX").read_text() == "--seed 7 --cipher_text ab12\n"
    assert not _attacker_ran(node)
    assert not list((node.tmp / "service").iterdir())  # the script goes with the run
    if run_dir:
        assert not (node.tmp / run_dir).exists()


async def test_inspector_does_not_execute_script_from_executor_root_dir(node):
    service = InspectorValidationService.__new__(InspectorValidationService)
    command = service._interactive_command(node.executor)
    done = subprocess.run(["sh", "-c", command], cwd=node.tmp, capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr
    assert (node.tmp / "VALIDATOR_INSPECTOR").read_text() == "--interactive\n"
    assert not _attacker_ran(node)
    assert not list((node.tmp / "service").iterdir())


async def test_honest_gpu_monitor_still_starts(node):
    runner = LocalRunner(node.tmp)
    ctx = _ctx(node, runner)
    started = await StartGPUMonitorCheck().run(ctx)
    assert started.passed and started.event.reason_code == "MONITOR_STARTED"
    assert await _until(node.tmp / "VALIDATOR_MONITOR")
    args = (node.tmp / "VALIDATOR_MONITOR").read_text().split()
    assert args[0::2] == [
        "--program_id", "--signature", "--executor_id", "--validator_hotkey", "--compute_rest_app_url"
    ]
    assert args[5:] == ["node-1", "--validator_hotkey", "5Validator", "--compute_rest_app_url", "http://validator"]
    # the running monitor is found: no second copy
    again = await StartGPUMonitorCheck().run(ctx)
    assert again.passed and again.event.reason_code == "MONITOR_RUNNING"
    assert len((node.tmp / "VALIDATOR_MONITOR").read_text().splitlines()) == 1


async def test_a_running_legacy_monitor_is_replaced_not_accepted(node, monkeypatch):
    # an executor image of before this change ran the monitor from its own source tree: that copy is stopped and the
    # validator's own starts
    from services.task.checks import start_gpu_monitor

    legacy = node.tmp / "root" / "app" / "src" / "gpus_utility.py"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(_marker_script(node.tmp / "LEGACY_MONITOR", sleep=True))
    monkeypatch.setattr(start_gpu_monitor, "LEGACY_MONITOR", str(legacy))
    running = subprocess.Popen([sys.executable, str(legacy)])
    try:
        assert await _until(node.tmp / "LEGACY_MONITOR")
        result = await StartGPUMonitorCheck().run(_ctx(node, LocalRunner(node.tmp)))
        assert result.passed and result.event.reason_code == "MONITOR_STARTED"
        assert running.wait(timeout=10) is not None
        assert await _until(node.tmp / "VALIDATOR_MONITOR")
    finally:
        if running.poll() is None:
            running.kill()


async def test_the_monitor_check_stops_and_counts_only_processes_in_the_executors_mount_namespace(node):
    # the executor runs with pid: host: a same-named process of another container or the host is neither killed
    # nor taken for the validator's monitor (a second mount namespace needs root, so the commands are checked)
    runner = LocalRunner(node.tmp)
    assert (await StartGPUMonitorCheck().run(_ctx(node, runner))).passed
    calls = re.findall(r"\b(?:pkill|pgrep|ps)\b(?: -KILL)? ([^;]*)", " ".join(runner.commands))
    assert len(calls) == 6  # the legacy monitor and an earlier one of ours: TERM, wait, KILL each
    assert all(call.startswith(executor_payload.SAME_MOUNT_NS) for call in calls), calls


async def test_a_monitor_that_ignores_term_is_gone_before_its_replacement_starts(node, monkeypatch):
    from services.task.checks import start_gpu_monitor

    legacy = node.tmp / "root" / "app" / "src" / "gpus_utility.py"
    legacy.parent.mkdir(parents=True)
    # stuck like a monitor in a driver call: TERM does not end it
    legacy.write_text(
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        + _marker_script(node.tmp / "LEGACY_MONITOR", sleep=True)
    )
    monkeypatch.setattr(start_gpu_monitor, "LEGACY_MONITOR", str(legacy))
    running = subprocess.Popen([sys.executable, str(legacy)])
    try:
        assert await _until(node.tmp / "LEGACY_MONITOR")
        result = await StartGPUMonitorCheck().run(_ctx(node, LocalRunner(node.tmp)))
        assert result.passed and result.event.reason_code == "MONITOR_STARTED"
        assert running.poll() is not None  # already gone when the replacement was reported
    finally:
        if running.poll() is None:
            running.kill()


@pytest.mark.parametrize("change", ["an earlier build", "a record naming another process"])
async def test_a_monitor_is_reused_only_when_it_is_the_current_one_the_validator_started(node, change):
    runner = LocalRunner(node.tmp)
    assert (await StartGPUMonitorCheck().run(_ctx(node, runner))).event.reason_code == "MONITOR_STARTED"
    assert await _until(node.tmp / "VALIDATOR_MONITOR")
    (monitor_dir,) = [p for p in (node.tmp / "service").iterdir() if p.is_dir()]
    first = int((monitor_dir / ".run").read_text().split()[0])
    if change == "an earlier build":
        (executor_payload.PAYLOAD_DIR / "gpus_utility.py").write_text(_marker_script(node.tmp / "NEW_BUILD", sleep=True))
    else:
        pid, _ = (monitor_dir / ".run").read_text().split()
        (monitor_dir / ".run").write_text(f"{pid} 1\n")
    replaced = await StartGPUMonitorCheck().run(_ctx(node, runner))
    assert replaced.passed and replaced.event.reason_code == "MONITOR_STARTED"
    for _ in range(100):
        if not Path(f"/proc/{first}").exists() or "Z" in Path(f"/proc/{first}/stat").read_text().split(") ")[1][:1]:
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("the earlier monitor still runs")
    (current,) = [p for p in (node.tmp / "service").iterdir() if p.is_dir()]
    assert current != monitor_dir


async def test_two_overlapping_checks_start_one_monitor(node):
    class Recorder(LocalRunner):
        async def run(self, cmd, **_):
            self.commands.append(cmd)
            return SSHCommandResult(
                command=cmd, command_id="c", exit_code=1, stdout="", stderr="", duration_ms=1,
                started_at=datetime.now(UTC), finished_at=datetime.now(UTC), success=False,
            )

    recorder = Recorder(node.tmp)
    await StartGPUMonitorCheck().run(_ctx(node, recorder))
    (command,) = recorder.commands
    both = [subprocess.Popen(["sh", "-c", command], cwd=node.tmp, stdout=subprocess.PIPE, text=True) for _ in "ab"]
    outcomes = sorted(p.communicate(timeout=60)[0].split()[0] for p in both)
    assert outcomes == ["RUNNING", "STARTED"]
    assert await _until(node.tmp / "VALIDATOR_MONITOR")
    assert len([p for p in (node.tmp / "service").iterdir() if p.is_dir()]) == 1


async def test_a_monitor_that_does_not_stay_up_is_not_reported_started(node):
    (executor_payload.PAYLOAD_DIR / "gpus_utility.py").write_text("import sys; sys.exit(1)\n")
    result = await StartGPUMonitorCheck().run(_ctx(node, LocalRunner(node.tmp)))
    assert not result.passed and result.event.reason_code == "MONITOR_START_FAILED"


async def test_a_failed_or_timed_out_check_is_not_taken_for_no_monitor(node):
    class TimedOut(LocalRunner):
        # the look for a running monitor times out; a command that only starts one would succeed
        async def run(self, cmd, **kwargs):
            if "nohup" in cmd and "sha256sum" not in cmd:
                return await super().run(cmd, **kwargs)
            return SSHCommandResult(
                command=cmd, command_id="c", exit_code=-1, stdout="", stderr="timeout", duration_ms=1,
                started_at=datetime.now(UTC), finished_at=datetime.now(UTC), success=False,
            )

    result = await StartGPUMonitorCheck().run(_ctx(node, TimedOut(node.tmp)))
    assert not result.passed and result.event.reason_code == "MONITOR_START_FAILED"


def test_every_payload_is_found():
    # in the repository the executor sources; in the image the copies the Dockerfile takes from them
    for name in executor_payload.GPU_MONITOR + executor_payload.INSPECTOR + executor_payload.VERIFYX + executor_payload.MATRIX:
        assert (executor_payload.PAYLOAD_DIR / name).read_bytes() == (EXECUTOR_SOURCES / name).read_bytes(), name


async def test_one_shot_payload_runs_with_its_environment_and_is_removed(node):
    command = executor_payload.run_once("matrix", executor_payload.MATRIX, "--x 1", env="CUDA_VISIBLE_DEVICES=0 ")
    done = subprocess.run(["sh", "-c", command], cwd=node.tmp, capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr
    assert (node.tmp / "VALIDATOR_MATRIX").read_text() == "--x 1\n"
    assert not list((node.tmp / "service").iterdir())


async def test_one_shot_payload_is_removed_when_its_write_fails(node):
    command = executor_payload.run_once("matrix", executor_payload.MATRIX, "--x 1").replace("base64 -d", "false")
    done = subprocess.run(["sh", "-c", command], cwd=node.tmp, capture_output=True, text=True, timeout=20)
    assert done.returncode == 1
    assert not list((node.tmp / "service").iterdir())


def test_the_kill_pattern_matches_its_own_run_only():
    # not the shell that cleans up, and not another validation's run on the same host
    tag, other = executor_payload.new_run_tag(), executor_payload.new_run_tag()
    pattern = executor_payload.process_pattern("matrix", executor_payload.MATRIX, tag)
    shell = executor_payload.run_once("matrix", executor_payload.MATRIX, "--x 1", tag=tag)
    worker = f"{executor_payload.EXECUTOR_PYTHON} -E -s /root/app/.lium-validator-matrix.%s.Ab3dE9xZ/decrypt_challenge.py"
    lines = f"{worker % tag} --x 1\n{shell}\n{worker % other} --x 1\n"
    found = subprocess.run(["grep", "-Ec", pattern], input=lines, capture_output=True, text=True)
    assert found.stdout.strip() == "1"


async def test_a_run_is_written_under_its_tag_and_the_kill_takes_only_that_tag(node):
    from services.matrix_validation_service import ValidationService

    tag = executor_payload.new_run_tag()
    command = executor_payload.run_once("matrix", executor_payload.MATRIX, "--x 1", tag=tag)
    assert f".lium-validator-matrix.{tag}.XXXXXXXX" in command
    ssh = SimpleNamespace(commands=[])

    async def run(cmd, timeout=None, check=True):
        ssh.commands.append(cmd)
        return SimpleNamespace(exit_status=0)

    ssh.run = run
    await ValidationService._kill_remote_matmul(None, ssh, tag, {})
    # the launcher's process group and the entry script of this tag (test_matrix_cancel_before_start runs it)
    assert ssh.commands[0] == executor_payload.kill_runs("matrix", executor_payload.MATRIX, tag)
    assert tag in ssh.commands[0]


def test_no_validator_code_reads_the_executor_reported_paths():
    # python_path and root_dir come from the executor's answer: any new read of them must be reviewed, not added here
    src = Path(__file__).resolve().parents[1] / "src"
    reads = [
        f"{path.relative_to(src)}:{node.lineno}"
        for path in sorted(src.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(), str(path)))
        if isinstance(node, ast.Attribute) and node.attr in ("python_path", "root_dir")
    ]
    assert reads == []


def test_scripts_are_written_where_the_disk_reserve_holds():
    # a renter filling the host disk must not stop the validator writing its scripts
    reserve = (EXECUTOR_SOURCES.parent / "setup_disk_reserve.sh").read_text()
    assert f'workdir=$MNT/app-work" {executor_payload.SERVICE_DIR_PARENT}\n' in reserve


async def test_uploads_go_where_the_disk_reserve_holds_and_sweeps_them():
    import re
    from contextlib import asynccontextmanager

    from services.task.checks.upload_files import upload_validation_files_to_fresh_remote_dir

    sent = []

    @asynccontextmanager
    async def sftp_client():
        async def put(local, remote, recurse):
            sent.append(remote)

        yield SimpleNamespace(put=put)

    ctx = SimpleNamespace(state=SimpleNamespace(upload_local_dir="/local"), ssh=SimpleNamespace(start_sftp_client=sftp_client))
    remote = await upload_validation_files_to_fresh_remote_dir(ctx)
    assert sent == [remote] and re.fullmatch(r"/root/app/[0-9a-f]{32}", remote)  # the reserve janitor's pattern
